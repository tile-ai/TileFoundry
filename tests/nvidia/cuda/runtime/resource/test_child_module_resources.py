"""Which reading supplies a child-Module call's constants, and from where.

A composed root holds no constants: each reached child fills its own from the
reading it was loaded into. The loading and execution checks in this file are
rewritten in M4 for the lazy first-use resource contract.

No GPU: a second device is spelled ``meta``, which needs no card.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.dsl import *
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.visitor import ExprCloner
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.runtime.resource import DictResource
from tilefoundry_nvidia import CudaTarget

SCALED_MODULES = Path(__file__).parents[2] / "fixtures" / "shapes" / "scaled_modules.py"


class _Weights:
    """A resource whose subtree is the dotted prefix, like a checkpoint's."""

    def __init__(self, values: dict) -> None:
        self.values = values

    def load(self, name: str):
        return self.values[name]

    def load_group(self, name: str):
        return None

    def subtree(self, name: str) -> "_Weights":
        prefix = f"{name}."
        return _Weights(
            {k[len(prefix):]: v for k, v in self.values.items() if k.startswith(prefix)}
        )


class _LongerLoop(ExprCloner):
    def visit_LoopRegion(self, loop: LoopRegion, ctx):
        return replace(self.default_visit(loop, ctx), extent=4)


def test_reading_a_resource_without_a_child_weight_is_refused() -> None:
    loaded = load_authored_ir(f"{SCALED_MODULES}:FusedScaledParent").load(_Weights({}))
    with pytest.raises(KeyError, match="missing declared weight 'w'"):
        evaluate(loaded.fused, torch.ones(4))


def test_a_child_weight_elsewhere_is_refused_before_anything_runs() -> None:
    elsewhere = torch.ones(4, device="meta")
    FusedScaledParent = load_authored_ir(f"{SCALED_MODULES}:FusedScaledParent")
    reading = FusedScaledParent.load(_Weights({"scaled.w": elsewhere}))

    with pytest.raises(ValueError, match="weight 'w' is on meta"):
        evaluate(reading.fused, torch.ones(4))
    assert str(reading.resource.subtree("scaled").load("w").device) == "meta"


def test_a_converter_may_call_a_child_staged_before_it(tmp_path) -> None:
    ScaledChild = load_authored_ir(f"{SCALED_MODULES}:ScaledChild")

    @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
    class _Converted:
        scaled = ScaledChild

        @func
        def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.add(x, w)

        @run.converter("w")
        def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return scaled(w)  # noqa: F821

    raw = _Weights({"w": torch.full((4,), 3.0), "scaled.w": torch.full((4,), 2.0)})
    _Converted.prepare(raw, str(tmp_path), device="cpu")

    from safetensors.torch import load_file  # noqa: PLC0415 — optional runtime dep

    prepared = load_file(str(tmp_path / "model-00001-of-00001.safetensors"))
    assert torch.equal(prepared["w"], torch.full((4,), 6.0))
    assert torch.equal(prepared["scaled.w"], torch.full((4,), 2.0))


def test_preparation_stages_on_the_device_it_was_given() -> None:
    """A converter runs on the requested device, and what it produced stays there."""

    @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
    class _Doubled:
        @func
        def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.add(x, w)

        @run.converter("w")
        def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.mul(w, w)

    staged: dict[str, object] = {}
    reading = _Doubled._prepare_into(
        _Weights({"w": torch.full((4,), 3.0)}), "", staged, "meta"
    )

    assert str(staged["w"].device) == "meta"
    assert str(reading.resource.load("w").device) == "meta"


def test_nested_same_name_entries_load_weights_from_the_current_instance() -> None:
    root = load_authored_ir(f"{SCALED_MODULES}:NestedScaledRoot")
    middle = root.children["middle"].elements[0]
    leaf = middle.children["scaled"].elements[0]
    for owner in (root, middle, leaf):
        assert owner.entry_function() is owner.lookup("run")
    assert len({id(owner.entry_function()) for owner in (root, middle, leaf)}) == 3
    loaded = root.load(DictResource({"middle.scaled.w": torch.full((4,), 2.0)}))
    assert torch.equal(loaded.run(torch.ones(4)), torch.full((4,), 2.0))


def test_indexed_instances_share_definitions_and_read_their_own_weights() -> None:
    scaled_modules = load_namespace(str(SCALED_MODULES))[0]
    ScaledChild = scaled_modules["ScaledChild"]
    root = scaled_modules["StackedScaledParent"]
    loaded = root.load(DictResource({
        f"layers.{index}.w": torch.full((4,), float(index + 1)) for index in range(3)
    }))
    assert isinstance(loaded.layers, tuple)
    assert [child.path for child in loaded.layers] == ["layers.0", "layers.1", "layers.2"]
    assert all(child.module.entry_function() is ScaledChild.entry_function() for child in loaded.layers)
    assert torch.equal(loaded.run(torch.ones(4)), torch.full((4,), 6.0))


def test_two_slots_of_one_definition_keep_their_own_weights() -> None:
    scaled_modules = load_namespace(str(SCALED_MODULES))[0]
    ScaledChild = scaled_modules["ScaledChild"]
    loaded = scaled_modules["PairedScaledParent"].load(DictResource({
        "left.w": torch.full((4,), 2.0), "right.w": torch.full((4,), 3.0),
    }))
    assert loaded.left.module is loaded.right.module is ScaledChild
    assert torch.equal(loaded.both(torch.ones(4)), torch.full((4,), 5.0))


def test_missing_indexed_weight_reports_the_instance_path() -> None:
    StackedScaledParent = load_authored_ir(f"{SCALED_MODULES}:StackedScaledParent")
    loaded = StackedScaledParent.load(DictResource({"layers.0.w": torch.ones(4)}))
    with pytest.raises(KeyError, match="missing declared weight 'w'.*layers.1.w"):
        loaded.run(torch.ones(4))


def test_runtime_index_checks_include_the_slot_index_and_count() -> None:
    StackedScaledParent = load_authored_ir(f"{SCALED_MODULES}:StackedScaledParent")
    function = StackedScaledParent.entry_function()
    function = replace(function, body=_LongerLoop().visit(function.body))
    root = replace(StackedScaledParent, functions=(function,))
    loaded = root.load(DictResource({f"layers.{i}.w": torch.ones(4) for i in range(3)}))
    with pytest.raises(EvalError, match="slot 'layers': index 3 out of range for N=3"):
        loaded.run(torch.ones(4))


@pytest.mark.parametrize("key,count,indexed", [("wrong", 3, True), ("layers", 2, True), ("layers", 1, False)])
def test_runtime_twins_match_slot_keys_indexing_and_element_count(key, count, indexed) -> None:
    scaled_modules = load_namespace(str(SCALED_MODULES))[0]
    ScaledChild = scaled_modules["ScaledChild"]
    StackedScaledParent = scaled_modules["StackedScaledParent"]

    @runtime_module(ScaledChild)
    class ChildTwin:
        @runtime_func
        def run(self, x, w):
            return x * w

    children = ModuleList(ChildTwin, count) if indexed else ChildTwin
    namespace = {key: children, "run": runtime_func(lambda self, x: x)}
    with pytest.raises(TypeError, match="children|layers|indexed|count"):
        runtime_module(StackedScaledParent)(type("WrongTwin", (), namespace))
