"""Which reading supplies a child-Module call's constants, and from where.

A composed root holds no constants: each reached child fills its own from the
reading it was loaded into. The loading and execution checks in this file are
rewritten in M4 for the lazy first-use resource contract.

No GPU: a second device is spelled ``meta``, which needs no card.
"""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.shapes.scaled_modules import FusedScaledParent, ScaledChild
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.target import CudaTarget


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


def test_reading_a_resource_without_a_child_weight_is_refused() -> None:
    loaded = FusedScaledParent.load(_Weights({}))
    with pytest.raises(KeyError, match="missing declared weight 'w'"):
        evaluate(loaded.fused, torch.ones(4))


def test_a_child_weight_elsewhere_is_refused_before_anything_runs() -> None:
    elsewhere = torch.ones(4, device="meta")
    reading = FusedScaledParent.load(_Weights({"scaled.w": elsewhere}))

    with pytest.raises(ValueError, match="weight 'w' is on meta"):
        evaluate(reading.fused, torch.ones(4))
    assert str(reading.resource.subtree("scaled").load("w").device) == "meta"


def test_a_converter_may_call_a_child_staged_before_it(tmp_path) -> None:
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
