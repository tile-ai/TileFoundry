"""Evaluator core mechanics.

Evaluator core mechanics: function-call binding, SSA identity, loop carry and
scalar-constant materialization.

Op-level value oracles live in the model References, which run whole decoders
through this same walker; what a real model's shape cannot make visible is kept
here on small parsed programs -- two params that are structurally equal, and a
carry whose init comes from the IR rather than from a first iteration.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry_nvidia import CudaTarget

_DEV = "cpu"
SCALED_MODULES = Path(__file__).parents[1] / "fixtures" / "shapes" / "scaled_modules.py"


class _Weights:
    """A resource whose subtree is the dotted prefix, like a checkpoint's."""

    def __init__(self, values: dict) -> None:
        self.values = values

    def load(self, name: str):
        return self.values[name]

    def subtree(self, name: str) -> "_Weights":
        prefix = f"{name}."
        return _Weights(
            {k[len(prefix) :]: v for k, v in self.values.items() if k.startswith(prefix)}
        )


def test_a_child_module_call_runs_against_that_child_reading() -> None:
    x = torch.arange(4, dtype=torch.float32)
    w = torch.tensor([2.0, 3.0, 4.0, 5.0])
    FusedScaledParent = load_authored_ir(f"{SCALED_MODULES}:FusedScaledParent")
    reading = FusedScaledParent.load(_Weights({"scaled.w": w}))

    assert torch.equal(evaluate(reading.fused, x), x * w)


def test_one_module_read_twice_yields_two_independent_readings() -> None:
    x = torch.ones(4)
    FusedScaledParent = load_authored_ir(f"{SCALED_MODULES}:FusedScaledParent")
    first = FusedScaledParent.load(_Weights({"scaled.w": torch.full((4,), 2.0)}))
    second = FusedScaledParent.load(_Weights({"scaled.w": torch.full((4,), 5.0)}))

    assert torch.equal(evaluate(second.fused, x), torch.full((4,), 5.0))
    assert torch.equal(evaluate(first.fused, x), torch.full((4,), 2.0))


def test_two_bindings_of_one_child_read_their_own_constants() -> None:
    x = torch.ones(4)
    PairedScaledParent = load_authored_ir(f"{SCALED_MODULES}:PairedScaledParent")
    reading = PairedScaledParent.load(
        _Weights({"left.w": torch.full((4,), 2.0), "right.w": torch.full((4,), 7.0)})
    )

    assert torch.equal(evaluate(reading.both, x), torch.full((4,), 9.0))


def test_a_child_call_inside_a_loop_keeps_its_reading_on_every_trip() -> None:
    ScaledChild = load_authored_ir(f"{SCALED_MODULES}:ScaledChild")

    @module(entry="looped", target=CudaTarget("nvidia.h200_sxm"))
    class _Looped:
        scaled = ScaledChild

        @func
        def looped(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            acc = x
            for _ in range(3):
                acc = scaled(acc)  # noqa: F821
            return acc

    x = torch.ones(4)
    w = torch.full((4,), 2.0)
    reading = _Looped.load(_Weights({"scaled.w": w}))

    assert torch.equal(evaluate(reading.looped, x), x * w * w * w)


def test_a_variant_body_reaches_its_child_the_same_way() -> None:
    DynamicScaledChild = load_authored_ir(f"{SCALED_MODULES}:DynamicScaledChild")
    _N_EVAL = load_namespace(str(SCALED_MODULES))[0]["EVALUATOR_N"]

    @module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
    class _Dispatch:
        scaled = DynamicScaledChild

        @func
        def dispatch(x: Tensor[(_N_EVAL,), "f32"]) -> Tensor[(_N_EVAL,), "f32"]:
            pass

        @dispatch.specialize(RangePattern("N_eval", 1, 7))
        def dynamic_variant(x: Tensor[(_N_EVAL,), "f32"]) -> Tensor[(_N_EVAL,), "f32"]:
            return scaled(x)  # noqa: F821

    child = _Dispatch.children["scaled"]
    (variant,) = _Dispatch.entry_function().variants
    assert variant.body.target is child
    assert len(variant.body.args) == 1
    assert [p.is_const for p in variant.body.target.elements[0].entry_function().params] == [False, True]
