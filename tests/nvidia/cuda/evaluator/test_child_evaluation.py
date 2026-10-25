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

import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.evaluator import evaluate

_DEV = "cpu"
SCALED_MODULES = Path(__file__).parents[1] / "fixtures" / "shapes" / "scaled_modules.py"
EVALUATOR = Path(__file__).parents[1] / "fixtures" / "evaluator"


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


class ChildReadingCase(Case):
    """A composed program whose child calls read the constants their own reading holds."""


@case
class FusedScaledParent(ChildReadingCase):
    path: Path = SCALED_MODULES
    name: str | None = "FusedScaledParent"

    def guard(self, state, problems):
        problems.run("runs against the child reading", self.runs_against_the_child_reading, state)
        problems.run("two independent readings", self.two_independent_readings, state)

    def runs_against_the_child_reading(self, state):
        x = torch.arange(4, dtype=torch.float32)
        w = torch.tensor([2.0, 3.0, 4.0, 5.0])
        reading = state.load(_Weights({"scaled.w": w}))

        assert torch.equal(evaluate(reading.fused, x), x * w)

    def two_independent_readings(self, state):
        x = torch.ones(4)
        first = state.load(_Weights({"scaled.w": torch.full((4,), 2.0)}))
        second = state.load(_Weights({"scaled.w": torch.full((4,), 5.0)}))

        assert torch.equal(evaluate(second.fused, x), torch.full((4,), 5.0))
        assert torch.equal(evaluate(first.fused, x), torch.full((4,), 2.0))


@case
class TwoBindingsOfOneChild(ChildReadingCase):
    path: Path = SCALED_MODULES
    name: str | None = "PairedScaledParent"

    def guard(self, state, problems):
        x = torch.ones(4)
        reading = state.load(
            _Weights({"left.w": torch.full((4,), 2.0), "right.w": torch.full((4,), 7.0)})
        )

        assert torch.equal(evaluate(reading.both, x), torch.full((4,), 9.0))


@case
class LoopedChildCall(ChildReadingCase):
    path: Path = EVALUATOR / "looped_child_call.py"
    name: str | None = "_Looped"

    def guard(self, state, problems):
        x = torch.ones(4)
        w = torch.full((4,), 2.0)
        reading = state.load(_Weights({"scaled.w": w}))

        assert torch.equal(evaluate(reading.looped, x), x * w * w * w)


@case
class DispatchVariantChildCall(ChildReadingCase):
    path: Path = EVALUATOR / "dispatch_variant_child_call.py"
    name: str | None = "_Dispatch"

    def guard(self, state, problems):
        child = state.children["scaled"]
        (variant,) = state.entry_function().variants
        assert variant.body.target is child
        assert len(variant.body.args) == 1
        assert [p.is_const for p in variant.body.target.elements[0].entry_function().params] == [
            False,
            True,
        ]


CASES = special(ChildReadingCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_child_reading_case(case: ChildReadingCase) -> None:
    check_case(case)
