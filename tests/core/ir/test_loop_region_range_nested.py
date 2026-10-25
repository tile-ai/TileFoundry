"""Define test grid region range nested behavior.

End-to-end (parse + evaluate) tests for the unified ``range`` / ``tile``
loop surface, nested LoopRegions, and dim-expression loop bounds.

``range`` and ``tile`` share one loop domain ``(start, extent, step)`` and lower
to the same ``LoopRegion``; they differ only in the loop-var binding
(``range`` → scalar, two-arg ``tile`` → slice). Neither is unrolled. Nested
``for`` loops produce nested LoopRegions, and loop bounds accept dim expressions
(e.g. ``C // N``) resolved at evaluate time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.evaluator import evaluate

LOOPS = Path(__file__).parents[1] / "fixtures" / "ir" / "loop_regions.py"


class LoopRegionCase(Case):
    """A loop-region program evaluated against its torch reduction."""


@case
class RangeStartStep(LoopRegionCase):
    """The three-argument `range` surface.

    The three-argument `range` surface: a scalar loop var over a DimVar extent
    with a non-unit start and step, none of which is unrolled.
    """

    path: Path = LOOPS
    name: str | None = "RangeStartStep"

    def guard(self, state, problems):
        range_start_step = state.entry_function()
        n = 7
        x = torch.arange(n, dtype=torch.float32)
        out = evaluate(range_start_step, x)
        assert torch.allclose(out.reshape(()), x[1:n:2].sum()), (n, out)


@case
class OuterCarryInInnerLoop(LoopRegionCase):
    path: Path = LOOPS
    name: str | None = "NestedSum"

    def guard(self, state, problems):
        nested_sum = state.entry_function()
        x = torch.randn(4, 5)
        out = evaluate(nested_sum, x)
        assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), out


@case
class DimExpressionExtent(LoopRegionCase):
    path: Path = LOOPS
    name: str | None = "DimExprHalfSum"

    def guard(self, state, problems):
        dim_expr_half_sum = state.entry_function()
        n = 8
        x = torch.arange(n, dtype=torch.float32)
        out = evaluate(dim_expr_half_sum, x)
        assert torch.allclose(out.reshape(()), x[: n // 2].sum()), (n, out)


@case
class InterleavedPartialReductionEqualsFlat(LoopRegionCase):
    path: Path = LOOPS
    name: str | None = "InterleavedTwoPartialSum"

    def guard(self, state, problems):
        interleaved_two_partial_sum = state.entry_function()
        n = 6
        x = torch.randn(n)
        out = evaluate(interleaved_two_partial_sum, x)
        assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), (n, out)


CASES = special(LoopRegionCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_loop_region_case(case: LoopRegionCase) -> None:
    check_case(case)
