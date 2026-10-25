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

import torch

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate

LOOPS = Path(__file__).parents[1] / "fixtures" / "ir" / "loop_regions.py"


def test_range_start_step():
    """The three-argument `range` surface.

    The three-argument `range` surface: a scalar loop var over a DimVar extent
    with a non-unit start and step, none of which is unrolled.
    """
    range_start_step = load_authored_ir(f"{LOOPS}:RangeStartStep").entry_function()
    n = 7
    x = torch.arange(n, dtype=torch.float32)
    out = evaluate(range_start_step, x)
    assert torch.allclose(out.reshape(()), x[1:n:2].sum()), (n, out)


def test_nested_loop_region_outer_carry_in_inner():
    nested_sum = load_authored_ir(f"{LOOPS}:NestedSum").entry_function()
    x = torch.randn(4, 5)
    out = evaluate(nested_sum, x)
    assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), out


def test_dim_expression_extent():
    dim_expr_half_sum = load_authored_ir(f"{LOOPS}:DimExprHalfSum").entry_function()
    n = 8
    x = torch.arange(n, dtype=torch.float32)
    out = evaluate(dim_expr_half_sum, x)
    assert torch.allclose(out.reshape(()), x[: n // 2].sum()), (n, out)


def test_interleaved_partial_reduction_equals_flat():
    interleaved_two_partial_sum = load_authored_ir(
        f"{LOOPS}:InterleavedTwoPartialSum"
    ).entry_function()
    n = 6
    x = torch.randn(n)
    out = evaluate(interleaved_two_partial_sum, x)
    assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), (n, out)
