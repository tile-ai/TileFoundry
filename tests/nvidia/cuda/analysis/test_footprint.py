"""Footprint accounting across precision and tuple boundaries."""

from __future__ import annotations

from pathlib import Path

import isl
import pytest

from tilefoundry import analysis, cli
from tilefoundry.ir import core, expr

INVARIANT_REUSE = f"{Path(__file__).parents[1] / 'fixtures/placed/hand_checked.py'}:InvariantReuse"


@pytest.mark.parametrize("counted_precision", tuple(analysis.Precision))
@pytest.mark.parametrize(
    "missing_precision", (analysis.Precision.LOWER_BOUND, analysis.Precision.UNKNOWN)
)
def test_uncounted_boundary_lowers_precision(
    counted_precision: analysis.Precision,
    missing_precision: analysis.Precision,
) -> None:
    buffer = cli.load_authored_ir(INVARIANT_REUSE).entry_function().params[0]
    uncounted = analysis.ReachedAddresses(
        target=core.SSAValue(buffer),
        output_index=0,
        dtype=None,
        reached=None,
        precision=missing_precision,
    )
    counted = analysis.ReachedAddresses(
        core.SSAValue(buffer),
        0,
        expr.DType.bf16,
        isl.set("{ [i] : 0 <= i < 4 }"),
        counted_precision,
    )

    footprint = analysis.footprint_of(
        analysis.merged((uncounted, counted)),
    )

    assert {value: held for value, _ownership, held in footprint.buffers}[core.SSAValue(buffer)].of(
        "gmem"
    ).total == 8
    expected = (
        missing_precision
        if counted_precision in (analysis.Precision.EXACT, missing_precision)
        else analysis.Precision.UNKNOWN
    )
    assert footprint.precision is expected
    assert analysis.footprint_of((uncounted,)).precision is missing_precision


@pytest.mark.parametrize("left", tuple(analysis.Precision))
@pytest.mark.parametrize("right", tuple(analysis.Precision))
def test_tuple_output_footprints_add(
    left: analysis.Precision,
    right: analysis.Precision,
) -> None:
    buffer = cli.load_authored_ir(INVARIANT_REUSE).entry_function().params[0]
    addresses = isl.set("{ [i] : 0 <= i < 4 }")
    reached = tuple(
        analysis.ReachedAddresses(
            target=core.SSAValue(buffer),
            output_index=index,
            dtype=expr.DType.bf16,
            reached=addresses,
            precision=precision,
        )
        for index, precision in enumerate((left, right))
    )

    distinct = analysis.merged(reached)
    footprint = analysis.footprint_of(
        distinct,
    )
    by_value = {value: held for value, _ownership, held in footprint.buffers}
    gmem = by_value[core.SSAValue(buffer)].of("gmem")

    assert len(distinct) == 2
    assert gmem is not None and gmem.total == 2 * 4 * 2
    expected = (
        right
        if left is analysis.Precision.EXACT
        else left
        if right is analysis.Precision.EXACT or left is right
        else analysis.Precision.UNKNOWN
    )
    assert footprint.precision is expected
