"""Footprint accounting across precision and tuple boundaries."""

from __future__ import annotations

import isl
import pytest

from tests.nvidia.cuda.fixtures.placed.hand_checked import (
    InvariantReuse,
)
from tilefoundry.analysis import (
    AnalysisPrecision,
)
from tilefoundry.analysis.footprint import ReachedAddresses, footprint_of, merged
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.types import DType


@pytest.mark.parametrize("counted_precision", tuple(AnalysisPrecision))
@pytest.mark.parametrize(
    "missing_precision", (AnalysisPrecision.LOWER_BOUND, AnalysisPrecision.UNKNOWN)
)
def test_uncounted_boundary_marks_the_footprint_incomplete(
    counted_precision: AnalysisPrecision,
    missing_precision: AnalysisPrecision,
) -> None:
    buffer = InvariantReuse.entry_function().params[0]
    uncounted = ReachedAddresses(
        target=SSAValue(buffer),
        output_index=0,
        dtype=None,
        reached=None,
        precision=missing_precision,
    )
    counted = ReachedAddresses(
        SSAValue(buffer),
        0,
        DType.bf16,
        isl.set("{ [i] : 0 <= i < 4 }"),
        counted_precision,
    )

    footprint = footprint_of(
        merged((uncounted, counted)),
    )

    assert {value: held for value, _ownership, held in footprint.buffers}[SSAValue(buffer)].of(
        "gmem"
    ).total == 8
    expected = (
        missing_precision
        if counted_precision in (AnalysisPrecision.EXACT, missing_precision)
        else AnalysisPrecision.UNKNOWN
    )
    assert footprint.precision is expected
    assert footprint_of((uncounted,)).precision is missing_precision


@pytest.mark.parametrize("left", tuple(AnalysisPrecision))
@pytest.mark.parametrize("right", tuple(AnalysisPrecision))
def test_tuple_output_boundaries_add_instead_of_union(
    left: AnalysisPrecision,
    right: AnalysisPrecision,
) -> None:
    buffer = InvariantReuse.entry_function().params[0]
    addresses = isl.set("{ [i] : 0 <= i < 4 }")
    reached = tuple(
        ReachedAddresses(
            target=SSAValue(buffer),
            output_index=index,
            dtype=DType.bf16,
            reached=addresses,
            precision=precision,
        )
        for index, precision in enumerate((left, right))
    )

    distinct = merged(reached)
    footprint = footprint_of(
        distinct,
    )
    by_value = {value: held for value, _ownership, held in footprint.buffers}
    gmem = by_value[SSAValue(buffer)].of("gmem")

    assert len(distinct) == 2
    assert gmem is not None and gmem.total == 2 * 4 * 2
    expected = (
        right
        if left is AnalysisPrecision.EXACT
        else left
        if right is AnalysisPrecision.EXACT or left is right
        else AnalysisPrecision.UNKNOWN
    )
    assert footprint.precision is expected
