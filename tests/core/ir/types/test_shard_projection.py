"""Logical ownership composes without choosing a physical storage layout."""

from itertools import product

import pytest

from tilefoundry.ir.types import Layout
from tilefoundry.ir.types.shard_layout import Broadcast, Partial, Split
from tilefoundry.ir.types.shard_projection import factor_domain, logical_projection, project_domain


def _indices(mapping, offset):
    return [offset + sum(i * s for i, s in zip(c, mapping.strides, strict=True))
            for c in product(*(range(n) for n in mapping.shape))]


@pytest.mark.parametrize("shared_axis", (False, True), ids=("different_axes", "same_axis"))
def test_staged_projection_composes_to_the_same_global_elements(shared_axis):
    """Fix one mesh coordinate, then the next, retaining the original index map."""
    shape = (12, 4) if shared_axis else (8, 6)
    attrs = (Split(0), Split(0 if shared_axis else 1))
    for row, column in product(range(2), range(3)):
        direct, offset = logical_projection(shape, shape, (2, 3), attrs, (row, column))
        first, start = logical_projection(shape, shape, (2, 3), attrs, (row, None))
        local = (shape[0] // 2, shape[1])
        second = project_domain(first.shape, local, (3,), (attrs[1],), (column,))
        composed, end = second.apply(first)
        assert _indices(composed, start + end) == _indices(direct, offset)
        rows = range(row * 6 + column * 2, row * 6 + column * 2 + 2) if shared_axis else range(row * 4, row * 4 + 4)
        columns = range(4) if shared_axis else range(column * 2, column * 2 + 2)
        assert _indices(direct, offset) == [i * shape[1] + j for i in rows for j in columns]


def test_one_projection_can_realize_distinct_storage_maps():
    """Choosing physical strides changes addresses, not the projected domain."""
    projection = project_domain((8, 6), (8, 6), (2,), (Split(0),), (1,))
    row, row_offset = projection.apply(Layout((8, 6), (6, 1)))
    column, column_offset = projection.apply(Layout((8, 6), (1, 8)))
    assert projection.local_shape == (4, 6)
    assert (row_offset, column_offset) == (24, 4)
    assert _indices(row, row_offset) == [i * 6 + j for i in range(4, 8) for j in range(6)]
    assert _indices(column, column_offset) == [i + 8 * j for i in range(4, 8) for j in range(6)]


def test_factorization_keeps_reduction_state_on_its_mesh_axis():
    domain, attrs = factor_domain((8, 6), (2, 2), (Partial("sum"), Split(0)))
    assert domain == (2, 4, 6)
    assert attrs == (Partial("sum"), Split(0))
    partial = logical_projection(domain, (8, 6), (2, 2), attrs, (1, 1))
    replicated = logical_projection(domain, (8, 6), (2, 2), (Broadcast(), Split(0)), (0, 1))
    assert partial == replicated
