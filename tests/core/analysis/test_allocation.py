"""Physical placement, semantic aliases, and call-boundary proofs."""

from math import prod

import isl
import pytest

from tilefoundry import analysis
from tilefoundry.ir import expr, types


@pytest.mark.parametrize(
    ("layout", "dtype", "capacity"),
    [
        (types.Layout((2, 3), (8, 1)), expr.DType.f32, 44),
        (types.Layout((2, 4), (0, 1)), expr.DType.f32, 16),
        (types.ComposedLayout(None, 3, types.Layout((4,), (-1,))), expr.DType.f32, 16),
        (types.ComposedLayout(None, 7, types.Layout((4,), (2,))), expr.DType.bf16, 28),
        (
            types.ComposedLayout(types.Swizzle(3, 4, 3), 0, types.Layout(((8, 8), 16), ((512, 64), 1))),
            expr.DType.bf16,
            8192,
        ),
        (types.ComposedLayout(types.Swizzle(3, 4, 3), 16, types.Layout((2, 4), (16, 1))), expr.DType.bf16, 2048),
        (types.ComposedLayout(types.Swizzle(1, 1, -1), 0, types.Layout((4,), (8,))), expr.DType.bf16, 64),
        (types.ComposedLayout(types.Layout((4,), (3,)), 5, types.Layout((2,), (1,))), expr.DType.f32, 40),
        (
            types.ComposedLayout(types.ComposedLayout(types.Swizzle(1, 1, -1), 2, None), 1, types.Layout((3,), (8,))),
            expr.DType.bf16,
            48,
        ),
        (types.Layout((3,), (9,)), expr.DType.bool, 3),
    ],
)
def test_smem_capacity_covers_the_applied_layout(layout, dtype, capacity):
    shape = tuple(types.flatten(layout).shape)
    held = types.TensorType(shape, dtype, layout, expr.StorageKind.SMEM)
    assert analysis.storage_bytes(held) == capacity
    addresses = {
        (types.apply(layout, coordinate) * dtype.bit_width + bit) // 8
        for coordinate in range(prod(shape))
        for bit in range(dtype.bit_width)
    }
    assert addresses and min(addresses) >= 0 and max(addresses) < capacity


def test_smem_capacity_rejects_negative_addresses_and_handles_empty_values():
    with pytest.raises(analysis.Error, match="invalid physical address range"):
        analysis.storage_bytes(types.TensorType((4,), expr.DType.f32, types.Layout((4,), (-1,)), expr.StorageKind.SMEM))
    assert analysis.storage_bytes(types.TensorType((0, 4), expr.DType.f32, None, expr.StorageKind.SMEM)) == 0


@pytest.mark.parametrize(
    ("level", "capacity", "lanes"), [(None, 508, 4), ("cta", 108, 4), ("thread", 12, 1)]
)
def test_smem_capacity_counts_only_the_selected_units_coordinates(level, capacity, lanes):
    topologies = (types.Topology("cta", 2), types.Topology("thread", 4))
    mesh = types.Mesh(topologies, types.Layout((2, 4), (4, 1)), names=("block", "lane"))
    layout = types.Layout((2, 4, 3), (100, 8, 1))
    held = types.TensorType(
        (2, 4, 3), expr.DType.f32, types.ShardLayout(layout, (types.Split(0), types.Split(1)), mesh), expr.StorageKind.SMEM
    )
    assert analysis.storage_bytes(held, topology_level=level, topologies=topologies) == capacity
    addresses = {
        types.apply(layout, block + 2 * lane + 8 * column) * 4 + byte
        for block in range(2 if level is None else 1)
        for lane in range(lanes)
        for column in range(3)
        for byte in range(4)
    }
    assert max(addresses) + 1 == capacity


def test_union_coverage_does_not_prove_complete_writes_per_iteration() -> None:
    """Complementary partial writes cannot justify destructive result aliasing."""
    outputs = isl.map(
        "[iteration] -> { [element] -> [element] : "
        "0 <= iteration <= 1 and "
        "((iteration = 0 and 0 <= element <= 1) or "
        "(iteration = 1 and 2 <= element <= 3)) }"
    )
    result_box = isl.set("{ [element] : 0 <= element <= 3 }")
    iterations = isl.set("[iteration] -> { : 0 <= iteration <= 1 }")
    full_result = isl.set(
        "[iteration] -> { [element] : 0 <= iteration <= 1 and 0 <= element <= 3 }"
    )

    assert outputs.range().project_out(isl.dim_type.PARAM, 0, 1).is_equal(result_box)
    assert not analysis.covers_result(outputs.domain(), full_result, outputs, iterations, result_box)
