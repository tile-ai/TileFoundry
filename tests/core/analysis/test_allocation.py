"""Physical placement, semantic aliases, and call-boundary proofs."""

from math import prod

import isl
import pytest

from tilefoundry.analysis.allocation import (
    covers_result,
    storage_bytes,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    Mesh,
    ShardLayout,
    Split,
    StorageKind,
    Swizzle,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.layout import apply, flatten


@pytest.mark.parametrize(
    ("layout", "dtype", "capacity"),
    [
        (Layout((2, 3), (8, 1)), DType.f32, 44),
        (Layout((2, 4), (0, 1)), DType.f32, 16),
        (ComposedLayout(None, 3, Layout((4,), (-1,))), DType.f32, 16),
        (ComposedLayout(None, 7, Layout((4,), (2,))), DType.bf16, 28),
        (
            ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), 16), ((512, 64), 1))),
            DType.bf16,
            8192,
        ),
        (ComposedLayout(Swizzle(3, 4, 3), 16, Layout((2, 4), (16, 1))), DType.bf16, 2048),
        (ComposedLayout(Swizzle(1, 1, -1), 0, Layout((4,), (8,))), DType.bf16, 64),
        (ComposedLayout(Layout((4,), (3,)), 5, Layout((2,), (1,))), DType.f32, 40),
        (
            ComposedLayout(ComposedLayout(Swizzle(1, 1, -1), 2, None), 1, Layout((3,), (8,))),
            DType.bf16,
            48,
        ),
        (Layout((3,), (9,)), DType.bool, 3),
    ],
)
def test_smem_capacity_covers_the_applied_layout(layout, dtype, capacity):
    shape = tuple(flatten(layout).shape)
    held = TensorType(shape, dtype, layout, StorageKind.SMEM)
    assert storage_bytes(held) == capacity
    addresses = {
        (apply(layout, coordinate) * dtype.bit_width + bit) // 8
        for coordinate in range(prod(shape))
        for bit in range(dtype.bit_width)
    }
    assert addresses and min(addresses) >= 0 and max(addresses) < capacity


def test_smem_capacity_rejects_negative_addresses_and_handles_empty_values():
    with pytest.raises(AnalysisError, match="invalid physical address range"):
        storage_bytes(TensorType((4,), DType.f32, Layout((4,), (-1,)), StorageKind.SMEM))
    assert storage_bytes(TensorType((0, 4), DType.f32, None, StorageKind.SMEM)) == 0


@pytest.mark.parametrize(
    ("level", "capacity", "lanes"), [(None, 508, 4), ("cta", 108, 4), ("thread", 12, 1)]
)
def test_smem_capacity_counts_only_the_selected_units_coordinates(level, capacity, lanes):
    topologies = (Topology("cta", 2), Topology("thread", 4))
    mesh = Mesh(topologies, Layout((2, 4), (4, 1)), names=("block", "lane"))
    layout = Layout((2, 4, 3), (100, 8, 1))
    held = TensorType(
        (2, 4, 3), DType.f32, ShardLayout(layout, (Split(0), Split(1)), mesh), StorageKind.SMEM
    )
    assert storage_bytes(held, topology_level=level, topologies=topologies) == capacity
    addresses = {
        apply(layout, block + 2 * lane + 8 * column) * 4 + byte
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
    assert not covers_result(outputs.domain(), full_result, outputs, iterations, result_box)
