"""Execution regions turn one unit's work into whole-scope cost."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.placed.region_boundaries import RegionBoundaries
from tilefoundry.analysis import (
    ComputeCostMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
    analyze,
)
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.metadata import shares
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry_nvidia import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 2), Topology("thread", 4))


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class NoScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        return x + x


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class WithScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (2,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as thread:
                local = tf.reshard(x, (8, 16 @ thread.t), "rmem")
                return tf.reshard(local + local, (8, 16), "gmem")


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class UnshardedInScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (2,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as _thread:
                local = tf.zeros(Tensor[(8, 16), "f32", "rmem"])
                return tf.reshard(local + local, (8, 16), "gmem")


def _cost(owner) -> tuple[int, int, int, int]:
    result = analyze(
        *check_program(owner, owner.entry_function(), topology_level="thread"),
        analysis=("compute-cost", "memory", "roofline"),
        topology_level="thread",
    )
    record = get_metadata(result.function, ComputeCostMetadata)
    assert record is not None
    memory = get_metadata(result.function, RegionMemoryMetadata)
    footprint_total = sum(
        held.of("gmem").total for _buffer, _ownership, held in memory.footprint.buffers
    )
    assert get_metadata(result.function, RooflineMetadata).memory_ns == -(
        -(footprint_total * 1_000_000_000) // 4_800_000_000_000
    )
    return (
        shares(record.flops, record.topologies)["f32"],
        shares(record.flops, record.topologies, "thread")["f32"],
        memory.traffic.storage.of("gmem").total.total_bytes,
        footprint_total,
    )


def test_scope_positions_turn_per_unit_cost_into_total_cost() -> None:
    assert _cost(NoScope) == (128, 128, 1536, 1024)
    assert _cost(WithScope) == (256, 32, 2048, 2048)
    assert _cost(UnshardedInScope) == (1024, 128, 4096, 1024)


def test_region_boundaries_price_calls_per_position_and_values_once() -> None:
    """Stepping into occurrences preserves repetition and counts the escaped value once."""
    helper = analyze(
        *check_program(RegionBoundaries, RegionBoundaries.lookup("helper"), topology_level="thread"),
        analysis="compute-cost",
        topology_level="thread",
    )
    helper_record = get_metadata(helper.function, ComputeCostMetadata)
    assert helper_record is not None
    helper_flops = shares(helper_record.flops, helper_record.topologies)["f32"]
    assert helper_flops == 8

    result = analyze(
        *check_program(RegionBoundaries, RegionBoundaries.entry_function(), topology_level="thread"),
        analysis=("compute-cost", "memory"),
        topology_level="thread",
    )
    record = get_metadata(result.function, ComputeCostMetadata)
    assert record is not None
    assert shares(record.flops, record.topologies)["f32"] == 40
    assert shares(record.flops, record.topologies, "thread")["f32"] == 6
    moved = get_metadata(result.function, RegionMemoryMetadata)
    assert {
        level: spread.total.total_bytes for level, spread in moved.traffic.storage.kinds
    } == {"gmem": 384, "smem": 64, "rmem": 720}
    binaries = [
        get_metadata(expr, ComputeCostMetadata)
        for function in result.functions
        for expr in collect_exprs(function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Binary)
    ]
    binary_flops = sorted(shares(item.flops, item.topologies)["f32"] for item in binaries)
    assert binary_flops == [8, helper_flops * 2, 16]
