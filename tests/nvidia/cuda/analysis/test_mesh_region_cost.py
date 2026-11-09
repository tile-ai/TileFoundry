"""Execution regions turn one unit's work into whole-scope cost."""

from __future__ import annotations

from pathlib import Path

from tilefoundry import analysis, cli
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import math

ANALYSIS = Path(__file__).parents[1] / "fixtures" / "analysis"


def _cost(owner) -> tuple[int, int, int, int]:
    result = analysis.analyze(
        *analysis.check_program(owner, owner.entry_function(), topology_level="thread"),
        analysis=("compute-cost", "memory", "roofline"),
        topology_level="thread",
    )
    record = expr.get_metadata(result.function, analysis.ComputeCostMetadata)
    assert record is not None
    memory = expr.get_metadata(result.function, analysis.RegionMemoryMetadata)
    footprint_total = sum(
        held.of("gmem").total for _buffer, _ownership, held in memory.footprint.buffers
    )
    assert expr.get_metadata(result.function, analysis.RooflineMetadata).memory_ns == -(
        -(footprint_total * 1_000_000_000) // 4_800_000_000_000
    )
    return (
        analysis.shares(record.flops, record.topologies)["f32"],
        analysis.shares(record.flops, record.topologies, "thread")["f32"],
        memory.traffic.storage.of("gmem").total.total_bytes,
        footprint_total,
    )


def test_mesh_scope_multiplies_unit_cost() -> None:
    NoScope = cli.load_authored_ir(f"{ANALYSIS / 'add_outside_mesh_scope.py'}:NoScope")
    WithScope = cli.load_authored_ir(f"{ANALYSIS / 'add_sharded_in_mesh_scope.py'}:WithScope")
    UnshardedInScope = cli.load_authored_ir(
        f"{ANALYSIS / 'add_unsharded_in_mesh_scope.py'}:UnshardedInScope"
    )
    assert _cost(NoScope) == (128, 128, 1536, 1024)
    assert _cost(WithScope) == (256, 32, 2048, 2048)
    assert _cost(UnshardedInScope) == (1024, 128, 4096, 1024)


def test_region_calls_priced_per_position() -> None:
    """Stepping into occurrences preserves repetition and counts the escaped value once."""
    region_boundaries = Path(__file__).parents[1] / "fixtures/placed/region_boundaries.py"
    RegionBoundaries = cli.load_authored_ir(f"{region_boundaries}:RegionBoundaries")
    helper = analysis.analyze(
        *analysis.check_program(RegionBoundaries, RegionBoundaries.lookup("helper"), topology_level="thread"),
        analysis="compute-cost",
        topology_level="thread",
    )
    helper_record = expr.get_metadata(helper.function, analysis.ComputeCostMetadata)
    assert helper_record is not None
    helper_flops = analysis.shares(helper_record.flops, helper_record.topologies)["f32"]
    assert helper_flops == 8

    result = analysis.analyze(
        *analysis.check_program(RegionBoundaries, RegionBoundaries.entry_function(), topology_level="thread"),
        analysis=("compute-cost", "memory"),
        topology_level="thread",
    )
    record = expr.get_metadata(result.function, analysis.ComputeCostMetadata)
    assert record is not None
    assert analysis.shares(record.flops, record.topologies)["f32"] == 40
    assert analysis.shares(record.flops, record.topologies, "thread")["f32"] == 6
    moved = expr.get_metadata(result.function, analysis.RegionMemoryMetadata)
    assert {
        level: spread.total.total_bytes for level, spread in moved.traffic.storage.kinds
    } == {"gmem": 384, "smem": 64, "rmem": 720}
    binaries = [
        expr.get_metadata(node, analysis.ComputeCostMetadata)
        for function in result.functions
        for node in core.collect_exprs(function.body)
        if isinstance(node, expr.Call) and isinstance(node.target, math.Binary)
    ]
    binary_flops = sorted(analysis.shares(item.flops, item.topologies)["f32"] for item in binaries)
    assert binary_flops == [8, helper_flops * 2, 16]
