"""The scheduling corpus's plain programs, read the way a scheduler reads them.

These are the programs an author states before choosing an instruction: a tiled
matmul chain, the same chain with both operand windows staged, its untiled
baseline, and the real matmul over a CTA grid. What each one is here to witness
is that it parses, holds together, and can be measured -- the three questions
anything downstream asks before it offers a schedule at all.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

import pytest
import torch

import tilefoundry.passes.transforms.convert_hir_to_tir as lowering_module
from tilefoundry.analysis.allocation import alignment_of, view_root
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import analyze_liveness, result_copies
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)
from tilefoundry.cli import main as cli_main
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.inspection import PatternPrinter, as_script
from tilefoundry.ir.core import Call, Op, Var, detach_metadata, get_metadata
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.schedule import ScheduleOp, issue_plan
from tilefoundry.ir.pattern import Tensor
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.cuda.nn.wgmma import Wgmma
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType, UnitType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import levels, starts
from tilefoundry.ir.types.utils import bytes_by_storage
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    boundary_maps,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry.visitor_registry.verify import verify_prim_function

PLAIN = (
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
)
TIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "tir").glob("*.py")))
HIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "hir").glob("*.py")))
WGMMA_DECLARATION = Path(__file__).parents[1] / "fixtures" / "schedule" / "Wgmma.described.txt"
ANALYSES = (
    ("compute-cost", ComputeCostMetadata),
    ("memory", MemoryMetadata),
    ("roofline", RooflineMetadata),
    ("performance", PerformanceSummaryMetadata),
)
_TENSOR_CLOCK_HZ = 1_830_000_000

M1_LOWERING = (
    "sm80_mma_ldmatrix",
    "wgmma_a_k_major",
    "wgmma_a_mn_major",
    "wgmma_cast_between_schedules",
    "wgmma_cp_async_loads",
    "wgmma_explicit_windows",
    "wgmma_insert_tiles_into_output",
    "wgmma_one_tile_of_larger_output",
    "wgmma_rs_a_from_accumulator",
    "wgmma_rs_a_from_smem",
    "wgmma_swizzled_smem",
    "wgmma_two_schedules",
)
M2_LOWERING = tuple(path.stem for path in HIR if path.stem not in M1_LOWERING)


@dataclass(frozen=True)
class _RmemExpectation:
    peak_bytes: int
    derivation: str


SMEM_GOLDEN = {
    "gemm_8192x17408x5120_cta_grid": 196_608,
    "gemm_8192x17408x5120_persistent": 196_608,
    "gemm_8192x17408x5120_register_store": 196_608,
    "gemm_8192x17408x5120_tma_store": 196_608,
    "sm80_mma_ldmatrix": 1_536,
    "wgmma_a_k_major": 6_144,
    "wgmma_a_mn_major": 6_144,
    "wgmma_cast_between_schedules": 6_144,
    "wgmma_cp_async_loads": 6_144,
    "wgmma_cta_grid_4x17": 13_824,
    "wgmma_explicit_windows": 6_144,
    "wgmma_insert_tiles_into_output": 6_144,
    "wgmma_k_slices_of_wide_run": 12_288,
    "wgmma_one_tile_of_larger_output": 6_144,
    "wgmma_repeat_along_k": 36_864,
    "wgmma_repeat_along_n": 24_576,
    "wgmma_rs_a_from_accumulator": 5_120,
    "wgmma_rs_a_from_smem": 6_144,
    "wgmma_swizzled_smem": 6_144,
    "wgmma_tma_3stage": 13_824,
    "wgmma_two_schedules": 12_288,
}

RMEM_EXPECTED = {
    "gemm_8192x17408x5120_cta_grid": {
        "thread@128:256#0": _RmemExpectation(131_072, "128x256 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(131_072, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(131_072, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131_072, "parent envelope of one alias chain"),
    },
    "gemm_8192x17408x5120_persistent": {
        "thread@128:256#0": _RmemExpectation(131_072, "128x256 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(131_072, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(131_072, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131_072, "parent envelope of one alias chain"),
    },
    "gemm_8192x17408x5120_register_store": {
        "thread@128:256#0": _RmemExpectation(131_072, "128x256 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(131_072, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(131_072, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131_072, "parent envelope of one alias chain"),
    },
    "gemm_8192x17408x5120_tma_store": {
        "thread@128:256#0": _RmemExpectation(131_072, "128x256 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(131_072, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(131_072, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131_072, "parent envelope of one alias chain"),
    },
    "sm80_mma_ldmatrix": {
        "thread@32:32#0": _RmemExpectation(512, "16x8 f32 zero accumulator"),
        "thread@32:32#1": _RmemExpectation(
            1_280,
            "512-byte accumulator plus 512-byte lhs and 256-byte rhs fragments",
        ),
        "thread@32:32#2": _RmemExpectation(512, "f32 loop result/bf16 cast alias"),
        "thread@0:64#0": _RmemExpectation(1_280, "parent envelope of accumulator/fragments"),
    },
    "wgmma_a_k_major": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_a_mn_major": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_cast_between_schedules": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_cp_async_loads": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_cta_grid_4x17": {
        "thread@128:256#0": _RmemExpectation(8_192, "128x16 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_explicit_windows": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_insert_tiles_into_output": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_k_slices_of_wide_run": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_one_tile_of_larger_output": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_repeat_along_k": {
        "thread@128:256#0": _RmemExpectation(8_192, "128x16 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_repeat_along_n": {
        "thread@128:256#0": _RmemExpectation(131_072, "128x256 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(131_072, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(131_072, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131_072, "parent envelope of one alias chain"),
    },
    "wgmma_rs_a_from_accumulator": {
        "thread@128:128#0": _RmemExpectation(4_096, "64x16 f32 p initializer"),
        "thread@128:128#1": _RmemExpectation(4_096, "p phi/first mma alias chain"),
        "thread@128:128#2": _RmemExpectation(
            10_240,
            "2048-byte narrowed p alias plus independent 8192-byte acc initializer",
        ),
        "thread@128:128#3": _RmemExpectation(8_192, "acc phi/second mma alias chain"),
        "thread@128:128#4": _RmemExpectation(8_192, "acc loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(10_240, "parent envelope of p/acc transition"),
    },
    "wgmma_rs_a_from_smem": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(
            10_240,
            "8192-byte accumulator alias chain plus independent 2048-byte A fragment",
        ),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(10_240, "parent envelope of accumulator/A fragment"),
    },
    "wgmma_swizzled_smem": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_tma_3stage": {
        "thread@128:256#0": _RmemExpectation(8_192, "128x16 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(8_192, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(8_192, "parent envelope of one alias chain"),
    },
    "wgmma_two_schedules": {
        "thread@128:128#0": _RmemExpectation(8_192, "64x32 f32 zero accumulator"),
        "thread@128:128#1": _RmemExpectation(8_192, "phi/first mma alias chain"),
        "thread@128:128#2": _RmemExpectation(8_192, "first/second mma alias chain"),
        "thread@128:128#3": _RmemExpectation(8_192, "f32 loop result/bf16 cast alias"),
        "thread@0:256#0": _RmemExpectation(8_192, "parent envelope of both mma chains"),
    },
}


def _region_key(region: MeshRegion, occurrences: dict[str, int]) -> str:
    """Name one region by its selected topology run and lexical occurrence."""
    mesh = region.mesh
    assert len(mesh.topologies) == 1
    topology = getattr(mesh.topologies[0], "name", mesh.topologies[0])
    offset = starts(mesh)[0]
    size = prod(flatten(levels(mesh)[0].shape))
    selection = f"{topology}@{offset}:{size}"
    occurrence = occurrences.get(selection, 0)
    occurrences[selection] = occurrence + 1
    return f"{selection}#{occurrence}"


def _prim_in(path: Path) -> PrimFunction:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if isinstance(value, PrimFunction))


def _module_in(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if type(value).__name__ == "Module")


@pytest.mark.parametrize("name", PLAIN)
def test_plain_program_is_analyzable(name: str) -> None:
    module = importlib.import_module(f"tests.fixtures.schedule.plain.{name}")
    program = next(value for value in vars(module).values() if type(value).__name__ == "Module")
    entry = next(function for function in program.functions if function.name == "gemm")
    check_program(program, entry)
    result = analyze(program, entry, analysis=("memory", "performance"))
    assert result.metadata_types


@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_scheduled_hir_program_is_well_typed(path: Path) -> None:
    program = _module_in(path)
    entry = next(function for function in program.functions if function.name == "gemm")
    check_program(program, entry)


@pytest.mark.parametrize(("analysis", "metadata_type"), ANALYSES)
@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_scheduled_hir_program_has_analysis_metadata(
    path: Path, analysis: str, metadata_type: type
) -> None:
    program = _module_in(path)
    entry = next(function for function in program.functions if function.name == "gemm")
    if path.stem == "wgmma_tma_3stage":
        read_write = []
        for expr in collect_exprs(entry.body):
            if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp):
                schema = type(expr.target.op)._op_schema
                writes = tuple(
                    param
                    for param in schema.signature
                    if param.kind == "input" and param.effect & MemoryEffect.WRITE
                )
                if writes[0].effect & MemoryEffect.READ:
                    read_write.append(expr)
        assert len(read_write) == 1
        read_write[0].target.buffers = 3
    result = analyze(program, entry, analysis=analysis)
    assert metadata_type in result.metadata_types

    if analysis == "compute-cost":
        schedules = (
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)
        )
        assert all(
            not get_metadata(expr, ComputeCostMetadata).other_ops.kinds for expr in schedules
        )

    if analysis == "memory" and path.stem == "wgmma_tma_3stage":
        assert RegionMemoryMetadata in result.metadata_types
        lifetimes = get_metadata(result.function, RegionMemoryMetadata).lifetimes
        smem = sorted(item.bytes for item in lifetimes if item.memory_level == "smem")
        rmem = sorted(item.bytes for item in lifetimes if item.memory_level == "rmem")
        assert smem == sorted((512 * 3, 4096 * 3))
        assert rmem == [4096, 8192, 8192, 8192, 8192]

    if analysis == "memory":
        placement = get_metadata(result.function, RegionMemoryMetadata)
        assert placement is not None
        smem_peak = next(
            item.peak_bytes for item in placement.peaks if item.memory_level == "smem"
        )
        assert smem_peak == SMEM_GOLDEN[path.stem]
        for expr in collect_exprs(result.function.body):
            if not isinstance(expr, Call):
                continue
            moved = get_metadata(expr, MemoryMetadata)
            assert moved is not None
            levels = bytes_by_storage(expr.type)
            if set(levels) == {"rmem"}:
                assert moved.offsets == ()
            addressable = set(levels) & {"gmem", "smem"}
            if not addressable or view_root(expr, {}) is not expr:
                continue
            assert len(addressable) == 1
            assert moved.buffer_bytes is not None
            assert len(moved.offsets) == result_copies(expr)
            peak = placement.peak_for(addressable.pop())
            assert peak is not None
            for offset in moved.offsets:
                assert offset % alignment_of(expr) == 0
                assert offset + moved.buffer_bytes <= peak.peak_bytes

        liveness = analyze_liveness(result.function)
        regions = tuple(window.region for window in liveness.regions)
        region_records = tuple(
            record
            for region in regions
            if (record := get_metadata(region, RegionMemoryMetadata)) is not None
        )
        assert len(region_records) == len(regions)
        for record in region_records:
            assert record.solver_status == "feasible"
            assert record.topologies == placement.topologies
            assert record.peaks
            assert not record.traffic.storage.kinds
            assert not record.traffic.communication.kinds
            assert record.footprint is None
            assert not record.reuse_windows
            assert not record.lifetimes
            assert not record.errors
            assert not record.advisories
        region_rmem_peaks = tuple(
            peak.peak_bytes
            for record in region_records
            if (peak := record.peak_for("rmem")) is not None
        )
        occurrences: dict[str, int] = {}
        observed_rmem = {}
        for window in liveness.regions:
            key = _region_key(window.region, occurrences)
            record = get_metadata(window.region, RegionMemoryMetadata)
            peak = record.peak_for("rmem") if record is not None else None
            if peak is not None:
                observed_rmem[key] = peak.peak_bytes
        expected_rmem = {
            key: expectation.peak_bytes
            for key, expectation in RMEM_EXPECTED[path.stem].items()
        }
        assert observed_rmem == expected_rmem, {
            key: expectation.derivation
            for key, expectation in RMEM_EXPECTED[path.stem].items()
        }
        assert placement.peak_for("rmem").peak_bytes == max(region_rmem_peaks, default=0)

        intervals = {id(item.value): item for item in liveness.intervals}
        loop_bounds = tuple(
            (
                intervals[id(loop.induction_var)].defined_at,
                max(
                    use.at
                    for use in liveness.uses
                    if any(use.value is yielded for yielded in loop.yield_values)
                    and use.at < intervals[id(loop)].defined_at
                ),
            )
            for loop in collect_exprs(result.function.body)
            if isinstance(loop, LoopRegion)
        )

        outside_uses = 0
        for use in liveness.uses:
            interval = intervals[id(use.value)]
            for phi, backedge in loop_bounds:
                if (
                    not use.synthetic
                    and phi < use.at < backedge
                    and interval.defined_at < phi
                ):
                    outside_uses += 1
                    assert interval.last_used_at >= backedge
        assert outside_uses

        for interval in liveness.intervals:
            if result_copies(interval.value) == 1:
                continue
            assert (interval.defined_at, interval.last_used_at) in loop_bounds

    if analysis == "performance" and path.stem == "gemm_8192x17408x5120_cta_grid":
        local_moves = []
        for expr in collect_exprs(result.function.body):
            if not isinstance(expr, Call) or not isinstance(expr.target, ScheduleOp):
                continue
            moved = get_metadata(expr, MemoryMetadata)
            levels = moved.traffic.storage.names()
            if "smem" in levels and "gmem" not in levels:
                local_moves.append(get_metadata(expr, PerformanceMetadata))
        assert local_moves and all(
            item is not None and item.timeline.end_ns > item.timeline.start_ns
            for item in local_moves
        )


@pytest.mark.parametrize(
    ("fixture", "n"),
    (
        ("wgmma_rs_a_from_accumulator", 16),
        ("wgmma_repeat_along_n", 64),
        ("gemm_8192x17408x5120_register_store", 256),
    ),
)
def test_wgmma_performance_prices_n_over_two_tensor_clocks(fixture: str, n: int) -> None:
    """Price real timeline output against an independent tensor-clock reference.

    The 1.83 GHz value comes from the stage2c instruction-throughput research
    section 2, not from the peak under test. Its 0.0071% difference from the
    peak-implied 1,830,129,912 Hz leaves 0.63, 0.11, and 0.44 ns before the three
    ceil boundaries. A future case crossing one boundary needs its expected
    integer time checked before treating the one-ns change as a bug.
    """
    path = next(path for path in HIR if path.stem == fixture)
    program = _module_in(path)
    entry = next(function for function in program.functions if function.name == "gemm")
    result = analyze(program, entry, analysis="performance")
    schedule = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and getattr(getattr(expr.target.op, "atom", None), "bindings", {}).get("n") == n
    )
    spread = get_metadata(schedule, ComputeCostMetadata).flops.of("bf16")
    issues = prod(schedule.target.repeat or (1,))
    flops_per_issue = spread.logical // issues
    timeline = get_metadata(schedule, PerformanceMetadata).timeline
    duration_ns = timeline.end_ns - timeline.start_ns
    expected_ns = -(-(n * issues * 1_000_000_000) // (2 * _TENSOR_CLOCK_HZ))

    assert flops_per_issue == 2 * 64 * n * 16
    assert duration_ns == expected_ns


def _copy_schedule_call(
    *, repeat=None, order=None, storage: StorageKind = StorageKind.GMEM
) -> Call:
    layout = Layout((4,), (1,))
    type_ = TensorType((4,), DType.bf16, layout, storage)
    return Call(
        target=ScheduleOp(
            op=CopyAsync(smem_layout=layout),
            repeat=repeat,
            order=order,
        ),
        args=(Var(name="src", type=type_),),
        type=type_,
    )


def _evaluate_call(
    op: Op,
    typed_inputs: tuple[tuple[TensorType, torch.Tensor], ...],
    result_type: TensorType,
) -> torch.Tensor:
    params = tuple(
        Var(name=f"arg{index}", type=type_) for index, (type_, _data) in enumerate(typed_inputs)
    )
    call = Call(target=op, args=params, type=result_type)
    function = Function.build(
        name="schedule_value",
        params=params,
        body=call,
        return_type=result_type,
    )
    return evaluate(function, *(data for _type, data in typed_inputs))


def test_schedule_copy_evaluates_to_the_source_value() -> None:
    layout = Layout((4,), (1,))
    source_type = TensorType((4,), DType.bf16, layout, StorageKind.GMEM)
    result_type = TensorType((4,), DType.bf16, layout, StorageKind.SMEM)
    source = torch.arange(4, dtype=torch.bfloat16)

    result = _evaluate_call(
        ScheduleOp(op=CopyAsync(smem_layout=layout)),
        ((source_type, source),),
        result_type,
    )

    assert torch.equal(result, source)


def test_schedule_mma_evaluates_like_matmul_plus_accumulator() -> None:
    acc_type = TensorType((16, 8), DType.f32, None, StorageKind.RMEM)
    lhs_type = TensorType((16, 16), DType.bf16, None, StorageKind.RMEM)
    rhs_type = TensorType((16, 8), DType.bf16, None, StorageKind.RMEM)
    product_type = TensorType((16, 8), DType.bf16, None, StorageKind.RMEM)
    acc = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    lhs = (torch.arange(16 * 16).reshape(16, 16) % 5).to(torch.bfloat16)
    rhs = (torch.arange(16 * 8).reshape(16, 8) % 3).to(torch.bfloat16)

    product = _evaluate_call(
        MatMul(),
        ((lhs_type, lhs), (rhs_type, rhs)),
        product_type,
    )
    scheduled = _evaluate_call(
        ScheduleOp(op=TiledMma(atom=Mma())),
        ((acc_type, acc), (lhs_type, lhs), (rhs_type, rhs)),
        acc_type,
    )

    assert torch.equal(scheduled, acc + product)


def test_single_issue_schedule_preserves_instruction_relations() -> None:
    schedule = _copy_schedule_call(repeat=(1,), order=(0,))
    source = schedule.args[0]
    destination_type = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.SMEM)
    instruction = Call(
        target=schedule.target.op,
        args=(source, Var(name="dst", type=destination_type)),
        type=UnitType(),
    )
    ctx = TypeInferContext()
    scheduled = boundary_maps(relations_of(schedule, ctx))
    instruction_relations = relations_of(instruction, ctx)
    single = boundary_maps(
        AccessRelations(
            inputs=(instruction_relations.inputs[0],),
            outputs=instruction_relations.outputs,
        )
    )
    assert len(scheduled) == len(single)
    assert all(left.is_equal(right) for left, right in zip(scheduled, single, strict=True))


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (_copy_schedule_call(repeat=(2,)), "conflicts with inferred repeat"),
        (_copy_schedule_call(order=(1,)), "is not a permutation"),
        (
            _copy_schedule_call(storage=StorageKind.SMEM),
            "src single-issue tile does not match",
        ),
    ),
    ids=("repeat", "order", "operand-pattern"),
)
def test_schedule_typeinfer_rejects_invalid_contract(call: Call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inference_type(call)


@register_op(dialect="T", category="schedule", name="unstated_instruction")
class _UnstatedInstruction(Op):
    value = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=Tensor,
    )


def test_schedule_typeinfer_requires_instruction_access_relations() -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    with pytest.raises(ValueError, match="states no access relations"):
        inference_type(call)


def test_schedule_evaluation_rejects_an_unregistered_instruction() -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    value = torch.arange(4, dtype=torch.bfloat16)

    with pytest.raises(EvalError, match="no schedule evaluator registered"):
        _evaluate_call(ScheduleOp(op=_UnstatedInstruction()), ((type_, value),), type_)


@pytest.mark.parametrize("path", TIR, ids=lambda path: path.stem)
def test_tir_program_is_verified_and_canonical(path: Path) -> None:
    function = _prim_in(path)
    verify_prim_function(function)
    assert as_script(function) == path.read_text()


def test_wgmma_declaration_is_canonical() -> None:
    assert PatternPrinter().declaration(Wgmma) + "\n" == WGMMA_DECLARATION.read_text()


@pytest.mark.parametrize("name", M1_LOWERING)
def test_schedule_finalize_writes_verified_tir(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / f"{name}.py"
    out = tmp_path / f"{name}.py"

    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    verify_prim_function(_prim_in(out))

    if name == "wgmma_a_k_major":
        rendered = out.read_text()
        assert "lhs_stages = (T.tensor_view(2048" in rendered
        assert "rhs_stages = (T.tensor_view(0" in rendered
        assert "lhs_stages[(k // 16) % 2]" in rendered
        assert "acc = T.alloc_tensor" in rendered
        assert "result = T.alloc_tensor" in rendered


@pytest.mark.parametrize("name", M2_LOWERING)
def test_schedule_finalize_marks_later_issue_shapes(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / f"{name}.py"
    out = tmp_path / f"{name}.py"

    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "tilefoundry: error:" in captured.err
    assert "M2 not implemented" in captured.err
    assert not out.exists()


def test_schedule_finalize_json_carries_the_same_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_a_k_major.py"
    out = tmp_path / "finalized.json"

    assert cli_main(["schedule", "finalize", str(source), str(out), "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    payload = json.loads(out.read_text())
    assert set(payload) == {"source"}
    python = tmp_path / "finalized.py"
    python.write_text(payload["source"])
    verify_prim_function(_prim_in(python))


def test_issue_plan_exposes_group_loop_and_row_facts() -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_repeat_along_k.py"
    module = _module_in(source)
    result = analyze(module, module.entry_function(), analysis=("memory",))
    mma = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and isinstance(expr.target.op, TiledMma)
    )
    plan = issue_plan(
        mma,
        TypeInferContext(scope=FunctionScope(module, result.function)),
    )

    assert tuple(
        (axis.name, axis.extent, axis.atom, axis.repeat, axis.row_copies, axis.is_group)
        for axis in plan.axes
    ) == (
        ("m", 128, 64, 2, 1, True),
        ("n", 16, 16, 1, 1, False),
        ("k", 64, 16, 4, 4, False),
    )
    assert plan.operand_axes == (("m", "n"), ("m", "k"), ("k", "n"))


def _m1_analysis():
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_a_k_major.py"
    module = _module_in(source)
    return source, module, analyze(module, module.entry_function(), analysis=("memory",))


def _assert_cli_lowering_error(
    source: Path,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "failed.py"
    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert not out.exists()


def test_lowering_rejects_instruction_without_access_relation(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    transfer = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and not isinstance(expr.target.op, TiledMma)
    )
    transfer.target.op = _UnstatedInstruction()
    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "no registered access relation", tmp_path, capsys)


def test_lowering_rejects_addressable_result_without_offsets(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    addressable = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and expr.type.storage is StorageKind.SMEM
    )
    detach_metadata(addressable, MemoryMetadata)
    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "addressable smem result but no offsets", tmp_path, capsys)


def test_lowering_rejects_unknown_hir_call(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    view = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and type(expr.target).__name__ == "Slice"
    )
    view.target = _UnstatedInstruction()
    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "unknown HIR call _UnstatedInstruction", tmp_path, capsys)
