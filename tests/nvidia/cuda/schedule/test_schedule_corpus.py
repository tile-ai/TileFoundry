"""Scheduled programs checked, analyzed, and lowered once per input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import tilefoundry
from tests.core._corpus import Case, Problems, case, cases, reload
from tests.nvidia.cuda.schedule._plain import SiteCounts, check_plain_report
from tests.nvidia.cuda.schedule._support import (
    _assert_live_smem_views_do_not_overlap,
    _direct_operand_matches,
    _region_key,
    _RetainedStatements,
)
from tilefoundry.analysis import AnalysisResult
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import LivenessMetadata, result_copies
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)
from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.ir.core import (
    Call,
    Constant,
    Tuple,
    Var,
    get_metadata,
)
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.memory import Copy, PtrOf, TensorView
from tilefoundry.ir.types import (
    ComposedLayout,
    ShardLayout,
    StorageKind,
    Swizzle,
    TensorType,
    TupleType,
)
from tilefoundry.ir.types.dim import DimAdd, DimMul
from tilefoundry.ir.types.int_tuple import compatible_profile
from tilefoundry.ir.types.layout import apply, flatten, unflatten
from tilefoundry.ir.types.stride import crd2idx
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import StmtVisitor, collect_exprs
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError
from tilefoundry.schedule import candidates, finalize
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.nn.wgmma import Wgmma

ANALYSES = (
    ("compute-cost", ComputeCostMetadata),
    ("memory", MemoryMetadata),
    ("roofline", RooflineMetadata),
    ("performance", PerformanceSummaryMetadata),
)

_TENSOR_CLOCK_HZ = 1_830_000_000

_SHARED_BYTES_PER_CLOCK = 128

ROOT = Path(__file__).parents[1] / "fixtures" / "schedule"


@dataclass(frozen=True)
class RmemPeak:
    bytes: int
    why: str


@dataclass(frozen=True)
class ScheduleCase(Case):
    dims: Mapping[str, int] | None = None
    refusal: str | None = None
    finalize_refusal: str | None = None
    candidates_refusal: str | None = None
    smem_peak: int = 0
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType({})
    sites: SiteCounts | None = None
    child_regions: bool = False


@dataclass(frozen=True)
class Scheduled:
    result: AnalysisResult | None
    module: Module | None
    lowered: Module | None
    candidates: dict | None


FAMILIES = ("compute-cost", "memory", "roofline", "performance")


def _analysis_metadata(case, result, analysis, metadata_type):
    assert metadata_type in result.metadata_types

    if analysis == "memory":
        for expr in collect_exprs(result.function.body):
            if not isinstance(expr, Call):
                continue
            record = get_metadata(expr, MemoryMetadata)
            if aliased_operand(expr) is not None:
                assert record.buffer_bytes is None
                assert not record.offsets
            elif isinstance(expr.target, Transpose) and expr.type.storage is not StorageKind.RMEM:
                assert record.buffer_bytes == prod(expr.type.shape) * expr.type.dtype.bit_width // 8
                assert record.offsets
            if isinstance(expr.target, ScheduleOp) and isinstance(
                expr.target.op, (Copy, CopyAsync, CopyAsyncTensor)
            ):
                tile = prod(expr.type.shape) * expr.type.dtype.bit_width // 8
                assert record.operands[1].write == tile, "a copy writes its whole tile"
            if isinstance(expr.target, ScheduleOp) and isinstance(expr.target.op, TiledMma):
                for operand, moved in zip(expr.args, record.operands, strict=False):
                    if operand.type.storage is StorageKind.SMEM:
                        tile = prod(operand.type.shape) * operand.type.dtype.bit_width // 8
                        assert moved.read == tile, "an mma reads its whole shared tile"

    if analysis == "compute-cost":
        schedules = (
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)
        )
        assert all(
            not get_metadata(expr, ComputeCostMetadata).other_ops.kinds for expr in schedules
        )

    if analysis == "memory":
        placement = get_metadata(result.function, RegionMemoryMetadata)
        assert placement is not None
        liveness = get_metadata(result.function, LivenessMetadata)
        local = {id(expr) for expr in collect_exprs(result.function.body)}
        region_windows = tuple(window for window in liveness.regions if id(window.region) in local)
        regions = tuple(window.region for window in region_windows)
        region_records = tuple(
            record
            for region in regions
            if (record := get_metadata(region, RegionMemoryMetadata)) is not None
        )
        if case.child_regions:
            region_records += tuple(
                record
                for function in result.functions[1:]
                for region in collect_exprs(function.body)
                if isinstance(region, MeshRegion)
                and (record := get_metadata(region, RegionMemoryMetadata)) is not None
            )
        _region_records(region_records, placement)
        region_rmem_peaks = tuple(
            peak.peak_bytes
            for record in region_records
            if (peak := record.peak_for("rmem")) is not None
        )
        occurrences: dict[str, int] = {}
        observed_rmem = {}
        for window in region_windows:
            key = _region_key(window.region, occurrences)
            record = get_metadata(window.region, RegionMemoryMetadata)
            peak = record.peak_for("rmem") if record is not None else None
            if peak is not None:
                observed_rmem[key] = peak.peak_bytes
        expected_rmem = {key: expectation.bytes for key, expectation in case.rmem_peaks.items()}
        assert observed_rmem == expected_rmem, {
            key: expectation.why for key, expectation in case.rmem_peaks.items()
        }
        global_rmem = placement.peak_for("rmem")
        assert (global_rmem.peak_bytes if global_rmem is not None else 0) == max(
            region_rmem_peaks, default=0
        )

        intervals = {item.value: item for item in liveness.intervals}
        loop_bounds = tuple(
            (
                intervals[SSAValue(loop.induction_var)].defined_at,
                max(
                    use.at
                    for use in liveness.uses
                    if any(use.value.expr is yielded for yielded in loop.yield_values)
                    and use.at < intervals[SSAValue(loop)].defined_at
                ),
            )
            for loop in collect_exprs(result.function.body)
            if isinstance(loop, LoopRegion)
        )

        outside_uses = 0
        for use in liveness.uses:
            interval = intervals[use.value]
            for phi, backedge in loop_bounds:
                if not use.synthetic and phi < use.at < backedge and interval.defined_at < phi:
                    outside_uses += 1
                    assert interval.last_used_at >= backedge
        if loop_bounds:
            assert outside_uses

        for interval in liveness.intervals:
            if result_copies(interval.value) == 1:
                continue
            assert (interval.defined_at, interval.last_used_at) in loop_bounds


def _region_records(records, placement):
    for record in records:
        assert record.solver_status == "feasible"
        assert record.topologies == placement.topologies
        assert record.peaks
        assert all(peak.memory_level in ("rmem", "smem") for peak in record.peaks)
        assert not record.traffic.storage.kinds
        assert not record.traffic.communication.kinds
        assert record.footprint is None
        assert not record.reuse_windows
        assert not record.lifetimes
        assert not record.errors
        assert not record.advisories


def _structural_layouts(program):
    structural = [
        expr
        for expr in collect_exprs(program.entry_function().body)
        if isinstance(expr, Call) and isinstance(expr.target, (Reshape, Slice, Transpose))
    ]
    missing = [type(expr.target).__name__ for expr in structural if expr.type.layout is None]
    assert not missing, (
        f"{len(missing)} of {len(structural)} structural calls omit layout: {missing}"
    )


def _memory_report(case, program, report):
    memory = report["functions"][0]["function_records"]["memory"]
    peaks = {item["memory_level"]: item["peak_bytes"] for item in memory["peaks"]}
    has_smem = any(
        isinstance(expr.type, TensorType) and expr.type.storage is StorageKind.SMEM
        for expr in collect_exprs(program.entry_function().body)
    )
    if has_smem:
        assert peaks["smem"] == case.smem_peak
    else:
        assert "smem" not in peaks
        assert case.smem_peak == 0
    assert memory["solver_status"] == "feasible"
    for row in report["functions"][0]["calls"]:
        allocation = row["memory"]
        if "operands" not in allocation:
            continue
        result = next(operand for operand in allocation["operands"] if operand["arg"] == "result")
        storage = result["type"].rsplit(" ", 1)[-1]
        offsets = allocation["offsets"]
        if storage == "rmem":
            assert offsets == []
            continue
        if storage not in {"gmem", "smem"} or allocation["buffer_bytes"] is None:
            continue
        assert offsets
        for offset in offsets:
            assert offset % 16 == 0
            assert offset + allocation["buffer_bytes"] <= peaks[storage]


def _tir_checks(lowered, source):
    functions = tuple(frame.function for frame in list_call_paths(lowered).values())
    found = []
    for function in functions:
        _assert_live_smem_views_do_not_overlap(function, source)
        found.extend(_direct_operand_matches(function))
    assert found
    lhs = [
        captures
        for op, name, captures in found
        if isinstance(getattr(op, "atom", None), Wgmma) and name == "lhs"
    ]
    assert all("a_major" in captures for captures in lhs)
    assert all("a_swizzle" in captures for captures in lhs if captures["form"].name == "SS")


def _hir_candidates(report):
    assert report["lines"]
    ops = [row["op"] for row in report["lines"]]
    assert "tf.matmul" not in ops
    assert "tf.reshard" not in ops
    assert any(
        op
        in {
            "tf.arange",
            "tf.binary",
            "tf.cast",
            "tf.clamp",
            "tf.relu",
            "tf.reduce",
            "tf.unary",
            "tf.transpose",
        }
        for op in ops
    )


def _candidates(case, checked):
    module, root = checked
    if case.candidates_refusal:
        with pytest.raises(ValueError) as refused:
            candidates(module, root, source=str(case.path))
        assert case.candidates_refusal in str(refused.value)
        return None
    return candidates(module, root, source=str(case.path))


def _finalize(case, result):
    if case.finalize_refusal:
        with pytest.raises(LoweringError) as refused:
            finalize(result)
        assert case.finalize_refusal in str(refused.value)
        return None
    return finalize(result)


def _host_launch(case, lowered, function):
    return_type = function.return_type
    def launchable(ty):
        if isinstance(ty, TupleType):
            return all(launchable(field) for field in ty.fields)
        return not isinstance(ty, TensorType) or ty.storage in (StorageKind.GMEM, StorageKind.UMAT)

    launches = []
    for fn in lowered.functions:
        if isinstance(fn, PrimFunction) and isinstance(fn.target, CpuTarget):
            statements = _RetainedStatements()
            statements.visit(fn.body)
            launches.extend(
                stmt for stmt in statements.statements
                if isinstance(stmt.callable, Launch)
                and not stmt.args[0].nested
                and stmt.args[0].name == function.name
            )
    if launchable(return_type):
        assert len(launches) == 1
        assert launches[0].callable.dynamic_smem == case.smem_peak
        assert isinstance(lowered.entry_function().target, CpuTarget)
    else:
        assert not launches
        assert lowered.entry == function.name


def _hir(case, problems):
    module = problems.run("load", case.load)
    checked = problems.run(
        "check", lambda m: check_program(m, m.entry_function(), dims=case.dims), module
    )
    problems.run("structural layouts", _structural_layouts, module)
    result = problems.run("analyze", lambda pair: analyze(*pair, analysis=FAMILIES), checked)
    for family, metadata_type in ANALYSES:
        problems.run(family, _analysis_metadata, case, result, family, metadata_type)
    report = problems.run("memory report", lambda r: render_analysis(r).data, result)
    problems.run("memory allocations", _memory_report, case, module, report)
    printed = problems.run("print HIR", as_script, module)
    problems.run("reload HIR", lambda text: reload(text, module.name), printed)
    candidates_report = problems.run("candidates", _candidates, case, checked)
    if case.candidates_refusal is None:
        problems.run("candidate sites", _hir_candidates, candidates_report)
    lowered = problems.run("finalize", _finalize, case, result)
    if case.finalize_refusal is None:
        printed = problems.run("print TIR", as_script, lowered)
        lowered = problems.run("reload TIR", lambda text: reload(text, lowered.name), printed)
        problems.run("TIR operands", _tir_checks, lowered, case.path)
        problems.run("host launch", _host_launch, case, lowered, result.function)
    state = Scheduled(result, module, lowered, candidates_report)
    problems.run("guard", case.guard, state, problems)


def _plain(case, problems):
    if case.refusal is not None:

        def refused():
            with pytest.raises(VerifyError, match=case.refusal):
                case.load()

        problems.run("load refusal", refused)
        return
    module = problems.run("load", case.load)
    checked = problems.run(
        "check", lambda m: check_program(m, m.entry_function(), dims=case.dims), module
    )
    result = problems.run(
        "analyze", lambda pair: analyze(*pair, analysis=("memory", "performance")), checked
    )
    problems.expect("analysis records", lambda r: bool(r.metadata_types), result)
    report = problems.run("candidates", _candidates, case, checked)
    problems.run("plain sites", check_plain_report, report, str(case.path), case.sites)
    printed = problems.run("print plain", as_script, module)
    problems.run("reload plain", lambda text: reload(text, module.name), printed)


def _dim(case, problems):
    module = problems.run("load", case.load)
    printed = problems.run("print", as_script, module)
    problems.run("reload", lambda text: reload(text, module.name), printed)


def _tir(case, problems):
    program = problems.run("load", case.load)
    problems.run("verify", verify_module, program)
    printed = problems.run("print", as_script, program)
    problems.expect("canonical text", lambda text: text == case.path.read_text(), printed)


@dataclass(frozen=True)
class WgmmaPrice(ScheduleCase):
    wgmma_n: int = 0
    bound_by: str = ""

    def guard(self, state, problems):
        n, bound_by, result = self.wgmma_n, self.bound_by, state.result
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
        shared_bytes = sum(
            prod(operand.type.shape) * operand.type.dtype.bit_width // 8
            for operand in schedule.args
            if operand.type.storage is StorageKind.SMEM
        )
        tensor_ns = -(-(n * issues * 1_000_000_000) // (2 * _TENSOR_CLOCK_HZ))
        shared_ns = -(
            -(shared_bytes * 1_000_000_000) // (_SHARED_BYTES_PER_CLOCK * _TENSOR_CLOCK_HZ)
        )

        assert flops_per_issue == 2 * 64 * n * 16
        assert (shared_ns > tensor_ns) == (bound_by == "smem")
        assert duration_ns == max(tensor_ns, shared_ns)


@dataclass(frozen=True)
class PhysicalSlots(ScheduleCase):
    def guard(self, state, problems):
        result, lowered = state.result, state.lowered
        peak = get_metadata(result.function, RegionMemoryMetadata).peak_for("smem").peak_bytes
        function = lowered.lookup(state.result.function.name)
        visitor = _RetainedStatements()
        visitor.visit(function.body)
        slots = []
        for value in visitor.bindings.values():
            fields = value.elements if isinstance(value, Tuple) else (value,)
            for slot in fields:
                if (
                    isinstance(slot, Call)
                    and isinstance(slot.target, TensorView)
                    and slot.target.storage is StorageKind.SMEM
                    and isinstance(slot.args[0], Constant)
                ):
                    held = slot.type
                    offset = slot.args[0].value
                    assert offset % 16 == 0
                    slots.append(
                        {
                            offset + (apply(held.layout, index) * held.dtype.bit_width + bit) // 8
                            for index in range(prod(flatten(held.layout).shape))
                            for bit in range(held.dtype.bit_width)
                        }
                    )
        assert len(slots) == 2
        assert slots[0].isdisjoint(slots[1])
        assert all(min(slot) >= 0 and max(slot) < peak for slot in slots)


@case
class BulkReduceStore(ScheduleCase):
    path: Path = ROOT / "hir/bulk_reduce_store.py"
    smem_peak: int = 16_384
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        lowered = state.lowered

        class Scope(StmtVisitor[None]):
            def __init__(self):
                self.scopes = []
                self.found = 0

            def visit_MeshScope(self, stmt, ctx=None):
                self.scopes.append(stmt.mesh)
                self.visit(stmt.body)
                self.scopes.pop()

            def visit_Evaluate(self, stmt, ctx=None):
                if isinstance(stmt.callable, CopyReduceAsyncTensor):
                    assert tuple(topology.name for topology in self.scopes[-1].topologies) == (
                        "thread",
                    )
                    assert prod(flatten(self.scopes[-1].layout).shape) == 32
                    self.found += 1

        visitor = Scope()
        visitor.visit(lowered.lookup(state.result.function.name).body)
        assert visitor.found == 1


@case
class CapturedInsertUpdate(ScheduleCase):
    path: Path = ROOT / "hir/captured_insert_update.py"
    smem_peak: int = 24
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        result = state.result
        liveness = get_metadata(result.function, LivenessMetadata)
        local = {id(expr) for expr in collect_exprs(result.function.body)}
        occurrences = {}
        observed_smem = {}
        for window in liveness.regions:
            if id(window.region) not in local:
                continue
            key = _region_key(window.region, occurrences)
            record = get_metadata(window.region, RegionMemoryMetadata)
            peak = record.peak_for("smem") if record is not None else None
            if peak is not None:
                observed_smem[key] = peak.peak_bytes
        assert observed_smem["thread@0:32#0"] == 24


@case
class Fp8BlockScaledGemm(ScheduleCase):
    path: Path = ROOT / "hir/fp8_block_scaled_gemm.py"
    smem_peak: int = 65_536
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "cta@0:1#0": RmemPeak(
                4,
                "the 1x1 f32 tile-scale copy is unsplit; thread-split accumulators "
                "do not contribute to the CTA record",
            ),
            "thread@128:256#0": RmemPeak(65_536, "128x128 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(
                131584,
                "independent live Buffers: 1 * 512B + 2 * 65536B",
            ),
            "thread@128:256#2": RmemPeak(65_536, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(131584, "independent live Buffers: 1 * 512B + 2 * 65536B"),
        }
    )

    def guard(self, state, problems):
        module = state.module
        m, n, k, block = 128, 128, 512, 128
        generator = torch.Generator().manual_seed(0)
        a = torch.randn(m, k, generator=generator).to(torch.float8_e4m3fn)
        weight = torch.randn(n, k, generator=generator).to(torch.float8_e4m3fn)
        b = weight.t()
        a_scale = torch.rand(m, k // block, generator=generator) + 0.5
        b_scale = torch.rand(k // block, n // block, generator=generator) + 0.5

        scheduled = evaluate(module.entry_function(), a, b, a_scale, b_scale)

        reference = sum(
            (a[:, kb * block : (kb + 1) * block].float() @ b[kb * block : (kb + 1) * block].float())
            * a_scale[:, kb : kb + 1]
            * b_scale[kb, :].repeat_interleave(block)
            for kb in range(k // block)
        ).to(torch.bfloat16)
        torch.testing.assert_close(scheduled.float(), reference.float(), rtol=2**-7, atol=0)


@case
class GemmRegisterStore(WgmmaPrice):
    path: Path = ROOT / "hir/gemm_8192x17408x5120_register_store.py"
    smem_peak: int = 196_608
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:256#0": RmemPeak(131_072, "128x256 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(131_072, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(131_072, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(131_072, "parent envelope of one alias chain"),
        }
    )
    wgmma_n: int = 256
    bound_by: str = "tensor"


@case
class GemmTmaStore(ScheduleCase):
    path: Path = ROOT / "hir/gemm_8192x17408x5120_tma_store.py"
    smem_peak: int = 212_992
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:256#0": RmemPeak(131_072, "128x256 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(131_072, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(131_072, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(131_072, "parent envelope of one alias chain"),
        }
    )

    def guard(self, state, problems):
        result = state.result
        liveness = get_metadata(result.function, LivenessMetadata)
        local = {id(expr) for expr in collect_exprs(result.function.body)}
        records = tuple(
            record
            for window in liveness.regions
            if id(window.region) in local
            and (record := get_metadata(window.region, RegionMemoryMetadata)) is not None
        )
        assert len(records) == 7
        schedule = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)
        )
        timeline = get_metadata(schedule, PerformanceMetadata).timeline
        assert timeline.end_ns - timeline.start_ns == 118


@case
class LoopOutputs(ScheduleCase):
    path: Path = ROOT / "hir/loop_outputs.py"
    smem_peak: int = 8
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        lowered = state.lowered
        assert lowered.lookup(state.result.function.name).output_count == 3
        assert reload(as_script(lowered), lowered.name).lookup(state.result.function.name).output_count == 3


@case
class ModuleInstances(ScheduleCase):
    path: Path = ROOT / "hir/module_instances.py"
    child_regions: bool = True
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "cta@0:1#0": RmemPeak(
                384, "three independent 128-byte callee operand/result Buffers in the CTA window"
            ),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        problems.run("module instances", self.module_instances, state.result)
        problems.run("retained shapes", self.retained_shapes, state.lowered)

    def module_instances(self, result):
        for analysis, metadata_type in ANALYSES:
            record_type = PerformanceMetadata if analysis == "performance" else metadata_type
            for function in result.functions[1:]:
                schedules = [
                    expr
                    for expr in collect_exprs(function.body)
                    if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)
                ]
                assert any(get_metadata(expr, record_type) is not None for expr in schedules)

    def retained_shapes(self, lowered):
        verify_module(lowered)
        frames = list_call_paths(lowered)
        assert len(frames) == 4
        assert all(isinstance(frame.function, PrimFunction) for frame in frames.values())
        layers, mm = lowered.children["layers"], lowered.children["mm"]
        assert len(layers.elements) == 3
        assert all(element is mm.elements[0] for element in layers.elements)
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        assert len(visitor.loops) == 1
        assert len(visitor.calls) == 5
        assert (
            sum(
                isinstance(call.callable, ModuleList) and call.callable.key == mm.key
                for call in visitor.calls
            )
            == 2
        )
        indexed = [call for call in visitor.calls if isinstance(call.callable, Call)]
        assert len(indexed) == 2
        assert all(call.callable.args[0].key == layers.key for call in indexed)
        assert any(isinstance(call.callable.args[1], Var) for call in indexed)
        assert any(
            isinstance(call.callable.args[1], Constant) and call.callable.args[1].value == 0
            for call in indexed
        )
        for call in visitor.calls:
            assert len(call.args) == (3 if isinstance(call.callable, PrimFunction) else 2)
            with pytest.raises(RuntimeError, match="no codegen handler for Op"):
                CppEmitter(context=CudaCodegenContext(target=lowered.target)).visit_Evaluate(call)


@case
class SmemPaddedStages(ScheduleCase):
    path: Path = ROOT / "hir/smem_padded_stages.py"
    smem_peak: int = 96
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        tilefoundry.compile(state.lowered)


@case
class ViewVisitMemo(ScheduleCase):
    path: Path = ROOT / "hir/view_visit_memo.py"
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@0:1#0": RmemPeak(32, "one eight-element f32 row"),
            "cta@0:1#0": RmemPeak(32, "parent envelope of the output row"),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        lowered = state.lowered
        verify_module(lowered)
        function = lowered.lookup(state.result.function.name)
        visitor = _RetainedStatements()
        visitor.visit(function.body)
        views = [
            value
            for value in visitor.bindings.values()
            if isinstance(value, Call) and isinstance(value.target, TensorView)
        ]
        assert [view.type.shape for view in views] == [(1,), (), (16, 8), (1, 8)]
        identity = views[2]
        assert isinstance(identity.args[0].target, PtrOf)
        assert identity.args[0].args[0] is function.params[0]
        assert len(visitor.statements) == 1
        assert isinstance(visitor.statements[0].callable, Copy)
        assert visitor.statements[0].args[1] is function.params[-1]


@case
class WgmmaBMnMajorGroupedN64(PhysicalSlots):
    path: Path = ROOT / "hir/wgmma_b_mn_major_grouped_n64.py"
    smem_peak: int = 18432
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@0:256#0": RmemPeak(32768, "two 64x64 f32 accumulator groups; MMA aliases acc"),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"


@case
class WgmmaBMnMajorWideLbo(PhysicalSlots):
    path: Path = ROOT / "hir/wgmma_b_mn_major_wide_lbo.py"
    smem_peak: int = 16384
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@0:128#0": RmemPeak(32768, "64x128 f32 accumulator and MMA result alias"),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"


@case
class WgmmaBMnMajorWindowUnitMode(PhysicalSlots):
    path: Path = ROOT / "hir/wgmma_b_mn_major_window_unit_mode.py"
    smem_peak: int = 34_816
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@0:256#0": RmemPeak(32768, "two 64x64 f32 accumulator groups; MMA aliases acc"),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"


@case
class WgmmaCastBetweenSchedules(ScheduleCase):
    path: Path = ROOT / "hir/wgmma_cast_between_schedules.py"
    smem_peak: int = 6_144
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(16384, "independent live Buffers: 2 * 8192B"),
            "thread@0:32#1": RmemPeak(
                2_048,
                "f32 loader tile (2048); bf16 cast follows its last read",
            ),
            "thread@0:256#0": RmemPeak(
                16384,
                "independent live Buffers: 2 * 8192B",
            ),
        }
    )

    def guard(self, state, problems):
        report = state.candidates
        ops = [row["op"] for row in report["lines"]]
        assert ops.count("tf.reduce") == 1


@case
class WgmmaN16WithinWarpgroupShare(ScheduleCase):
    path: Path = ROOT / "hir/wgmma_n16_within_warpgroup_share.py"
    smem_peak: int = 61_440
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:256#0": RmemPeak(
                24_576, "64x96 f32 accumulator across two warpgroups; MMA aliases acc"
            ),
            "thread@0:384#0": RmemPeak(24_576, "parent envelope of the accumulator"),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        result, lowered = state.result, state.lowered
        schedule = next(
            value
            for value in collect_exprs(result.function.body)
            if isinstance(value, Call)
            and isinstance(value.target, ScheduleOp)
            and isinstance(value.target.op, TiledMma)
        )
        original_c, _, original_b = (arg.type for arg in schedule.args)
        visitor = _RetainedStatements()
        function = lowered.lookup(state.result.function.name)
        visitor.visit(function.body)
        issues = [stmt for stmt in visitor.statements if isinstance(stmt.callable, TiledMma)]
        assert len(issues) == 36
        readers, pointers = {}, {}

        def dimension(value, group):
            if (literal := static_dim_value(value)) is not None:
                return literal
            if isinstance(value, Var):
                return dimension(visitor.bindings[id(value)], group)
            assert isinstance(value, Call)
            if isinstance(value.target, MeshCoord):
                assert value.target.mesh.names[static_dim_value(value.args[0])] == "group"
                return group
            a, b = (dimension(arg, group) for arg in value.args)
            if isinstance(value.target, DimAdd):
                return a + b
            assert isinstance(value.target, DimMul)
            return a * b

        def address(type_, coordinates, group):
            key = id(type_), group
            if key not in readers:
                eb = type_.dtype.bit_width // 8

                def reader(layout):
                    if isinstance(layout, ShardLayout):
                        return reader(layout.layout)
                    if isinstance(layout, ComposedLayout):
                        outer = reader(layout.outer)
                        offset = dimension(layout.offset, group) * eb
                        if isinstance(layout.inner, Swizzle):
                            return lambda linear: layout.inner(outer(linear) + offset)
                        return lambda linear: outer(linear) + offset
                    flat = flatten(layout)
                    grouped = unflatten(flat, compatible_profile(flat.shape, type_.shape))
                    return lambda linear: crd2idx(linear, grouped.shape, grouped.strides) * eb

                readers[key] = reader(type_.layout)
            linear = sum(
                coordinate * prod(type_.shape[axis + 1 :])
                for axis, coordinate in enumerate(coordinates)
            )
            return readers[key](linear)

        def pointer(value, group):
            key = id(value), group
            if key not in pointers:
                if isinstance(value, Var):
                    result = pointer(visitor.bindings[id(value)], group)
                elif isinstance(value, Constant):
                    result = value.value
                elif isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf)):
                    result = pointer(value.args[0], group)
                elif isinstance(value, Call) and isinstance(value.target, Slice):
                    base = value.args[0]
                    layout = base.type.layout
                    layout = layout.layout if isinstance(layout, ShardLayout) else layout
                    result = pointer(base, group)
                    if not isinstance(layout, ComposedLayout):
                        starts = tuple(dimension(start, group) for start in value.args[1].elements)
                        result += address(base.type, starts, group)
                else:
                    assert isinstance(value, Call) and type(value.target).__name__ == "AllocTensor"
                    result = 0
                pointers[key] = result
            return pointers[key]

        def pointer_root(value):
            if isinstance(value, Var):
                return pointer_root(visitor.bindings[id(value)])
            if isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf, Slice)):
                return pointer_root(value.args[0])
            return value

        c_root = pointer_root(issues[0].args[0])
        b_root = pointer_root(issues[0].args[2])
        copied_k = next(
            stmt.args[1]
            for stmt in visitor.statements
            if isinstance(stmt.callable, CopyAsyncTensor) and stmt.args[0] is function.params[1]
        )
        assert isinstance(b_root, Constant)
        assert b_root.value == pointer_root(copied_k).value
        assert c_root.type.shape == original_c.shape
        assert c_root.type.storage is StorageKind.RMEM
        for ordinal, issue in enumerate(issues):
            step, k = divmod(ordinal, 12)
            c, _, b = issue.args
            assert pointer_root(c) is c_root
            assert pointer_root(b) is b_root
            assert c.type.shape == (64, 16)
            assert b.type.shape == (16, 16)
            for group in range(2):
                for i in range(16):
                    n = 48 * group + 16 * step + i
                    for row in range(64):
                        assert pointer(c, group) + address(c.type, (row, i), group) == address(
                            original_c, (row, 16 * step + i), group
                        )
                    for j in range(16):
                        assert pointer(b, group) + address(
                            b.type, (j, i), group
                        ) == b_root.value + address(original_b, (16 * k + j, n), group)


@case
class WgmmaRepeatAlongNOrder(WgmmaPrice):
    path: Path = ROOT / "hir/wgmma_repeat_along_n_order.py"
    smem_peak: int = 24_576
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:256#0": RmemPeak(131_072, "128x256 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(131_072, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(131_072, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(131_072, "parent envelope of one alias chain"),
        }
    )
    wgmma_n: int = 64
    bound_by: str = "tensor"


@case
class WgmmaRsAFromAccumulator(WgmmaPrice):
    path: Path = ROOT / "hir/wgmma_rs_a_from_accumulator.py"
    smem_peak: int = 5_120
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:128#0": RmemPeak(4_096, "64x16 f32 p initializer"),
            "thread@128:128#1": RmemPeak(4_096, "p phi/first mma alias chain"),
            "thread@128:128#2": RmemPeak(
                10_240,
                "2048-byte narrowed p alias plus independent 8192-byte acc initializer",
            ),
            "thread@128:128#3": RmemPeak(
                10_240, "8192-byte acc phi/mma chain plus the live 2048-byte RS lhs"
            ),
            "thread@128:128#4": RmemPeak(8_192, "acc loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(10_240, "parent envelope of p/acc transition"),
        }
    )
    wgmma_n: int = 16
    bound_by: str = "smem"


@case
class WgmmaTma3Stage(ScheduleCase):
    path: Path = ROOT / "hir/wgmma_tma_3stage.py"
    smem_peak: int = 13_824
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@128:256#0": RmemPeak(8_192, "128x16 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        }
    )

    def guard(self, state, problems):
        result = state.result
        assert RegionMemoryMetadata in result.metadata_types
        lifetimes = get_metadata(result.function, RegionMemoryMetadata).lifetimes
        smem = sorted(item.bytes for item in lifetimes if item.memory_level == "smem")
        rmem = sorted(item.bytes for item in lifetimes if item.memory_level == "rmem")
        assert smem == sorted((512 * 3, 4096 * 3))
        assert rmem == [4096, 8192, 8192]


@case
class Sm89E4m3Gemm(ScheduleCase):
    path: Path = ROOT / "hir/sm89_e4m3_gemm.py"
    smem_peak: int = 512
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@0:32#0": RmemPeak(
                1_280,
                "512-byte f32 accumulator plus 512-byte lhs and 256-byte rhs e4m3 fragments",
            ),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        tilefoundry.compile(state.lowered)


@case
class ZeroTripInsertSlice(ScheduleCase):
    path: Path = ROOT / "hir/zero_trip_insert_slice.py"
    rmem_peaks: Mapping[str, RmemPeak] = MappingProxyType(
        {
            "thread@0:1#0": RmemPeak(16, "one four-element f32 update row"),
            "cta@0:1#0": RmemPeak(16, "parent envelope of the update row"),
        }
    )
    candidates_refusal: str | None = "source has no unscheduled candidate site"

    def guard(self, state, problems):
        lowered = state.lowered
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup(state.result.function.name).body)
        assert len(visitor.loops) == 1
        loop = visitor.loops[0]
        assert tuple(static_dim_value(value) for value in (loop.start, loop.stop, loop.step)) == (
            0,
            0,
            1,
        )


EXPECTED = {
    "hir/child_weight": dict(
        rmem_peaks={
            "thread@0:32#0": RmemPeak(384, "independent live Buffers: 3 * 128B"),
            "cta@0:1#0": RmemPeak(384, "independent live Buffers: 3 * 128B"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/cta_submesh_thread_scope": dict(
        rmem_peaks={
            "thread@0:128#0": RmemPeak(512, "4x32 f32 register tile"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/duplicate_output_storage": dict(
        smem_peak=8,
        finalize_refusal="return leaf (2,) shares storage with another output",
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/finalize_footprint_mesh_offset": dict(
        rmem_peaks={
            "thread@0:32#0": RmemPeak(136, "32 f32 row elements and one i64 mesh offset"),
            "cta@0:4#0": RmemPeak(8, "one i64 mesh offset"),
            "thread@0:32#1": RmemPeak(256, "32 f32 accumulator and loaded row elements"),
            "cta@0:4#1": RmemPeak(8, "one i64 output offset"),
        },
    ),
    "hir/inplace_output_update": dict(
        rmem_peaks={
            "cta@0:1#0": RmemPeak(16, "CTA envelope of the row register storage"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/loop_bound_from_outer_induction": dict(
        rmem_peaks={
            "thread@0:32#0": RmemPeak(256, "32 f32 accumulator and loaded row elements"),
        },
    ),
    "hir/loop_carry_from_param": dict(
        candidates_refusal="source has no unscheduled candidate site"
    ),
    "hir/loop_carry_rebind": dict(
        rmem_peaks={
            "thread@0:32#0": RmemPeak(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
            "cta@0:1#0": RmemPeak(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
        },
    ),
    "hir/loop_carry_window": dict(
        rmem_peaks={
            "thread@0:32#0": RmemPeak(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
            "cta@0:1#0": RmemPeak(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/loop_variable_rebound": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(64, "two independent live 2x4 f32 buffers"),
            "cta@0:1#0": RmemPeak(64, "parent envelope of the independent loop buffers"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/mesh_capture_outside": dict(candidates_refusal="source has no unscheduled candidate site"),
    "hir/mesh_views": dict(
        rmem_peaks={
            "thread@0:32#0": RmemPeak(256, "independent live Buffers: 2 * 128B"),
            "thread@0:32#1": RmemPeak(256, "independent live Buffers: 2 * 128B"),
            "thread@0:32#2": RmemPeak(256, "independent live Buffers: 2 * 128B"),
        },
    ),
    "hir/mesh_zeros_submesh": dict(
        smem_peak=128, candidates_refusal="source has no unscheduled candidate site"
    ),
    "hir/persistent_tile_union_count": dict(
        smem_peak=163_840,
        rmem_peaks={
            "thread@128:256#0": RmemPeak(131_072, "128x256 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(131_072, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(131_072, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(131_072, "parent envelope of one accumulator chain"),
        },
    ),
    "hir/register_arange": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(64, "independent 8-element i32 indices and output buffers"),
            "cta@0:1#0": RmemPeak(64, "parent envelope of the two independent buffers"),
        },
    ),
    "hir/register_full_like": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(65, "two independent 8-element f32 buffers and a predicate"),
            "cta@0:1#0": RmemPeak(65, "parent envelope of the independent buffers and predicate"),
        },
    ),
    "hir/register_index_window": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(
                32, "8 f32 output elements; index/view lifetimes reuse their storage"
            ),
            "cta@0:1#0": RmemPeak(32, "parent envelope of the register output"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/register_index_window_offset": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(
                32, "8 f32 output elements; index/view lifetimes reuse their storage"
            ),
            "cta@0:1#0": RmemPeak(32, "parent envelope of the register output"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/reshard_views": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(32, "8 f32 register output elements"),
            "cta@0:2#0": RmemPeak(32, "parent envelope of the output"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/scalar_binary": dict(
        rmem_peaks={
            "cta@0:1#0": RmemPeak(
                260,
                "independent live Buffers: 1 * 4B + 2 * 128B",
            ),
            "thread@0:32#0": RmemPeak(
                260,
                "independent live Buffers: 1 * 4B + 2 * 128B",
            ),
        },
    ),
    "hir/scalar_ceildiv_gate": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(32, "8 f32 register output elements"),
            "cta@0:4#0": RmemPeak(32, "parent envelope of the output"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/sm80_mma_ldmatrix": dict(
        smem_peak=1_536,
        rmem_peaks={
            "thread@32:32#0": RmemPeak(512, "16x8 f32 zero accumulator"),
            "thread@32:32#1": RmemPeak(
                1_280,
                "512-byte accumulator plus 512-byte lhs and 256-byte rhs fragments",
            ),
            "thread@32:32#2": RmemPeak(
                512, "analysis coalesces the f32 loop result and bf16 cast: max(512, 256)"
            ),
            "thread@0:64#0": RmemPeak(1_280, "parent envelope of accumulator/fragments"),
        },
    ),
    "hir/sparse_decode": dict(
        smem_peak=16_384,
        rmem_peaks={
            "cta@0:1#0": RmemPeak(
                520,
                "independent live Buffers: 1 * 8B + 1 * 512B",
            ),
            "thread@128:128#0": RmemPeak(33_024, "attention intermediates and row reductions"),
            "thread@0:256#0": RmemPeak(33_024, "parent envelope of attention compute"),
        },
    ),
    "hir/staged_dynamic_inner": dict(
        smem_peak=32,
        rmem_peaks={
            "thread@0:32#0": RmemPeak(
                16, "the 1x8 bf16 accumulator and insert aliases form one 16-byte group"
            ),
            "cta@0:1#0": RmemPeak(16, "parent envelope of the 1x8 bf16 accumulator"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/staged_nested": dict(
        smem_peak=32,
        rmem_peaks={
            "thread@0:32#0": RmemPeak(
                96, "the 6x8 bf16 accumulator and its insert aliases form one 96-byte group"
            ),
            "cta@0:1#0": RmemPeak(
                96, "the 6x8 bf16 accumulator and its insert aliases form one 96-byte group"
            ),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/strided_return_window": dict(
        rmem_peaks={
            "thread@0:1#0": RmemPeak(12288, "24 * 256 bf16 register output elements"),
            "cta@0:32#0": RmemPeak(12288, "parent envelope of the register output"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/submesh_slice_in_sibling": dict(
        smem_peak=32, candidates_refusal="source has no unscheduled candidate site"
    ),
    "hir/wgmma_a_k_major": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_a_mn_major": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_accumulator_stmatrix": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@0:128#0": RmemPeak(
                12_288, "64x48 f32 accumulator; bf16 cast aliases the fragment"
            ),
        },
    ),
    "hir/wgmma_b_mn_major_one_atom_column_slice": dict(
        smem_peak=49_152,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(16384, "64x64 f32 accumulator; MMA aliases acc"),
            "thread@0:256#0": RmemPeak(16384, "parent envelope of the accumulator"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/wgmma_b_mn_major_two_atom_column_slice": dict(
        smem_peak=49_152,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(32768, "64x128 f32 accumulator; MMA aliases acc"),
            "thread@0:256#0": RmemPeak(32768, "parent envelope of the accumulator"),
        },
        candidates_refusal="source has no unscheduled candidate site",
    ),
    "hir/wgmma_b_nk": dict(
        smem_peak=2_560,
        rmem_peaks={
            "thread@0:128#0": RmemPeak(
                4096, "64x16 f32 C fragment, shared by zeros and MMA result"
            ),
        },
    ),
    "hir/wgmma_cp_async_loads": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_cta_grid_4x17": dict(
        smem_peak=13_824,
        rmem_peaks={
            "thread@128:256#0": RmemPeak(8_192, "128x16 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_explicit_windows": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_k_slices_of_wide_run": dict(
        smem_peak=12_288,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_one_tile_of_larger_output": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_repeat_along_k": dict(
        smem_peak=36_864,
        rmem_peaks={
            "thread@128:256#0": RmemPeak(8_192, "128x16 f32 zero accumulator"),
            "thread@128:256#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:256#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:384#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_rs_a_from_smem": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(
                10_240,
                "8192-byte accumulator alias chain plus independent 2048-byte A fragment",
            ),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(10_240, "parent envelope of accumulator/A fragment"),
        },
    ),
    "hir/wgmma_swizzled_smem": dict(
        smem_peak=6_144,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "f32 phi/mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of one alias chain"),
        },
    ),
    "hir/wgmma_two_schedules": dict(
        smem_peak=12_288,
        rmem_peaks={
            "thread@128:128#0": RmemPeak(8_192, "64x32 f32 zero accumulator"),
            "thread@128:128#1": RmemPeak(8_192, "phi/first mma alias chain"),
            "thread@128:128#2": RmemPeak(8_192, "first/second mma alias chain"),
            "thread@128:128#3": RmemPeak(8_192, "f32 loop result/bf16 cast alias"),
            "thread@0:256#0": RmemPeak(8_192, "parent envelope of both mma chains"),
        },
    ),
    "plain/chunk_rmsnorm": dict(
        sites=SiteCounts(0, 7, detailed=True, binaries=5), dims={"chunks": 16}
    ),
    "plain/fp8_block_scaled_gemm": dict(sites=SiteCounts(1, 5, detailed=True)),
    "plain/gemm_8192x17408x5120_cta_grid": dict(sites=SiteCounts(1, 3, detailed=True)),
    "plain/gemm_relu_gemm_tiled": dict(sites=SiteCounts(2, 2, detailed=True)),
    "plain/gemm_relu_gemm_untiled": dict(sites=SiteCounts(2, 0, detailed=True)),
    "plain/reshard_locations": dict(sites=SiteCounts(0, 4)),
    "plain/sm89_e4m3_gemm": dict(sites=SiteCounts(1, 3, target="nvidia.rtx_6000_ada")),
    "plain/sparse_decode": dict(sites=SiteCounts(2, 4)),
    "plain/unit_mesh_axis_loop_carry": dict(sites=SiteCounts(0, 2)),
    "plain/gemm_relu_gemm_smem_staged": dict(
        refusal=r"no nvidia\.h200_sxm MMA reads lhs f32 smem and rhs f32 smem into f32"
        r"(.|\n)*gemm_relu_gemm_smem_staged\.py:39:29"
    ),
}

CASES = cases(
    ROOT / "hir",
    ROOT / "hir" / "dim",
    ROOT / "plain",
    ROOT / "tir",
    case_type=ScheduleCase,
    expected=EXPECTED,
)
STAGES = {"hir": _hir, "plain": _plain, "tir": _tir, "dim": _dim}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: ScheduleCase) -> None:
    problems = Problems(case)
    STAGES[case.path.parent.name](case, problems)
    problems.dump()
