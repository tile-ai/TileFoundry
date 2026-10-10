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
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

import pytest
import torch

import tilefoundry.passes.transforms.convert_hir_to_tir as lowering_module
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program, resolve_program_geometry
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
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.inspection import PatternPrinter, as_script
from tilefoundry.ir.core import Call, Op, OpCapability, Var, detach_metadata, get_metadata
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef, collect_param_defs
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, operand_relations
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.pattern import (
    PatternMatcher,
    TensorPattern,
    declared_execution_mesh,
    is_ranked_tensor,
)
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.cuda.nn.wgmma import Form, Wgmma
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    ShardLayout,
    StorageKind,
    TensorType,
)
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import levels, starts
from tilefoundry.ir.visitor import StmtVisitor, collect_exprs
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    identity_relations,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry.visitor_registry.verify import verify_prim_function

PLAIN = (
    "chunk_rmsnorm",
    "fp8_block_scaled_gemm",
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
)
PLAIN_DIMS = {"chunk_rmsnorm": {"chunks": 16}}
PLAIN_REFUSED = {
    "gemm_relu_gemm_smem_staged": (
        r"no nvidia\.h200_sxm MMA reads lhs f32 smem and rhs f32 smem into f32"
        r"(.|\n)*gemm_relu_gemm_smem_staged\.py:41:29"
    ),
}
TIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "tir").glob("*.py")))
HIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "hir").glob("*.py")))
WGMMA_FACTS = Path(__file__).parents[1] / "fixtures" / "schedule" / "Wgmma.facts.txt"
CANDIDATE_GOLDEN = (
    Path(__file__).parents[1]
    / "fixtures"
    / "schedule"
    / "plain"
    / "gemm_8192x17408x5120_cta_grid.candidates.txt"
)
ANALYZED_GOLDEN = (
    Path(__file__).parents[1]
    / "fixtures"
    / "schedule"
    / "hir"
    / "gemm_8192x17408x5120_tma_store.analyzed.txt"
)
ANALYSES = (
    ("compute-cost", ComputeCostMetadata),
    ("memory", MemoryMetadata),
    ("roofline", RooflineMetadata),
    ("performance", PerformanceSummaryMetadata),
)
_TENSOR_CLOCK_HZ = 1_830_000_000
_SHARED_BYTES_PER_CLOCK = 128


@dataclass(frozen=True)
class _RmemExpectation:
    peak_bytes: int
    derivation: str


SMEM_GOLDEN = {
    "scalar_binary": 0,
    "fp8_block_scaled_gemm": 65_536,
    "gemm_8192x17408x5120_register_store": 196_608,
    "gemm_8192x17408x5120_tma_store": 212_992,
    "sm80_mma_ldmatrix": 1_536,
    "wgmma_a_k_major": 6_144,
    "wgmma_a_mn_major": 6_144,
    "wgmma_cast_between_schedules": 6_144,
    "wgmma_cp_async_loads": 6_144,
    "wgmma_cta_grid_4x17": 13_824,
    "wgmma_explicit_windows": 6_144,
    "wgmma_k_slices_of_wide_run": 12_288,
    "wgmma_one_tile_of_larger_output": 6_144,
    "wgmma_repeat_along_k": 36_864,
    "wgmma_repeat_along_n_order": 24_576,
    "wgmma_rs_a_from_accumulator": 5_120,
    "wgmma_rs_a_from_smem": 6_144,
    "wgmma_swizzled_smem": 6_144,
    "wgmma_tma_3stage": 13_824,
    "wgmma_two_schedules": 12_288,
}

RMEM_EXPECTED = {
    "scalar_binary": {
        "thread@0:32#0": _RmemExpectation(
            132,
            "rhs literal materializes in rmem during lowering; the new result reuses "
            "the same register tile, so the HIR peak stays unchanged",
        ),
    },
    "fp8_block_scaled_gemm": {
        "thread@128:256#0": _RmemExpectation(65_536, "128x128 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(
            66_048,
            "128x128 f32 block product and its scaled aliases plus the 512-byte 128x1 row scale",
        ),
        "thread@128:256#2": _RmemExpectation(65_536, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(66_048, "parent envelope of the block-scaling region"),
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
        "thread@32:32#2": _RmemExpectation(
            768, "512-byte f32 loop result plus 256-byte bf16 cast result"
        ),
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
        "thread@128:128#2": _RmemExpectation(
            8_448, "f32 loop result plus f32 reduction results and bf16 cast alias"
        ),
        "thread@0:32#1": _RmemExpectation(
            3_072,
            "16x32 f32 loader tile (2048) plus bf16 cast tile (1024)",
        ),
        "thread@0:256#0": _RmemExpectation(
            11_264,
            "64x32 f32 accumulator (8192) plus overlapping loader tiles (3072)",
        ),
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
    "wgmma_repeat_along_n_order": {
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
        "thread@128:128#3": _RmemExpectation(
            10_240, "8192-byte acc phi/mma chain plus the live 2048-byte RS lhs"
        ),
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


class _OperandMatchVisitor(StmtVisitor[None]):
    """Exercise operand declarations directly, independently of the report renderer."""

    def __init__(self) -> None:
        self.matches: list[tuple[object, str, dict]] = []

    def visit_Evaluate(self, stmt: Evaluate) -> None:
        op = stmt.callable
        schema = getattr(type(op), "_op_schema", None)
        if schema is None:
            return
        inputs = tuple(param for param in schema.signature if param.kind == "input")
        supplied_optional = len(stmt.args) - sum(not param.optional for param in inputs)
        optional = tuple(param for param in inputs if param.optional)
        included = {id(param) for param in optional[: max(0, supplied_optional)]}
        inputs = tuple(
            param for param in inputs if not param.optional or id(param) in included
        )
        atom = getattr(op, "atom", None)
        for param, arg in zip(inputs, stmt.args, strict=True):
            if param.pattern is None:
                continue
            pattern = (
                param.pattern.read_on(op)
                if hasattr(param.pattern, "read_on")
                else param.pattern
            )
            matcher = PatternMatcher(dict(getattr(atom, "bindings", {})))
            assert matcher.match(pattern, arg.type)
            assert matcher.solve()
            self.matches.append((op, param.name, dict(matcher.bindings)))


def _direct_operand_matches(function: PrimFunction) -> list[tuple[object, str, dict]]:
    visitor = _OperandMatchVisitor()
    visitor.visit(function.body)
    return visitor.matches


@pytest.mark.parametrize("name", PLAIN)
def test_plain_program_is_analyzable(name: str) -> None:
    if name in PLAIN_REFUSED:
        with pytest.raises(VerifyError, match=PLAIN_REFUSED[name]):
            importlib.import_module(f"tests.fixtures.schedule.plain.{name}")
        return
    module = importlib.import_module(f"tests.fixtures.schedule.plain.{name}")
    program = next(value for value in vars(module).values() if type(value).__name__ == "Module")
    entry = program.entry_function()
    dims = PLAIN_DIMS.get(name)
    concrete, concrete_entry = resolve_program_geometry(
        program, entry, dims, TypeInferContext(scope=FunctionScope(program, entry))
    )
    check_program(concrete, concrete_entry)
    result = analyze(program, entry, analysis=("memory", "performance"), dims=dims)
    assert result.metadata_types


@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_scheduled_hir_program_is_well_typed(path: Path) -> None:
    program = _module_in(path)
    entry = next(function for function in program.functions if function.name == "gemm")
    check_program(program, entry)


def test_scheduled_hir_structural_calls_declare_layouts() -> None:
    structural_calls = []
    for path in HIR:
        program = _module_in(path)
        entry = next(function for function in program.functions if function.name == "gemm")
        check_program(program, entry)
        structural_calls.extend(
            (path.stem, type(expr.target).__name__, expr)
            for expr in collect_exprs(entry.body)
            if isinstance(expr, Call)
            and isinstance(expr.target, (Reshape, Slice, Transpose))
        )
    missing = [(path, op) for path, op, expr in structural_calls if expr.type.layout is None]
    assert not missing, (
        f"{len(missing)} of {len(structural_calls)} structural calls omit layout: {missing}"
    )


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

    if analysis == "memory":
        for expr in collect_exprs(result.function.body):
            if not isinstance(expr, Call):
                continue
            record = get_metadata(expr, MemoryMetadata)
            if aliased_operand(expr) is not None:
                assert record.buffer_bytes is None
                assert not record.offsets
            elif isinstance(expr.target, Transpose):
                assert record.buffer_bytes == prod(expr.type.shape) * expr.type.dtype.bit_width // 8
                assert record.offsets
            if isinstance(expr.target, ScheduleOp) and isinstance(
                expr.target.op, (Copy, CopyAsync, CopyAsyncTensor)
            ):
                tile = prod(expr.type.shape) * expr.type.dtype.bit_width // 8
                assert record.operands[-1].write == tile, "a copy writes its whole tile"
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

    if analysis == "memory" and path.stem == "wgmma_tma_3stage":
        assert RegionMemoryMetadata in result.metadata_types
        lifetimes = get_metadata(result.function, RegionMemoryMetadata).lifetimes
        smem = sorted(item.bytes for item in lifetimes if item.memory_level == "smem")
        rmem = sorted(item.bytes for item in lifetimes if item.memory_level == "rmem")
        assert smem == sorted((512 * 3, 4096 * 3))
        assert rmem == [4096, 8192, 8192, 8192]

    if analysis == "memory":
        placement = get_metadata(result.function, RegionMemoryMetadata)
        assert placement is not None
        liveness = analyze_liveness(result.function)
        regions = tuple(window.region for window in liveness.regions)
        region_records = tuple(
            record
            for region in regions
            if (record := get_metadata(region, RegionMemoryMetadata)) is not None
        )
        assert region_records
        if path.stem == "gemm_8192x17408x5120_tma_store":
            assert len(region_records) == 4
        for record in region_records:
            assert record.solver_status == "feasible"
            assert record.topologies == placement.topologies
            assert record.peaks
            assert all(peak.memory_level == "rmem" for peak in record.peaks)
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
        if loop_bounds:
            assert outside_uses

        for interval in liveness.intervals:
            if result_copies(interval.value) == 1:
                continue
            assert (interval.defined_at, interval.last_used_at) in loop_bounds

@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_schedule_memory_report_carries_allocations(
    path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{path.stem}.json"

    assert cli_main(["analyze", str(path), str(out), "--memory", "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    report = json.loads(out.read_text())
    memory = report["function_records"]["memory"]
    peaks = {item["memory_level"]: item["peak_bytes"] for item in memory["peaks"]}
    program = _module_in(path)
    has_smem = any(
        isinstance(expr.type, TensorType) and expr.type.storage is StorageKind.SMEM
        for expr in collect_exprs(program.entry_function().body)
    )
    if has_smem:
        assert peaks["smem"] == SMEM_GOLDEN[path.stem]
    else:
        assert "smem" not in peaks
        assert SMEM_GOLDEN[path.stem] == 0
    assert memory["solver_status"] == "feasible"
    for row in report["calls"]:
        allocation = row["memory"]
        if "operands" not in allocation:
            continue
        result = next(
            operand for operand in allocation["operands"] if operand["arg"] == "result"
        )
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

@pytest.mark.parametrize(
    ("fixture", "n", "bound_by"),
    (
        ("wgmma_rs_a_from_accumulator", 16, "smem"),
        ("wgmma_repeat_along_n_order", 64, "tensor"),
        ("gemm_8192x17408x5120_register_store", 256, "tensor"),
    ),
)
def test_wgmma_performance_prices_the_slower_of_tensor_clocks_and_shared_reads(
    fixture: str, n: int, bound_by: str
) -> None:
    """Price real timeline output against independent tensor-clock and smem references.

    An issue takes ``n / 2`` tensor clocks; its shared tiles cross at 128 bytes
    per SM clock (SM90 microbenchmark, arXiv 2402.13499), both at the 1.83 GHz
    of the stage2c throughput research rather than the peaks under test. The
    slower one is the time. The n=16 call reads 2560 shared bytes for 8 clocks of
    work, so shared reads decide it; the wider calls stay tensor-bound.
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
    shared_bytes = sum(
        prod(operand.type.shape) * operand.type.dtype.bit_width // 8
        for operand in schedule.args
        if operand.type.storage is StorageKind.SMEM
    )
    tensor_ns = -(-(n * issues * 1_000_000_000) // (2 * _TENSOR_CLOCK_HZ))
    shared_ns = -(-(shared_bytes * 1_000_000_000) // (_SHARED_BYTES_PER_CLOCK * _TENSOR_CLOCK_HZ))

    assert flops_per_issue == 2 * 64 * n * 16
    assert (shared_ns > tensor_ns) == (bound_by == "smem")
    assert duration_ns == max(tensor_ns, shared_ns)


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


@pytest.mark.parametrize(
    ("atom", "dtype", "shape"),
    (
        pytest.param(Mma(), DType.bf16, (16, 16, 8), id="sm80_bf16"),
        pytest.param(
            Wgmma(n=8, dtype="fp8e4m3", form=Form.SS), DType.fp8e4m3, (64, 32, 8), id="fp8_ss"
        ),
        pytest.param(
            Wgmma(n=8, dtype="fp8e4m3", form=Form.RS), DType.fp8e4m3, (64, 32, 8), id="fp8_rs"
        ),
    ),
)
def test_schedule_mma_evaluates_like_matmul_plus_accumulator(atom, dtype, shape) -> None:
    """Products are summed in the accumulator dtype, not rounded to the operand dtype first.

    Column 0 of lhs holds 2**-8, so each sum carries bits an operand-dtype result drops.
    """
    m, k, n = shape
    operand = to_torch_dtype(dtype)
    acc_type = TensorType((m, n), DType.f32, None, StorageKind.RMEM)
    lhs_type = TensorType((m, k), dtype, None, StorageKind.RMEM)
    rhs_type = TensorType((k, n), dtype, None, StorageKind.RMEM)
    acc = torch.arange(m * n, dtype=torch.float32).reshape(m, n)
    lhs = (torch.arange(m * k).reshape(m, k) % 5).float()
    lhs[:, 0] = 2.0**-8
    lhs = lhs.to(operand)
    rhs = (torch.arange(k * n).reshape(k, n) % 3).to(operand)

    scheduled = _evaluate_call(
        ScheduleOp(op=TiledMma(atom=atom)),
        ((acc_type, acc), (lhs_type, lhs), (rhs_type, rhs)),
        acc_type,
    )

    exact = lhs.float() @ rhs.float()
    assert torch.equal(scheduled, acc + exact)
    assert not torch.equal(scheduled, acc + exact.to(operand).float())


def test_fp8_block_scaled_gemm_matches_its_block_scaled_reference() -> None:
    """The scheduled program equals sum_kb (A_kb @ B_kb) * a_scale[:, kb] * b_scale[kb, :].

    Each 128-wide K block's FP8 product is exact in f32, then scaled by that
    block's own positive row and tile scales; the scales differ across blocks,
    so scaling a running total instead of each block would not match.
    """
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "fp8_block_scaled_gemm.py"
    module = _module_in(source)
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


def test_single_issue_schedule_preserves_instruction_relations() -> None:
    schedule = _copy_schedule_call(repeat=(1,), order=(0,))
    source = schedule.args[0]
    destination_type = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.SMEM)
    inferred = TypeInferContext()
    inference_type(schedule, inferred)
    scheduled = relations_of(schedule, inferred)
    source, _destination, *result = operand_relations(
        schedule.target.op, (source.type, destination_type)
    )
    single = (source, *result)
    assert len(scheduled) == len(single)
    assert all(
        left.relation.is_equal(right.relation)
        for left, right in zip(scheduled, single, strict=True)
    )


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (_copy_schedule_call(repeat=(2,)), "transfer tiling is not yet supported"),
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
        pattern=is_ranked_tensor(),
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


def test_schedule_typeinfer_requires_instruction_verifier(monkeypatch) -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    monkeypatch.setitem(
        access_relation_registry._map,
        _UnstatedInstruction,
        identity_relations,
    )

    with pytest.raises(ValueError, match="has no registered verifier"):
        inference_type(call)


def test_schedule_typeinfer_requires_whole_instruction_tiles(monkeypatch) -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    param = _UnstatedInstruction._op_schema.signature[0]
    monkeypatch.setattr(param, "pattern", TensorPattern(shape=(3,)))
    monkeypatch.setitem(
        access_relation_registry._map,
        _UnstatedInstruction,
        identity_relations,
    )

    with pytest.raises(
        ValueError,
        match="iteration extent 4 is not divisible by single-issue extent 3",
    ):
        inference_type(call)


def test_schedule_typeinfer_requires_declared_write_type(monkeypatch) -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(),
        type=type_,
    )
    param = _UnstatedInstruction._op_schema.signature[0]
    monkeypatch.setattr(param, "effect", MemoryEffect.WRITE)
    monkeypatch.setattr(param, "pattern", None)

    with pytest.raises(
        ValueError,
        match="value is write-only and declares no result shape",
    ):
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
    source = path.read_text()
    assert as_script(function) == source[source.index("from __future__ import annotations") :]


def test_parameter_structure_does_not_repeat_its_name() -> None:
    declarations = {
        schema.op_class
        for schema in iter_schemas()
        if schema.op_class is not None
    }
    for schema in iter_schemas():
        if schema.op_class is None:
            continue
        stated = vars(schema.op_class).get("capability")
        capabilities = (stated,) if isinstance(stated, OpCapability) else stated or ()
        declarations.update(
            capability.declaration
            for capability in capabilities
            if capability.declaration is not None
        )
    printer = PatternPrinter()
    for declaration in declarations:
        parameters = tuple(getattr(declaration, "parameters", ())) or collect_param_defs(
            declaration
        )
        for parameter in parameters:
            if parameter.pattern is not None:
                assert printer.written(parameter.pattern, parameter.name) != parameter.name, (
                    f"{declaration.__name__}.{parameter.name} puts a predicate in its "
                    "structural pattern slot"
                )


def test_instruction_requires_an_execution_mesh_declaration() -> None:
    with pytest.raises(ValueError, match="_UnstatedInstruction execution_mesh must be"):
        declared_execution_mesh(_UnstatedInstruction)


def test_schedule_facts_writes_wgmma_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "facts.txt"

    assert (
        cli_main(
            [
                "schedule",
                "facts",
                "T.cuda.sm90.Wgmma",
                "--target",
                "nvidia.h200_sxm",
                str(out),
            ]
        )
        == 0
    )
    assert capsys.readouterr() == ("", "")
    assert out.read_bytes() == WGMMA_FACTS.read_bytes()


def test_schedule_facts_lists_target_instructions_as_text_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text_out = tmp_path / "facts.txt"
    json_out = tmp_path / "facts.json"
    args = ["schedule", "facts", "--target", "nvidia.h200_sxm"]

    assert cli_main([*args, str(text_out)]) == 0
    assert cli_main([*args, str(json_out), "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    expected = {
        "target": "nvidia.h200_sxm",
        "instructions": [
            {"id": "T.binary", "capability": None},
            {"id": "T.cast", "capability": None},
            {"id": "T.clamp", "capability": None},
            {"id": "T.unary", "capability": None},
            {"id": "T.copy_async", "capability": "cp.async"},
            {
                "id": "T.copy_async_tensor",
                "capability": "cp.async.bulk.tensor",
            },
            {"id": "T.copy", "capability": None},
            {"id": "T.ldmatrix", "capability": "ldmatrix"},
            {"id": "T.relu", "capability": None},
            {
                "id": "T.tiled_mma",
                "capability": ["wgmma.mma_async", "mma.sync"],
            },
            {"id": "T.reduce", "capability": None},
        ],
    }
    assert json.loads(json_out.read_text()) == expected
    assert (
        text_out.read_text()
        == """\
target nvidia.h200_sxm
instructions
  T.binary             all targets
  T.cast               all targets
  T.clamp              all targets
  T.unary              all targets
  T.copy_async         cp.async
  T.copy_async_tensor  cp.async.bulk.tensor
  T.copy               all targets
  T.ldmatrix           ldmatrix
  T.relu               all targets
  T.tiled_mma          wgmma.mma_async, mma.sync
  T.reduce             all targets
"""
    )


@pytest.mark.parametrize("target", ("cpu", "apple.m2_pro"))
def test_schedule_facts_only_lists_target_neutral_instructions_for_non_cuda(
    target: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "facts.txt"

    assert cli_main(["schedule", "facts", "--target", target, str(out)]) == 0
    assert capsys.readouterr() == ("", "")
    assert out.read_text() == f"""target {target}
instructions
  T.binary  all targets
  T.cast    all targets
  T.clamp   all targets
  T.unary   all targets
  T.copy    all targets
  T.relu    all targets
  T.reduce  all targets
"""


@pytest.mark.parametrize(
    ("selection", "message"),
    (
        (
            ["T.cuda.sm90.Unknown", "--target", "nvidia.h200_sxm"],
            "unknown instruction 'T.cuda.sm90.Unknown'",
        ),
        (
            ["T.cuda.sm90.Wgmma", "--target", "nvidia.unknown"],
            "unknown target identity 'nvidia.unknown'",
        ),
    ),
    ids=("instruction", "target"),
)
def test_schedule_facts_rejects_unknown_selection_without_output(
    selection: list[str],
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "facts.txt"

    assert cli_main(["schedule", "facts", *selection, str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("tilefoundry: error: ")
    assert message in captured.err
    assert not out.exists()


def test_schedule_candidates_writes_canonical_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    name = CANDIDATE_GOLDEN.name.removesuffix(".candidates.txt")
    source = f"tests/fixtures/schedule/plain/{name}.py"
    out = tmp_path / "candidates.txt"

    assert cli_main(["schedule", "candidates", source, str(out)]) == 0
    assert capsys.readouterr() == ("", "")
    assert out.read_bytes() == CANDIDATE_GOLDEN.read_bytes()


def test_schedule_candidate_reports_cover_every_site(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = []
    for name in (name for name in PLAIN if name not in PLAIN_REFUSED):
        source = f"tests/fixtures/schedule/plain/{name}.py"
        out = tmp_path / f"{name}.json"
        dims = [f"--dim={key}={value}" for key, value in PLAIN_DIMS.get(name, {}).items()]
        assert cli_main(["schedule", "candidates", source, str(out), "--json", *dims]) == 0
        reports.append((name, json.loads(out.read_text())))

    assert capsys.readouterr() == ("", "")
    sites = [(name, row) for name, report in reports for row in report["lines"]]
    assert all(row["candidates"] or row["refused"] for _name, row in sites)
    assert all(row["candidates"] for _name, row in sites if row["op"] == "tf.reshard")
    matmuls = [(name, row) for name, row in sites if row["op"] == "tf.matmul"]
    assert len(matmuls) == 6
    assert [name for name, row in matmuls if row["candidates"]] == [
        "fp8_block_scaled_gemm",
        "gemm_8192x17408x5120_cta_grid",
    ]


@pytest.mark.parametrize(
    ("name", "dims", "matmuls", "reshards", "accepted_matmuls"),
    (
        ("chunk_rmsnorm", ("chunks=16",), 0, 7, 0),
        ("chunk_rmsnorm", ("chunks=32",), 0, 7, 0),
        ("fp8_block_scaled_gemm", (), 1, 5, 1),
        ("gemm_8192x17408x5120_cta_grid", (), 1, 4, 1),
        ("gemm_relu_gemm_tiled", (), 2, 2, 0),
        ("gemm_relu_gemm_untiled", (), 2, 0, 0),
    ),
)
def test_schedule_candidates_reports_every_plain_site(
    name: str,
    dims: tuple[str, ...],
    matmuls: int,
    reshards: int,
    accepted_matmuls: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = f"tests/fixtures/schedule/plain/{name}.py"
    out = tmp_path / f"{name}.json"

    assert (
        cli_main(
            ["schedule", "candidates", source, str(out), "--json", *(f"--dim={d}" for d in dims)]
        )
        == 0
    )
    assert capsys.readouterr() == ("", "")
    report = json.loads(out.read_text())
    assert report["source"] == source
    assert report["target"] == "nvidia.h200_sxm"
    matmul_rows = [row for row in report["lines"] if row["op"] == "tf.matmul"]
    reshard_rows = [row for row in report["lines"] if row["op"] == "tf.reshard"]
    assert (len(matmul_rows), len(reshard_rows)) == (matmuls, reshards)
    assert sum(bool(row["candidates"]) for row in matmul_rows) == accepted_matmuls
    assert all(row["candidates"] or row["refused"] for row in report["lines"])
    assert all(row["candidates"] for row in reshard_rows)

    if name == "fp8_block_scaled_gemm":
        (matmul,) = matmul_rows
        assert [candidate["id"] for candidate in matmul["candidates"]] == ["T.cuda.sm90.Wgmma"]
        (wgmma,) = matmul["candidates"]
        assert wgmma["bindings"]
        assert all(
            binding.endswith(", dtype=fp8e4m3, form=SS") for binding in wgmma["bindings"]
        )

    if name == "chunk_rmsnorm":
        binaries = [row for row in report["lines"] if row["op"] == "tf.binary"]
        assert len(binaries) == 5
        assert all(any(c["id"] == "T.binary" for c in row["candidates"]) for row in binaries)
        assert all(row["op"] != "tf.schedule" for row in report["lines"])
        if dims == ("chunks=16",):
            invalid_out = tmp_path / "invalid.json"
            assert (
                cli_main(["schedule", "candidates", source, str(invalid_out), "--dim=chunks=33"])
                != 0
            )
            assert "[1, 32]" in capsys.readouterr().err
            assert not invalid_out.exists()
            assert cli_main(["schedule", "candidates", source, str(invalid_out)]) != 0
            error = capsys.readouterr().err
            assert "chunks is declared as [1, 32]" in error
            assert "bind it with --dim" in error
            assert not invalid_out.exists()


@pytest.mark.parametrize("source", HIR, ids=lambda path: path.stem)
def test_schedule_candidates_omit_selected_schedule_calls(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{source.stem}.json"

    assert cli_main(["schedule", "candidates", str(source), str(out), "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    report = json.loads(out.read_text())
    assert report["lines"]
    ops = [row["op"] for row in report["lines"]]
    assert "tf.matmul" not in ops
    assert "tf.reshard" not in ops
    assert any(
        op in {"tf.binary", "tf.cast", "tf.clamp", "tf.relu", "tf.reduce", "tf.unary"}
        for op in ops
    )
    if source.stem == "wgmma_cast_between_schedules":
        assert ops.count("tf.reduce") == 1


def test_operand_match_refusal_names_the_failed_pattern() -> None:
    function = _prim_in(
        Path(__file__).parents[1]
        / "fixtures"
        / "schedule"
        / "tir"
        / "wgmma_a_k_major.py"
    )
    rejected = None

    class RejectOne(StmtVisitor[None]):
        def visit_Evaluate(self, stmt: Evaluate) -> None:
            nonlocal rejected
            if rejected is not None or not isinstance(getattr(stmt.callable, "atom", None), Wgmma):
                return
            op = stmt.callable
            lhs = next(param for param in type(op)._op_schema.signature if param.name == "lhs")
            pattern = lhs.pattern.read_on(op)
            matcher = PatternMatcher(dict(op.atom.bindings))
            assert not matcher.match(pattern, replace(stmt.args[1].type, storage=StorageKind.GMEM))
            rejected = PatternPrinter().refusal(matcher.refusal)

    RejectOne().visit(function.body)
    assert rejected is not None
    assert "StorageKind.GMEM" in rejected
    assert "StorageKind.SMEM" in rejected


def test_schedule_analyze_writes_memory_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = "tests/fixtures/schedule/hir/gemm_8192x17408x5120_tma_store.py"
    out = tmp_path / "analyzed.txt"

    assert cli_main(["analyze", source, str(out), "--memory"]) == 0
    assert capsys.readouterr() == ("", "")
    assert out.read_bytes() == ANALYZED_GOLDEN.read_bytes()


@pytest.mark.parametrize("source", HIR, ids=lambda path: path.stem)
def test_schedule_finalize_writes_verified_tir(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / source.name
    expected = source.parent.parent / "tir" / source.name

    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert out.read_bytes() == expected.read_bytes()
    found = _direct_operand_matches(_prim_in(out))
    assert found
    wgmma_lhs = [
        captures
        for op, name, captures in found
        if isinstance(getattr(op, "atom", None), Wgmma) and name == "lhs"
    ]
    assert all("a_major" in captures for captures in wgmma_lhs)
    assert all(
        "a_swizzle" in captures
        for captures in wgmma_lhs
        if captures["form"].name == "SS"
    )


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


def test_lowering_reports_smem_placement_above_capacity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (
        Path(__file__).parents[1]
        / "fixtures"
        / "schedule"
        / "hir"
        / "gemm_8192x17408x5120_tma_store.py"
    )
    four_stage = tmp_path / source.name
    four_stage.write_text(source.read_text().replace("STAGES = 3", "STAGES = 4", 1))

    out = tmp_path / "finalized.py"
    assert cli_main(["schedule", "finalize", str(four_stage), str(out)]) == 0
    assert capsys.readouterr() == ("", "")
    assert 'error="smem placement peak 256.00KB exceeds capacity 227.00KB"' in out.read_text()
    verify_prim_function(_prim_in(out))


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


def test_lowering_rejects_unscheduled_gmem_cast(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    cast = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, HirCast)
    )
    cast.type = replace(cast.type, storage=StorageKind.GMEM)
    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(
        source,
        "T.cast accepts only rmem operands, so write an explicit tf.schedule",
        tmp_path,
        capsys,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("group", "is not outer modes followed by participant frame"),
        ("atom", "axis n extent 17 is not divisible by atom 16"),
        ("row", "axis k extent 80 is not divisible by row 16 * 4"),
    ),
)
def test_lowering_rejects_invalid_atom_geometry(
    case: str,
    message: str,
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_repeat_along_k.py"
    module = _module_in(source)
    result = analyze(module, module.entry_function(), analysis=("memory",))
    emit = lowering_module.Lowering._emit_instruction

    def malformed(self, call, op, operands, mesh, output_window, written, cursor):
        if getattr(op, "atom", None) is None:
            return emit(self, call, op, operands, mesh, output_window, written, cursor)
        acc, lhs, rhs = (value for _role, value in operands)
        acc_type = self.logical.get(id(acc), acc.type)
        assert isinstance(acc_type.layout, ShardLayout)
        if case == "atom":
            acc_type = replace(acc_type, shape=(acc_type.shape[0], 17))
            rhs_type = self.logical.get(id(rhs), rhs.type)
            self.logical[id(rhs)] = replace(rhs_type, shape=(rhs_type.shape[0], 17))
        elif case == "row":
            lhs_type = self.logical.get(id(lhs), lhs.type)
            assert isinstance(lhs_type.layout, ComposedLayout)
            assert isinstance(lhs_type.layout.outer, Layout)
            lhs_outer = replace(
                lhs_type.layout.outer,
                shape=(*lhs_type.layout.outer.shape[:-1], (5, 16)),
            )
            self.logical[id(lhs)] = replace(
                lhs_type,
                shape=(lhs_type.shape[0], 80),
                layout=replace(lhs_type.layout, outer=lhs_outer),
            )
            rhs_type = self.logical.get(id(rhs), rhs.type)
            assert isinstance(rhs_type.layout, Layout)
            rhs_layout = replace(
                rhs_type.layout,
                shape=((5, *rhs_type.layout.shape[0][1:]), *rhs_type.layout.shape[1:]),
            )
            self.logical[id(rhs)] = replace(
                rhs_type,
                shape=(80, rhs_type.shape[1]),
                layout=rhs_layout,
            )
        self.logical[id(acc)] = acc_type
        return emit(self, call, op, operands, mesh, output_window, written, cursor)

    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)
    if case == "group":
        frames = lowering_module.issue_frames

        def misplaced(source, required, repeat, tile):
            return frames(source, required, (3, *repeat[1:]), tile)

        monkeypatch.setattr(lowering_module, "issue_frames", misplaced)
    monkeypatch.setattr(lowering_module.Lowering, "_emit_instruction", malformed)

    _assert_cli_lowering_error(source, message, tmp_path, capsys)
