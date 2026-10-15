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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import combinations
from math import prod
from pathlib import Path

import pytest
import torch

import tilefoundry.passes.transforms.convert_hir_to_tir as lowering_module
from tests._source import import_dsl
from tilefoundry.analysis.access import SSAValue
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import Liveness, result_copies
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)
from tilefoundry.cli import main as cli_main
from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.inspection import PatternPrinter, as_script
from tilefoundry.ir.core import (
    Call,
    Constant,
    Op,
    OpCapability,
    Tuple,
    Var,
    detach_metadata,
    get_metadata,
    op_identifier,
    supported_op_capabilities,
)
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef, collect_param_defs
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, operand_relations
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.pattern import (
    PatternMatcher,
    TensorPattern,
    declared_execution_mesh,
    is_ranked_tensor,
    thread_execution_mesh,
)
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.cuda.nn.wgmma import Form, Wgmma
from tilefoundry.ir.tir.memory import Copy, TensorView
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, MeshScope, Sequential, While
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    Mesh,
    ShardLayout,
    StorageKind,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.layout_algebra import cosize
from tilefoundry.ir.types.mesh import levels, make_mesh, starts
from tilefoundry.ir.visitor import StmtVisitor, collect_exprs
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError
from tilefoundry.schedule import finalize
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    identity_relations,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext, VerifyContext
from tilefoundry.visitor_registry.instruction import check_call
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function

PLAIN = (
    "reshard_locations",
    "chunk_rmsnorm",
    "fp8_block_scaled_gemm",
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
    "sparse_decode",
)
PLAIN_DIMS = {"chunk_rmsnorm": {"chunks": 16}}
PLAIN_REFUSED = {
    "gemm_relu_gemm_smem_staged": (
        r"no nvidia\.h200_sxm MMA reads lhs f32 smem and rhs f32 smem into f32"
        r"(.|\n)*gemm_relu_gemm_smem_staged\.py:39:29"
    ),
}
FINALIZE_REFUSED = {
    "staged_dynamic_inner": "staged selection needs a constant trip count for 'j'",
    "duplicate_output_storage": "return leaf (2,) shares storage with another output",
}
CANDIDATES_REFUSED = {
    "reshard_views": "source has no unscheduled candidate site",
    "scalar_ceildiv_gate": "source has no unscheduled candidate site",
    "register_index_window": "source has no unscheduled candidate site",
    "register_index_window_offset": "source has no unscheduled candidate site",
    "register_arange": "source has no unscheduled candidate site",
    "wgmma_b_mn_major_wide_lbo": "source has no unscheduled candidate site",
    "cta_submesh_thread_scope": "source has no unscheduled candidate site",
    "loop_variable_rebound": "source has no unscheduled candidate site",
    "submesh_slice_in_sibling": "source has no unscheduled candidate site",
    "module_instances": "source has no unscheduled candidate site",
    "child_weight": "source has no unscheduled candidate site",
    "loop_carry_window": "source has no unscheduled candidate site",
    "staged_dynamic_inner": "source has no unscheduled candidate site",
    "mesh_capture_outside": "source has no unscheduled candidate site",
    "mesh_zeros_submesh": "source has no unscheduled candidate site",
    "loop_outputs": "source has no unscheduled candidate site",
    "duplicate_output_storage": "source has no unscheduled candidate site",
    "staged_nested": "source has no unscheduled candidate site",
    "captured_insert_update": "source has no unscheduled candidate site",
    "loop_carry_from_param": "source has no unscheduled candidate site",
}
TIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "tir").glob("*.py")))
HIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "hir").glob("*.py")))
FACT_INSTRUCTIONS = sorted(
    {
        op_identifier(declaration)
        for op_type, capability in supported_op_capabilities(CudaTarget("nvidia.h200_sxm"))
        for declaration in (
            (op_type, capability.declaration) if capability.attribute is not None else (op_type,)
        )
    }
)
WGMMA_FACTS = Path(__file__).parents[1] / "fixtures" / "schedule" / "Wgmma.facts.txt"
_H200_FACTS_TEXT = """\
target nvidia.h200_sxm
instructions
  T.binary             all targets
  T.cast               all targets
  T.clamp              all targets
  T.unary              all targets
  T.where              all targets
  T.copy_async         cp.async
  T.copy_async_tensor  cp.async.bulk.tensor
  T.copy               all targets
  T.ldmatrix           ldmatrix
  T.relu               all targets
  T.tiled_mma          wgmma.mma_async, mma.sync
  T.reduce             all targets
"""
_H200_FACTS_JSON = {
    "target": "nvidia.h200_sxm",
    "instructions": [
        {"id": "T.binary", "capability": None},
        {"id": "T.cast", "capability": None},
        {"id": "T.clamp", "capability": None},
        {"id": "T.unary", "capability": None},
        {"id": "T.where", "capability": None},
        {"id": "T.copy_async", "capability": "cp.async"},
        {"id": "T.copy_async_tensor", "capability": "cp.async.bulk.tensor"},
        {"id": "T.copy", "capability": None},
        {"id": "T.ldmatrix", "capability": "ldmatrix"},
        {"id": "T.relu", "capability": None},
        {"id": "T.tiled_mma", "capability": ["wgmma.mma_async", "mma.sync"]},
        {"id": "T.reduce", "capability": None},
    ],
}
_TARGET_NEUTRAL_FACTS = """\
target {target}
instructions
  T.binary  all targets
  T.cast    all targets
  T.clamp   all targets
  T.unary   all targets
  T.where   all targets
  T.copy    all targets
  T.relu    all targets
  T.reduce  all targets
"""
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
    "wgmma_b_mn_major_wide_lbo": 10240,
    "register_index_window": 0,
    "register_index_window_offset": 0,
    "register_full_like": 0,
    "register_arange": 0,
    "reshard_views": 0,
    "scalar_ceildiv_gate": 0,
    "cta_submesh_thread_scope": 0,
    "loop_variable_rebound": 0,
    "submesh_slice_in_sibling": 32,
    "module_instances": 0,
    "duplicate_output_storage": 8,
    "loop_carry_rebind": 0,
    "loop_carry_window": 0,
    "staged_dynamic_inner": 32,
    "mesh_capture_outside": 0,
    "mesh_views": 0,
    "mesh_zeros_submesh": 128,
    "loop_outputs": 8,
    "staged_nested": 32,
    "child_weight": 0,
    "captured_insert_update": 24,
    "loop_carry_from_param": 0,
    "sparse_decode": 16_384,
    "scalar_binary": 0,
    "fp8_block_scaled_gemm": 65_536,
    "gemm_8192x17408x5120_register_store": 196_608,
    "gemm_8192x17408x5120_tma_store": 212_992,
    "sm80_mma_ldmatrix": 1_536,
    "wgmma_a_k_major": 6_144,
    "wgmma_a_mn_major": 6_144,
    "wgmma_b_nk": 2_560,
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
    "wgmma_b_mn_major_wide_lbo": {
        "thread@0:128#0": _RmemExpectation(32768, "64x128 f32 accumulator and MMA result alias"),
    },
    "register_index_window": {
        "thread@0:1#0": _RmemExpectation(32, "8 f32 output elements; index/view lifetimes reuse their storage"),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the register output"),
    },
    "register_index_window_offset": {
        "thread@0:1#0": _RmemExpectation(32, "8 f32 output elements; index/view lifetimes reuse their storage"),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the register output"),
    },
    "register_full_like": {
        "thread@0:1#0": _RmemExpectation(33, "8 f32 register values and a predicate"),
        "cta@0:1#0": _RmemExpectation(33, "parent envelope of Where and its predicate"),
    },
    "register_arange": {
        "thread@0:1#0": _RmemExpectation(32, "8 i32 indices and the output share storage"),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the indices"),
    },
    "reshard_views": {
        "thread@0:1#0": _RmemExpectation(32, "8 f32 register output elements"),
        "cta@0:2#0": _RmemExpectation(32, "parent envelope of the output"),
    },
    "scalar_ceildiv_gate": {
        "thread@0:1#0": _RmemExpectation(32, "8 f32 register output elements"),
        "cta@0:4#0": _RmemExpectation(32, "parent envelope of the output"),
    },
    "cta_submesh_thread_scope": {
        "thread@0:128#0": _RmemExpectation(512, "4x32 f32 register tile"),
    },
    "loop_variable_rebound": {
        "thread@0:1#0": _RmemExpectation(32, "2x4 f32 output and loop accumulator share storage"),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the loop accumulator"),
    },
    "submesh_slice_in_sibling": {},
    "duplicate_output_storage": {},
    "loop_carry_rebind": {
        "thread@0:32#0": _RmemExpectation(
            96, "independent live Buffers: 1 * 32B + 1 * 64B"
        ),
        "cta@0:1#0": _RmemExpectation(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
    },
    "loop_carry_window": {
        "thread@0:32#0": _RmemExpectation(
            96, "independent live Buffers: 1 * 32B + 1 * 64B"
        ),
        "cta@0:1#0": _RmemExpectation(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
    },
    "staged_dynamic_inner": {
        "thread@0:32#0": _RmemExpectation(
            16, "the 1x8 bf16 accumulator and insert aliases form one 16-byte group"
        ),
        "cta@0:1#0": _RmemExpectation(16, "parent envelope of the 1x8 bf16 accumulator"),
    },
    "mesh_capture_outside": {},
    "mesh_views": {
        "thread@0:32#0": _RmemExpectation(
            256, "independent live Buffers: 2 * 128B"
        ),
        "thread@0:32#1": _RmemExpectation(
            256, "independent live Buffers: 2 * 128B"
        ),
        "thread@0:32#2": _RmemExpectation(
            256, "independent live Buffers: 2 * 128B"
        ),
    },
    "mesh_zeros_submesh": {},
    "loop_outputs": {},
    "staged_nested": {
        "thread@0:32#0": _RmemExpectation(
            96, "the 6x8 bf16 accumulator and its insert aliases form one 96-byte group"
        ),
        "cta@0:1#0": _RmemExpectation(
            96, "the 6x8 bf16 accumulator and its insert aliases form one 96-byte group"
        ),
    },
    "child_weight": {
        "thread@0:32#0": _RmemExpectation(
            384, "independent live Buffers: 3 * 128B"
        ),
        "cta@0:1#0": _RmemExpectation(
            384, "independent live Buffers: 3 * 128B"
        ),
    },
    "wgmma_b_nk": {
        "thread@0:128#0": _RmemExpectation(
            4096, "64x16 f32 C fragment, shared by zeros and MMA result"
        ),
    },
    "module_instances": {
        "cta@0:1#0": _RmemExpectation(
            384, "three independent 128-byte callee operand/result Buffers in the CTA window"
        ),
    },
    "captured_insert_update": {},
    "loop_carry_from_param": {},
    "sparse_decode": {
        "cta@0:1#0": _RmemExpectation(
            520,
            "independent live Buffers: 1 * 8B + 1 * 512B",
        ),
        "thread@128:128#0": _RmemExpectation(33_024, "attention intermediates and row reductions"),
        "thread@0:256#0": _RmemExpectation(33_024, "parent envelope of attention compute"),
    },
    "scalar_binary": {
        "cta@0:1#0": _RmemExpectation(
            260,
            "independent live Buffers: 1 * 4B + 2 * 128B",
        ),
        "thread@0:32#0": _RmemExpectation(
            260,
            "independent live Buffers: 1 * 4B + 2 * 128B",
        ),
    },
    "fp8_block_scaled_gemm": {
        "cta@0:1#0": _RmemExpectation(
            4,
            "the 1x1 f32 tile-scale copy is unsplit; thread-split accumulators "
            "do not contribute to the CTA record",
        ),
        "thread@128:256#0": _RmemExpectation(65_536, "128x128 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(
            131584,
            "independent live Buffers: 1 * 512B + 2 * 65536B",
        ),
        "thread@128:256#2": _RmemExpectation(65_536, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131584, "independent live Buffers: 1 * 512B + 2 * 65536B"),
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
            512, "analysis coalesces the f32 loop result and bf16 cast: max(512, 256)"
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
            16384, "independent live Buffers: 2 * 8192B"
        ),
        "thread@0:32#1": _RmemExpectation(
            2_048,
            "f32 loader tile (2048); bf16 cast follows its last read",
        ),
        "thread@0:256#0": _RmemExpectation(
            16384,
            "independent live Buffers: 2 * 8192B",
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


def _tir_in(path: Path) -> Module | PrimFunction:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if isinstance(value, (Module, PrimFunction)))


def _prim_in(path: Path) -> PrimFunction:
    program = _tir_in(path)
    return program.entry_function() if isinstance(program, Module) else program


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

    def visit_MeshScope(self, stmt: MeshScope) -> None:
        previous = getattr(self, "meshes", ())
        self.meshes = (*previous, stmt.mesh)
        try:
            self.visit(stmt.body)
        finally:
            self.meshes = previous

    def visit_Evaluate(self, stmt: Evaluate) -> None:
        op = stmt.callable
        if getattr(type(op), "_op_schema", None) is None:
            return
        meshes = getattr(self, "meshes", ())
        mesh = make_mesh(*meshes) if meshes else None
        bindings, projection = check_call(
            op,
            stmt.args,
            mesh,
            frame=VerifyContext(current_mesh=mesh, mesh_scope=meshes),
        )
        for param, pattern in zip(projection.params, projection.patterns, strict=True):
            if pattern is not None:
                self.matches.append((op, param.name, dict(bindings)))


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
    result = analyze(*check_program(program, entry, dims=dims), analysis=("memory", "performance"))
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
            if isinstance(expr, Call) and isinstance(expr.target, (Reshape, Slice, Transpose))
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
    result = analyze(*check_program(program, entry), analysis=analysis)
    assert metadata_type in result.metadata_types
    if path.stem == "module_instances":
        record_type = PerformanceMetadata if analysis == "performance" else metadata_type
        for function in result.functions[1:]:
            schedules = [expr for expr in collect_exprs(function.body)
                         if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)]
            assert any(get_metadata(expr, record_type) is not None
                       for expr in schedules)

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

    if analysis == "memory" and path.stem == "wgmma_tma_3stage":
        assert RegionMemoryMetadata in result.metadata_types
        lifetimes = get_metadata(result.function, RegionMemoryMetadata).lifetimes
        smem = sorted(item.bytes for item in lifetimes if item.memory_level == "smem")
        rmem = sorted(item.bytes for item in lifetimes if item.memory_level == "rmem")
        assert smem == sorted((512 * 3, 4096 * 3))
        assert rmem == [4096, 8192, 8192]

    if analysis == "memory":
        placement = get_metadata(result.function, RegionMemoryMetadata)
        assert placement is not None
        liveness = get_metadata(result.function, Liveness)
        regions = tuple(window.region for window in liveness.regions)
        region_records = tuple(
            record
            for region in regions
            if (record := get_metadata(region, RegionMemoryMetadata)) is not None
        )
        if path.stem == "module_instances":
            region_records += tuple(
                record
                for function in result.functions[1:]
                for region in collect_exprs(function.body)
                if isinstance(region, MeshRegion)
                and (record := get_metadata(region, RegionMemoryMetadata)) is not None
            )
        if path.stem == "gemm_8192x17408x5120_tma_store":
            assert len(region_records) == 7
        for record in region_records:
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
        region_rmem_peaks = tuple(
            peak.peak_bytes
            for record in region_records
            if (peak := record.peak_for("rmem")) is not None
        )
        occurrences: dict[str, int] = {}
        observed_rmem = {}
        observed_smem = {}
        for window in liveness.regions:
            key = _region_key(window.region, occurrences)
            record = get_metadata(window.region, RegionMemoryMetadata)
            peak = record.peak_for("rmem") if record is not None else None
            if peak is not None:
                observed_rmem[key] = peak.peak_bytes
            smem_peak = record.peak_for("smem") if record is not None else None
            if smem_peak is not None:
                observed_smem[key] = smem_peak.peak_bytes
        if path.stem == "captured_insert_update":
            assert observed_smem["thread@0:32#0"] == 24
        expected_rmem = {
            key: expectation.peak_bytes for key, expectation in RMEM_EXPECTED[path.stem].items()
        }
        assert observed_rmem == expected_rmem, {
            key: expectation.derivation for key, expectation in RMEM_EXPECTED[path.stem].items()
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


@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_schedule_memory_report_carries_allocations(
    path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{path.stem}.json"

    assert cli_main(["analyze", str(path), str(out), "--memory", "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    report = json.loads(out.read_text())
    memory = report["functions"][0]['function_records']["memory"]
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
    for row in report["functions"][0]['calls']:
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
    result = analyze(*check_program(program, entry), analysis="performance")
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
            op=CopyAsync(),
            repeat=repeat,
            order=order,
        ),
        args=(
            Var(name="src", type=type_),
            Call(
                Empty(type=replace(type_, storage=StorageKind.SMEM)),
                (),
                type=replace(type_, storage=StorageKind.SMEM),
            ),
        ),
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
        ScheduleOp(op=CopyAsync()),
        ((source_type, source), (result_type, torch.zeros_like(source))),
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
    source = (
        Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "fp8_block_scaled_gemm.py"
    )
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


@dataclass(frozen=True)
class FixtureCase:
    name: str
    argv: tuple[str, ...]
    check: Callable[[Path, pytest.CaptureResult[str]], None]
    code: int = 0
    files: Mapping[str, Callable[[], torch.Tensor]] = field(default_factory=dict)


def _quiet_json(out: Path, captured: pytest.CaptureResult[str]) -> dict:
    assert captured == ("", "")
    return json.loads(out.read_text())


def _check_selected_rows(out: Path, captured: pytest.CaptureResult[str]) -> None:
    records = _quiet_json(out, captured)
    gather = next(
        row["memory"]
        for row in [row for function in records["functions"] for row in function["calls"]]
        if any(
            operand["arg"] == 1 and operand["name"] == "idx"
            for operand in row["memory"]["operands"]
        )
    )
    assert gather["operands"][0]["read"] == 8192, "gather reads 64 selected bf16 rows"
    footprint = records["functions"][0]["function_records"]["memory"]["footprint"]
    assert footprint["buffers"]["kv"]["gmem"]["total"] == 8192
    assert footprint["buffers"]["kv"]["gmem"]["total"] <= gather["operands"][0]["read"]


def _check_service_work(out: Path, captured: pytest.CaptureResult[str]) -> None:
    records = _quiet_json(out, captured)
    cost = records["functions"][0]["function_records"]["compute-cost"]
    assert not {"i64", "bool"}.intersection(cost["flops"]), "services are not flops"
    assert cost["other_ops"]["integer"]["total"] > 0, "i64 arithmetic counts as integer work"
    assert cost["other_ops"]["predicate"]["total"] > 0, "predicates count separately"


def _check_gmem_gather_refusal(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    indexed = next(row for row in report["lines"] if row["op"] == "tf.index_select")
    assert not indexed["candidates"] and [item["id"] for item in indexed["refused"]] == [
        "T.copy_async"
    ], "gather keeps its gmem result"
    assert indexed["refused"][0]["reason"].startswith("mesh:")
    assert 'Topology("cta"' in indexed["refused"][0]["reason"]


def _check_smem_placement(out: Path, captured: pytest.CaptureResult[str]) -> None:
    """The second 64x64 bf16 tile ends at 8192 + 8192, the analyzed peak."""
    assert captured == ("", "")
    lines = out.read_text().splitlines()
    peaks = next(
        item.removeprefix("peak=") for item in lines[2].split() if item.startswith("peak=")
    )
    assert dict(item.split(":") for item in peaks.split(";"))["smem"] == "16.00KB", (
        "two shared tiles need 16384 bytes"
    )
    key_view = next(i for i, line in enumerate(lines) if line.strip() == "ks = T.tensor_view(")
    assert lines[key_view + 1].strip().split(",", 1)[0] == "0", (
        "the lowered key tile starts at 0 bytes"
    )


def _attention_tensor(name: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9)
    q = torch.randn(1, 2, 128, 64, generator=generator).to(torch.bfloat16)
    kv = torch.randn(1, 256, 1, 64, generator=generator).to(torch.bfloat16)
    idx = torch.arange(64, dtype=torch.int64) * 3
    idx[1], idx[17] = -1, 256
    live = (idx >= 0) & (idx < 256)
    rows = torch.zeros(64, 64, dtype=torch.bfloat16)
    rows[live] = kv.reshape(256, 64)[idx[live]]
    scores = q[0, 1, 32:96].float() @ rows.float().t()
    scores[:, ~live] = -torch.inf
    probabilities = torch.softmax(scores, dim=1).to(torch.bfloat16)
    reference = (probabilities.float() @ rows.float()).to(torch.bfloat16)
    return {"q": q, "kv": kv, "idx": idx, "expected": reference}[name]


def _check_attention(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    comparison = report["runs"][0]["outputs"][0]["fns"][0]
    assert comparison["fn"] == "allclose"
    assert comparison["rtol"] == 2**-7 and comparison["atol"] == 0
    assert report["passed"] and comparison["passed"], (
        "gather fill, mask/key alignment and natural output heads match torch"
    )


def _check_golden(out: Path, captured: pytest.CaptureResult[str], *, golden: Path) -> None:
    assert captured == ("", "")
    if golden.suffix == ".json":
        assert json.loads(out.read_text()) == json.loads(golden.read_text())
    else:
        assert out.read_bytes() == golden.read_bytes()


def _check_facts(
    out: Path,
    captured: pytest.CaptureResult[str],
    *,
    expected: str | dict,
) -> None:
    if isinstance(expected, dict):
        assert _quiet_json(out, captured) == expected
    else:
        assert captured == ("", "")
        assert out.read_text() == expected


def _check_facts_declaration(
    out: Path,
    captured: pytest.CaptureResult[str],
    *,
    instruction: str,
) -> None:
    report = _quiet_json(out, captured)
    assert report["id"] == instruction
    assert report["target"] == "nvidia.h200_sxm"
    assert report["execution_mesh"]


def _check_facts_refusal(out: Path, captured: pytest.CaptureResult[str], *, message: str) -> None:
    assert captured.out == ""
    assert captured.err.startswith("tilefoundry: error: ")
    assert message in captured.err
    assert not out.exists()


def _check_finalized_rows(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured == ("", "")
    lowered = import_dsl(out.read_text())
    assert isinstance(lowered, PrimFunction)
    assert lowered.params[-1].type.shape == (128,)


def _check_finalize_unbound_dim(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured.out == ""
    assert "schedule finalize" in captured.err
    assert "bind it with --dim" in captured.err
    assert not out.exists()


def _check_reshard_locations(out: Path, captured: pytest.CaptureResult[str]) -> None:
    rows = _quiet_json(out, captured)["functions"][0]["calls"]
    rows = {row["value"].rsplit(":", 1)[1]: row["memory"] for row in rows}
    for label, byte_count in (("12", 256), ("14", 32)):
        memory = rows[label]
        assert memory["operands"][0]["read"] == byte_count
        assert memory["operands"][1]["write"] == byte_count
    for label in ("15", "16", "17"):
        memory = rows[label]
        assert memory["buffer_bytes"] is None
        assert memory["offsets"] == []
        assert memory["traffic"] == {"communication": {}, "storage": {}}


def _check_reshape_matches_torch(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    assert report["passed"]
    comparison = report["runs"][0]["outputs"][0]["fns"][0]
    assert comparison["fn"] == "allclose" and comparison["passed"]


def _check_crossing_axes_refusal(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured.out == ""
    assert "layout positions (2, 8, 2, 4, 2, 4, 8) cross logical axes of (64, 128)" in captured.err
    assert "layout_crossing_axes.py:13:" in captured.err
    assert not out.exists()


def _check_selector_help(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured.err == ""
    assert "model.py[:Module[.function]]" in captured.out
    assert "child_module" not in captured.out


def _reshape_tensor(expected: bool = False) -> torch.Tensor:
    x = torch.arange(128, dtype=torch.float32).reshape(1, 8, 1, 16)
    return x.reshape(8, 16) if expected else x


FIXTURES = (
    FixtureCase(
        "reshard is charged only when an element moves",
        ("analyze", "tests/fixtures/schedule/plain/reshard_locations.py", "{out}", "--memory", "--json"),
        _check_reshard_locations,
    ),
    FixtureCase(
        "check binds two DimVars in a reshape",
        ("check", "tests/fixtures/placed/two_dimvar_reshape.py:TwoDimVarReshape", "--inputs", "files:{tmp}/x.pt", "--expected", "{tmp}/expected.pt", "--dim", "T=8", "--dim", "C=16", "--device", "cpu", "--out", "output", "--fn", "allclose", "--atol", "0", "--rtol", "0", "--json", "{out}"),
        _check_reshape_matches_torch,
        files={"x.pt": _reshape_tensor, "expected.pt": partial(_reshape_tensor, True)},
    ),
    FixtureCase(
        "finalize refuses a layout crossing logical axes where it is declared",
        ("schedule", "finalize", "tests/fixtures/schedule/hir/refused/layout_crossing_axes.py", "{out}"),
        _check_crossing_axes_refusal,
        code=1,
    ),
    FixtureCase(
        "analyze help names only Module.function", ("analyze", "--help"), _check_selector_help,
    ),
    FixtureCase(
        "finalize binds an open row dimension",
        (
            "schedule",
            "finalize",
            "tests/fixtures/schedule/hir/dim/row_copy.py",
            "{out}",
            "--dim=rows=128",
        ),
        _check_finalized_rows,
    ),
    FixtureCase(
        "finalize refuses an unbound row dimension without output",
        ("schedule", "finalize", "tests/fixtures/schedule/hir/dim/row_copy.py", "{out}"),
        _check_finalize_unbound_dim,
        code=1,
    ),
    FixtureCase(
        "gather reads selected rows and caps the footprint",
        (
            "analyze",
            "tests/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--memory",
            "--json",
        ),
        _check_selected_rows,
    ),
    FixtureCase(
        "integer and predicate work uses services",
        (
            "analyze",
            "tests/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--compute-cost",
            "--json",
        ),
        _check_service_work,
    ),
    FixtureCase(
        "gather into smem is refused for a gmem result",
        (
            "schedule",
            "candidates",
            "tests/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--json",
        ),
        _check_gmem_gather_refusal,
    ),
    FixtureCase(
        "analyzed smem peak matches the lowered placement",
        ("schedule", "finalize", "tests/fixtures/schedule/hir/sparse_decode.py", "{out}"),
        _check_smem_placement,
    ),
    *(
        FixtureCase(
            name,
            (
                "check",
                f"tests/fixtures/schedule/{stage}/sparse_decode.py:SparseDecode",
                "--inputs",
                "files:{tmp}/q.pt,{tmp}/kv.pt,{tmp}/idx.pt",
                "--expected",
                "{tmp}/expected.pt",
                "--device",
                "cpu",
                "--out",
                "output",
                "--fn",
                "allclose",
                "--rtol",
                str(2**-7),
                "--atol",
                "0",
                "--json",
                "{out}",
            ),
            _check_attention,
            files={
                f"{key}.pt": partial(_attention_tensor, key)
                for key in ("q", "kv", "idx", "expected")
            },
        )
        for name, stage in (
            ("plain sparse attention matches torch", "plain"),
            ("scheduled sparse attention matches torch", "hir"),
        )
    ),
    FixtureCase(
        "facts write the Wgmma declaration",
        ("schedule", "facts", "T.cuda.sm90.Wgmma", "--target", "nvidia.h200_sxm", "{out}"),
        partial(_check_golden, golden=WGMMA_FACTS),
    ),
    FixtureCase(
        "facts write the tiled MMA atom alternatives",
        ("schedule", "facts", "T.tiled_mma", "--target", "nvidia.h200_sxm", "{out}"),
        partial(_check_golden, golden=WGMMA_FACTS.with_name("tiled_mma.facts.txt")),
    ),
    *(
        FixtureCase(
            f"facts describe {instruction}",
            (
                "schedule",
                "facts",
                instruction,
                "--target",
                "nvidia.h200_sxm",
                "{out}",
                "--json",
            ),
            partial(_check_facts_declaration, instruction=instruction),
        )
        for instruction in FACT_INSTRUCTIONS
        if instruction not in {"T.cuda.sm90.Wgmma", "T.tiled_mma"}
    ),
    *(
        FixtureCase(
            name,
            ("schedule", "facts", "--target", target, "{out}", *options),
            partial(_check_facts, expected=expected),
        )
        for name, target, options, expected in (
            ("facts list CUDA instructions as text", "nvidia.h200_sxm", (), _H200_FACTS_TEXT),
            (
                "facts list CUDA instructions as JSON",
                "nvidia.h200_sxm",
                ("--json",),
                _H200_FACTS_JSON,
            ),
            (
                "CPU facts list only target neutral instructions",
                "cpu",
                (),
                _TARGET_NEUTRAL_FACTS.format(target="cpu"),
            ),
            (
                "Apple facts list only target neutral instructions",
                "apple.m2_pro",
                (),
                _TARGET_NEUTRAL_FACTS.format(target="apple.m2_pro"),
            ),
        )
    ),
    *(
        FixtureCase(
            name,
            ("schedule", "facts", instruction, "--target", target, "{out}"),
            partial(_check_facts_refusal, message=message),
            code=1,
        )
        for name, instruction, target, message in (
            (
                "facts refuse an unknown instruction without output",
                "T.cuda.sm90.Unknown",
                "nvidia.h200_sxm",
                "unknown instruction 'T.cuda.sm90.Unknown'",
            ),
            (
                "facts refuse an unknown target without output",
                "T.cuda.sm90.Wgmma",
                "nvidia.unknown",
                "unknown target identity 'nvidia.unknown'",
            ),
        )
    ),
    FixtureCase(
        "candidates write the canonical report",
        (
            "schedule",
            "candidates",
            f"tests/fixtures/schedule/plain/{CANDIDATE_GOLDEN.name.removesuffix('.candidates.txt')}.py",
            "{out}",
        ),
        partial(_check_golden, golden=CANDIDATE_GOLDEN),
    ),
    FixtureCase(
        "analyze writes the memory report",
        (
            "analyze",
            "tests/fixtures/schedule/hir/gemm_8192x17408x5120_tma_store.py",
            "{out}",
            "--memory",
        ),
        partial(_check_golden, golden=ANALYZED_GOLDEN),
    ),
)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_fixture(case: FixtureCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name, make in case.files.items():
        torch.save(make(), tmp_path / name)
    out = tmp_path / "out"
    argv = [arg.format(out=out, tmp=tmp_path) for arg in case.argv]
    try:
        code = cli_main(argv)
    except SystemExit as error:
        code = error.code
    assert code == case.code
    case.check(out, capsys.readouterr())


def test_single_issue_schedule_preserves_instruction_relations() -> None:
    schedule = _copy_schedule_call(repeat=(1,), order=(0,))
    source = schedule.args[0]
    destination_type = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.SMEM)
    inferred = TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,))))
    inference_type(schedule, inferred)
    scheduled = relations_of(schedule, inferred)
    single = operand_relations(schedule.target.op, (source.type, destination_type))
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
            "src:",
        ),
    ),
    ids=("repeat", "order", "operand-pattern"),
)
def test_schedule_typeinfer_rejects_invalid_contract(call: Call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


@register_op(dialect="T", category="schedule", name="unstated_instruction")
class _UnstatedInstruction(Op):
    execution_mesh = thread_execution_mesh()

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
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


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

    context = TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,))))
    assert inference_type(call, replace(context, memo={})) == type_
    param = _UnstatedInstruction._op_schema.signature[0]
    monkeypatch.setattr(param, "pattern", TensorPattern(shape=(3,)))
    with pytest.raises(
        ValueError, match="iteration extent 4 is not divisible by single-issue extent 3"
    ):
        inference_type(call, replace(context, memo={}))
    monkeypatch.setattr(param, "pattern", None)
    with pytest.raises(
        ValueError, match="has no input pattern, between relation, or registered verifier"
    ):
        inference_type(call, replace(context, memo={}))


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
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_instruction_operands(monkeypatch) -> None:
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
        match="expects operands.*value.*got 0",
    ):
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


def test_schedule_evaluation_rejects_an_unregistered_instruction() -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    value = torch.arange(4, dtype=torch.bfloat16)

    with pytest.raises(EvalError, match="no schedule evaluator registered"):
        _evaluate_call(ScheduleOp(op=_UnstatedInstruction()), ((type_, value),), type_)


@pytest.mark.parametrize("path", TIR, ids=lambda path: path.stem)
def test_tir_program_is_verified_and_canonical(path: Path) -> None:
    program = _tir_in(path)
    if isinstance(program, Module):
        verify_module(program)
    else:
        verify_prim_function(program)
    source = path.read_text()
    expected = source[source.index("from __future__ import annotations") :]
    if isinstance(program, Module):
        expected = "\n".join(
            line for line in expected.splitlines()
            if not line.lstrip().startswith(("# analysis ", "# selection ", "# memory "))
        ) + "\n"
    assert as_script(program) == expected


def test_parameter_structure_does_not_repeat_its_name() -> None:
    declarations = {schema.op_class for schema in iter_schemas() if schema.op_class is not None}
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


def test_instruction_requires_an_execution_mesh_declaration(monkeypatch) -> None:
    monkeypatch.delattr(_UnstatedInstruction, "execution_mesh")
    with pytest.raises(ValueError, match="_UnstatedInstruction execution_mesh must be"):
        declared_execution_mesh(_UnstatedInstruction)


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
    assert all(
        row["candidates"] or all(item["reason"].startswith("mesh:") for item in row["refused"])
        for _name, row in sites
        if row["op"] == "tf.reshard"
    )
    matmuls = [(name, row) for name, row in sites if row["op"] == "tf.matmul"]
    assert len(matmuls) == 8
    assert not any(row["candidates"] for _name, row in matmuls)


@pytest.mark.parametrize(
    ("name", "dims", "matmuls", "reshards", "accepted_matmuls", "b_layout"),
    (
        ("chunk_rmsnorm", ("chunks=16",), 0, 7, 0, "KN"),
        ("chunk_rmsnorm", ("chunks=32",), 0, 7, 0, "KN"),
        ("fp8_block_scaled_gemm", (), 1, 5, 0, "KN"),
        ("gemm_8192x17408x5120_cta_grid", (), 1, 3, 0, "KN"),
        ("gemm_relu_gemm_tiled", (), 2, 2, 0, "KN"),
        ("gemm_relu_gemm_untiled", (), 2, 0, 0, "KN"),
        ("gemm_relu_gemm_untiled", (), 2, 0, 0, "NK"),
    ),
)
def test_schedule_candidates_reports_every_plain_site(
    name: str,
    dims: tuple[str, ...],
    matmuls: int,
    reshards: int,
    accepted_matmuls: int,
    b_layout: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = f"tests/fixtures/schedule/plain/{name}.py"
    if b_layout == "NK":
        variant = tmp_path / f"{name}_nk.py"
        variant.write_text(
            Path(source)
            .read_text()
            .replace("tf.matmul(a, b)", 'tf.matmul(a, b, b_layout="NK")')
            .replace("tf.matmul(y, c)", 'tf.matmul(y, c, b_layout="NK")')
        )
        source = str(variant)
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
    assert all(
        row["candidates"] or all(item["reason"].startswith("mesh:") for item in row["refused"])
        for row in reshard_rows
    )

    for row in matmul_rows:
        assert not row["candidates"]
        assert {item["id"] for item in row["refused"]} == {"T.cuda.sm90.Wgmma", "T.cuda.sm80.Mma"}
        assert all(item["reason"] for item in row["refused"])

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


@pytest.mark.parametrize(
    "source",
    HIR,
    ids=lambda path: path.stem,
)
def test_schedule_candidates_omit_selected_schedule_calls(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{source.stem}.json"

    if source.stem in CANDIDATES_REFUSED:
        assert cli_main(["schedule", "candidates", str(source), str(out), "--json"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert CANDIDATES_REFUSED[source.stem] in captured.err
        assert not out.exists()
        return

    assert cli_main(["schedule", "candidates", str(source), str(out), "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    report = json.loads(out.read_text())
    assert report["lines"]
    ops = [row["op"] for row in report["lines"]]
    assert "tf.matmul" not in ops
    assert "tf.reshard" not in ops
    assert any(
        op
        in {"tf.binary", "tf.cast", "tf.clamp", "tf.relu", "tf.reduce", "tf.unary", "tf.transpose"}
        for op in ops
    )
    if source.stem == "wgmma_cast_between_schedules":
        assert ops.count("tf.reduce") == 1


def test_operand_match_refusal_names_the_failed_pattern() -> None:
    function = _prim_in(
        Path(__file__).parents[1] / "fixtures" / "schedule" / "tir" / "wgmma_a_k_major.py"
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
    assert "gmem" in rejected
    assert "smem" in rejected


def _assert_live_smem_views_do_not_overlap(function: PrimFunction, source: Path) -> None:
    """Check literal views in lowered TIR, including uses across loop backedges."""
    views: list[tuple[Var, int, int]] = []
    defined: dict[int, int] = {}
    last: dict[int, int] = {}
    clock = 0

    def use(expressions) -> None:
        for expression in expressions:
            for value in collect_exprs(expression):
                if isinstance(value, Var) and id(value) in defined:
                    last[id(value)] = clock

    def walk(statement) -> None:
        nonlocal clock
        if statement is None:
            return
        clock += 1
        if isinstance(statement, Sequential):
            for child in statement.body:
                walk(child)
        elif isinstance(statement, LetStmt):
            use((statement.value,))
            defined[id(statement.var)] = last[id(statement.var)] = clock
            fields = (
                statement.value.elements
                if isinstance(statement.value, Tuple)
                else (statement.value,)
            )
            for field in fields:
                if (
                    isinstance(field, Call)
                    and isinstance(field.target, TensorView)
                    and field.target.storage is StorageKind.SMEM
                    and isinstance(field.args[0], Constant)
                ):
                    span = (
                        cosize(field.type.layout)
                        if field.type.layout is not None
                        else prod(field.type.shape)
                    )
                    views.append(
                        (
                            statement.var,
                            int(field.args[0].value),
                            span * field.type.dtype.bit_width // 8,
                        )
                    )
            walk(statement.body)
        elif isinstance(statement, (For, While)):
            use(
                (statement.start, statement.stop, statement.step)
                if isinstance(statement, For)
                else (statement.cond,)
            )
            entered = clock
            walk(statement.body)
            for key, used in last.items():
                if defined[key] < entered < used:
                    last[key] = clock
        elif isinstance(statement, If):
            use((statement.cond,))
            walk(statement.then_body)
            walk(statement.else_body)
        elif isinstance(statement, MeshScope):
            walk(statement.body)
        elif isinstance(statement, Evaluate):
            use(statement.args)

    walk(function.body)
    for (left, start, size), (right, other_start, other_size) in combinations(views, 2):
        if left is right:
            continue
        live = max(defined[id(left)], defined[id(right)]) <= min(last[id(left)], last[id(right)])
        overlap = max(start, other_start) < min(start + size, other_start + other_size)
        assert not (live and overlap), (
            f"{source.name}: live smem views overlap: "
            f"{left.name}@{start}+{size}, {right.name}@{other_start}+{other_size}"
        )


@pytest.mark.parametrize("source", HIR, ids=lambda path: path.stem)
def test_schedule_finalize_writes_verified_tir(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if source.stem in FINALIZE_REFUSED:
        _assert_cli_lowering_error(source, FINALIZE_REFUSED[source.stem], tmp_path, capsys)
        return

    out = tmp_path / source.name
    golden_name = "loop_carry_window.py" if source.stem == "loop_carry_rebind" else source.name
    expected = source.parent.parent / "tir" / golden_name

    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert out.read_bytes() == expected.read_bytes()
    lowered = _tir_in(out)
    functions = (
        tuple(frame.function for frame in list_call_paths(lowered).values())
        if isinstance(lowered, Module) else (lowered,)
    )
    found = []
    for function in functions:
        _assert_live_smem_views_do_not_overlap(function, source)
        found.extend(_direct_operand_matches(function))
    if source.stem == "loop_outputs":
        assert lowered.output_count == 3
        assert import_dsl(as_script(lowered)).output_count == 3
    assert found
    wgmma_lhs = [
        captures
        for op, name, captures in found
        if isinstance(getattr(op, "atom", None), Wgmma) and name == "lhs"
    ]
    assert all("a_major" in captures for captures in wgmma_lhs)
    assert all("a_swizzle" in captures for captures in wgmma_lhs if captures["form"].name == "SS")


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
    return source, module, analyze(
        *check_program(module, module.entry_function()),
        analysis=("memory",),
    )


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
    monkeypatch.delitem(access_relation_registry._map, type(transfer.target.op))
    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "states no access relations", tmp_path, capsys)


def test_lowering_rejects_addressable_result_without_offsets(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    addressable = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, Empty)
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
        "dst: gmem does not match rmem",
        tmp_path,
        capsys,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("group", "input 0 is laid out more finely than the scope"),
        ("atom", "iteration extent 17 is not divisible by single-issue extent 16"),
        ("row", "iteration axis 2 repeat 5 is not divisible by row issues 4"),
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
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    check = lowering_module.check_call

    def malformed(op, operands, mesh, *, frame, repeat=None, order=None):
        if getattr(op, "atom", None) is None:
            return check(op, operands, mesh, frame=frame, repeat=repeat, order=order)
        acc_type, lhs_type, rhs_type = operands
        assert isinstance(acc_type.layout, ShardLayout)
        if case == "group":
            held = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
            changed = replace(held, shape=(*held.shape[:-1], (3, *held.shape[-1][1:])))
            mesh = replace(
                mesh,
                layout=replace(mesh.layout, outer=changed)
                if isinstance(mesh.layout, ComposedLayout)
                else changed,
            )
        elif case == "atom":
            acc_type = replace(acc_type, shape=(acc_type.shape[0], 17))
            rhs_type = replace(rhs_type, shape=(rhs_type.shape[0], 17))
        elif case == "row":
            repeat = (2, 1, 5)
            assert isinstance(lhs_type.layout, ComposedLayout)
            assert isinstance(lhs_type.layout.outer, Layout)
            lhs_outer = replace(
                lhs_type.layout.outer,
                shape=(*lhs_type.layout.outer.shape[:-1], (5, 16)),
            )
            lhs_type = replace(
                lhs_type,
                shape=(lhs_type.shape[0], 80),
                layout=replace(lhs_type.layout, outer=lhs_outer),
            )
            assert isinstance(rhs_type.layout, Layout)
            rhs_layout = replace(
                rhs_type.layout,
                shape=((5, *rhs_type.layout.shape[0][1:]), *rhs_type.layout.shape[1:]),
            )
            rhs_type = replace(rhs_type, shape=(80, rhs_type.shape[1]), layout=rhs_layout)
        return check(
            op, (acc_type, lhs_type, rhs_type), mesh, frame=frame, repeat=repeat, order=order
        )

    monkeypatch.setattr(lowering_module, "analyze", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(lowering_module, "check_call", malformed)

    _assert_cli_lowering_error(source, message, tmp_path, capsys)


class _RetainedStatements(StmtVisitor[None]):
    def __init__(self):
        self.calls = []
        self.loops = []
        self.operations = []

    def visit_For(self, stmt, ctx=None):
        self.loops.append(stmt)
        self.visit(stmt.body)

    def visit_Evaluate(self, stmt, ctx=None):
        self.operations.append(stmt.callable)
        if isinstance(stmt.callable, (PrimFunction, ModuleList, Call)):
            self.calls.append(stmt)


def test_lowering_retains_all_four_instance_and_local_call_shapes():
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/module_instances.py"
    lowered = finalize(_module_in(source))
    verify_module(lowered)
    frames = list_call_paths(lowered)
    assert len(frames) == 3
    assert all(isinstance(frame.function, PrimFunction) for frame in frames.values())
    layers, mm = lowered.children["layers"], lowered.children["mm"]
    assert len(layers.elements) == 3
    assert all(element is mm.elements[0] for element in layers.elements)
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    assert len(visitor.loops) == 1
    assert len(visitor.calls) == 5
    assert sum(isinstance(call.callable, ModuleList) and call.callable.key == mm.key
               for call in visitor.calls) == 2
    indexed = [call for call in visitor.calls if isinstance(call.callable, Call)]
    assert len(indexed) == 2
    assert all(call.callable.args[0].key == layers.key for call in indexed)
    assert any(isinstance(call.callable.args[1], Var) for call in indexed)
    assert any(isinstance(call.callable.args[1], Constant) and call.callable.args[1].value == 0
               for call in indexed)
    for call in visitor.calls:
        assert len(call.args) == (3 if isinstance(call.callable, PrimFunction) else 2)
        with pytest.raises(RuntimeError, match="no codegen handler for Op"):
            CppEmitter(context=CudaCodegenContext(target=lowered.target)).visit_Evaluate(call)


def _returned_view(storage):
    type_ = TensorType((32,), DType.f32, Layout((32,), (1,)), storage)
    param = Var("x", type=type_)
    function = Function.build(name="run", params=(param,), body=param, return_type=type_)
    leaf = Module("View", (function,), "run")
    slot = ModuleList(leaf)
    activation = replace(param)
    call = Call(slot, (activation,), type=type_)
    root = Function.build(name="root", params=(activation,), body=call, return_type=type_)
    return Module("ViewCaller", (root,), "root", children={"child": slot},
                  target=CudaTarget("nvidia.h200_sxm"),
                  topologies=(Topology("cta", 1), Topology("thread", 32)))


def test_a_returned_gmem_view_requires_explicit_scheduled_copy():
    with pytest.raises(LoweringError, match=r"return leaf \(\) cannot be placed on its output: x"):
        finalize(_returned_view(StorageKind.GMEM))


def test_a_returned_smem_view_requires_explicit_scheduled_copy():
    with pytest.raises(LoweringError, match=r"return leaf \(\) cannot be placed on its output: x"):
        finalize(_returned_view(StorageKind.SMEM))


@pytest.mark.parametrize("wrong,refused", [
    ("count", "arg count"), ("type", r"arg\[0\] type"), ("recursion", "recursion not allowed"),
])
def test_verifier_rejects_malformed_retained_tir_calls(wrong, refused):
    param = Var("x", type=TensorType.scalar(DType.f32))
    callee = PrimFunction("callee", (param,), Sequential(()), target=CudaTarget("nvidia.h200_sxm"))
    args = () if wrong == "count" else (Constant(0, type=TensorType.scalar(DType.i32)),)
    caller = PrimFunction("caller", (), Sequential((Evaluate(callee, args),)), target=callee.target)
    if wrong == "recursion":
        caller.body = Sequential((Evaluate(caller, ()),))
    with pytest.raises(VerifyError, match=refused):
        verify_prim_function(caller, module_fns=(caller, callee))


def test_verifier_rejects_non_integer_indexed_tir_target():
    lowered = finalize(_module_in(Path(__file__).parents[1] / "fixtures/schedule/hir/module_instances.py"))
    slot = lowered.children["layers"]
    target = Call(TupleGetItem(), (slot, Constant(0.5, type=TensorType.scalar(DType.f32))),
                  type=slot.type.fields[0])
    root = replace(lowered.entry_function(), body=Sequential((Evaluate(target, ()),)))
    functions = tuple(root if function is lowered.entry_function() else function
                      for function in lowered.functions)
    malformed = replace(lowered, functions=functions)
    with pytest.raises(VerifyError, match="index must be an integer scalar"):
        verify_module(malformed)
