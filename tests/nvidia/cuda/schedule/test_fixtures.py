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
from functools import partial
from itertools import combinations
from math import prod
from pathlib import Path

import isl
import pytest
import torch

import tilefoundry.ir.hir.schedule as schedules
import tilefoundry.passes.transforms.convert_hir_to_tir as lowering_module
import tilefoundry.visitor_registry.instruction as instructions
from tests.core._source import import_dsl
from tests.core.schedule._fixtures import FixtureCase, _check_facts, _quiet_json, check_fixture
from tilefoundry.analysis.access import AliasKind
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
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
from tilefoundry.cli import main as cli_main
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
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef, collect_param_defs
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, operand_relations
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.arange import Arange as HirArange
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
from tilefoundry.ir.tir.arange import Arange as TirArange
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.memory import Copy, PtrOf, TensorView
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, MeshScope, Sequential, While
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    StorageKind,
    Swizzle,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.dim import DimAdd, DimMul, DimVar
from tilefoundry.ir.types.int_tuple import compatible_profile
from tilefoundry.ir.types.layout import apply, flatten, unflatten
from tilefoundry.ir.types.layout_algebra import cosize
from tilefoundry.ir.types.mesh import levels, make_mesh, starts
from tilefoundry.ir.types.stride import crd2idx
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import StmtVisitor, collect_exprs
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    access_relation_registry,
    identity_relations,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.candidates import candidate_operands
from tilefoundry.visitor_registry.contexts import TypeInferContext, VerifyContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, check_call, project
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.nn.sm80_mma import Mma
from tilefoundry_nvidia.cuda.ir.tir.nn.wgmma import Form, Wgmma

PLAIN = (
    "unit_mesh_axis_loop_carry",
    "reshard_locations",
    "chunk_rmsnorm",
    "fp8_block_scaled_gemm",
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
    "sm89_e4m3_gemm",
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
    "duplicate_output_storage": "return leaf (2,) shares storage with another output",
}
CANDIDATES_REFUSED = {
    "bulk_reduce_store": "source has no unscheduled candidate site",
    "wgmma_n16_within_warpgroup_share": "source has no unscheduled candidate site",
    "wgmma_b_mn_major_one_atom_column_slice": "source has no unscheduled candidate site",
    "wgmma_b_mn_major_two_atom_column_slice": "source has no unscheduled candidate site",
    "smem_padded_stages": "source has no unscheduled candidate site",
    "zero_trip_insert_slice": "source has no unscheduled candidate site",
    "strided_return_window": "source has no unscheduled candidate site",
    "inplace_output_update": "source has no unscheduled candidate site",
    "wgmma_b_mn_major_grouped_n64": "source has no unscheduled candidate site",
    "reshard_views": "source has no unscheduled candidate site",
    "scalar_ceildiv_gate": "source has no unscheduled candidate site",
    "register_index_window": "source has no unscheduled candidate site",
    "register_index_window_offset": "source has no unscheduled candidate site",
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
    "sm89_e4m3_gemm": "source has no unscheduled candidate site",
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
  T.arange                     all targets
  T.binary                     all targets
  T.cast                       all targets
  T.clamp                      all targets
  T.unary                      all targets
  T.where                      all targets
  T.copy_async                 cp.async
  T.copy_async_bulk            cp.async.bulk
  T.copy_async_tensor          cp.async.bulk.tensor
  T.copy_reduce_async_tensor   cp.async.bulk.tensor
  T.copy                       all targets
  T.ldmatrix                   ldmatrix
  T.stmatrix                   stmatrix
  T.relu                       all targets
  T.tiled_mma                  wgmma.mma_async, mma.sync
  T.mbarrier_arrive_expect_tx  mbarrier.expect_tx
  T.mbarrier_wait_parity       mbarrier.try_wait
  T.reduce                     all targets
"""
_H200_FACTS_JSON = {
    "target": "nvidia.h200_sxm",
    "instructions": [
        {"id": "T.arange", "capability": None},
        {"id": "T.binary", "capability": None},
        {"id": "T.cast", "capability": None},
        {"id": "T.clamp", "capability": None},
        {"id": "T.unary", "capability": None},
        {"id": "T.where", "capability": None},
        {"id": "T.copy_async", "capability": "cp.async"},
        {"id": "T.copy_async_bulk", "capability": "cp.async.bulk"},
        {"id": "T.copy_async_tensor", "capability": "cp.async.bulk.tensor"},
        {"id": "T.copy_reduce_async_tensor", "capability": "cp.async.bulk.tensor"},
        {"id": "T.copy", "capability": None},
        {"id": "T.ldmatrix", "capability": "ldmatrix"},
        {"id": "T.stmatrix", "capability": "stmatrix"},
        {"id": "T.relu", "capability": None},
        {"id": "T.tiled_mma", "capability": ["wgmma.mma_async", "mma.sync"]},
        {"id": "T.mbarrier_arrive_expect_tx", "capability": "mbarrier.expect_tx"},
        {"id": "T.mbarrier_wait_parity", "capability": "mbarrier.try_wait"},
        {"id": "T.reduce", "capability": None},
    ],
}
_SM89_FACTS_TEXT = """\
target nvidia.rtx_6000_ada
instructions
  T.arange      all targets
  T.binary      all targets
  T.cast        all targets
  T.clamp       all targets
  T.unary       all targets
  T.where       all targets
  T.copy_async  cp.async
  T.copy        all targets
  T.ldmatrix    ldmatrix
  T.relu        all targets
  T.tiled_mma   mma.sync
  T.reduce      all targets
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
    "bulk_reduce_store": 16_384,
    "wgmma_accumulator_stmatrix": 6_144,
    "wgmma_n16_within_warpgroup_share": 61_440,
    "wgmma_b_mn_major_one_atom_column_slice": 49_152,
    "wgmma_b_mn_major_two_atom_column_slice": 49_152,
    "loop_bound_from_outer_induction": 0,
    "finalize_footprint_mesh_offset": 0,
    "smem_padded_stages": 96,
    "view_visit_memo": 0,
    "persistent_tile_union_count": 163_840,
    "wgmma_b_mn_major_window_unit_mode": 34_816,
    "zero_trip_insert_slice": 0,
    "strided_return_window": 0,
    "inplace_output_update": 0,
    "wgmma_b_mn_major_grouped_n64": 18432,
    "wgmma_b_mn_major_wide_lbo": 16384,
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
    "sm89_e4m3_gemm": 512,
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
    "bulk_reduce_store": {},
    "wgmma_accumulator_stmatrix": {
        "thread@0:128#0": _RmemExpectation(
            12_288, "64x48 f32 accumulator; bf16 cast aliases the fragment"
        ),
    },
    "wgmma_n16_within_warpgroup_share": {
        "thread@128:256#0": _RmemExpectation(
            24_576, "64x96 f32 accumulator across two warpgroups; MMA aliases acc"
        ),
        "thread@0:384#0": _RmemExpectation(24_576, "parent envelope of the accumulator"),
    },
    "wgmma_b_mn_major_one_atom_column_slice": {
        "thread@128:128#0": _RmemExpectation(16384, "64x64 f32 accumulator; MMA aliases acc"),
        "thread@0:256#0": _RmemExpectation(16384, "parent envelope of the accumulator"),
    },
    "wgmma_b_mn_major_two_atom_column_slice": {
        "thread@128:128#0": _RmemExpectation(32768, "64x128 f32 accumulator; MMA aliases acc"),
        "thread@0:256#0": _RmemExpectation(32768, "parent envelope of the accumulator"),
    },
    "loop_bound_from_outer_induction": {
        "thread@0:32#0": _RmemExpectation(256, "32 f32 accumulator and loaded row elements"),
    },
    "finalize_footprint_mesh_offset": {
        "thread@0:32#0": _RmemExpectation(136, "32 f32 row elements and one i64 mesh offset"),
        "cta@0:4#0": _RmemExpectation(8, "one i64 mesh offset"),
        "thread@0:32#1": _RmemExpectation(256, "32 f32 accumulator and loaded row elements"),
        "cta@0:4#1": _RmemExpectation(8, "one i64 output offset"),
    },
    "smem_padded_stages": {},
    "view_visit_memo": {
        "thread@0:1#0": _RmemExpectation(32, "one eight-element f32 row"),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the output row"),
    },
    "zero_trip_insert_slice": {
        "thread@0:1#0": _RmemExpectation(16, "one four-element f32 update row"),
        "cta@0:1#0": _RmemExpectation(16, "parent envelope of the update row"),
    },
    "strided_return_window": {
        "thread@0:1#0": _RmemExpectation(12288, "24 * 256 bf16 register output elements"),
        "cta@0:32#0": _RmemExpectation(12288, "parent envelope of the register output"),
    },
    "inplace_output_update": {
        "cta@0:1#0": _RmemExpectation(16, "CTA envelope of the row register storage"),
    },
    "wgmma_b_mn_major_grouped_n64": {
        "thread@0:256#0": _RmemExpectation(
            32768, "two 64x64 f32 accumulator groups; MMA aliases acc"
        ),
    },
    "wgmma_b_mn_major_wide_lbo": {
        "thread@0:128#0": _RmemExpectation(32768, "64x128 f32 accumulator and MMA result alias"),
    },
    "register_index_window": {
        "thread@0:1#0": _RmemExpectation(
            32, "8 f32 output elements; index/view lifetimes reuse their storage"
        ),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the register output"),
    },
    "register_index_window_offset": {
        "thread@0:1#0": _RmemExpectation(
            32, "8 f32 output elements; index/view lifetimes reuse their storage"
        ),
        "cta@0:1#0": _RmemExpectation(32, "parent envelope of the register output"),
    },
    "register_full_like": {
        "thread@0:1#0": _RmemExpectation(
            65, "two independent 8-element f32 buffers and a predicate"
        ),
        "cta@0:1#0": _RmemExpectation(
            65, "parent envelope of the independent buffers and predicate"
        ),
    },
    "register_arange": {
        "thread@0:1#0": _RmemExpectation(
            64, "independent 8-element i32 indices and output buffers"
        ),
        "cta@0:1#0": _RmemExpectation(64, "parent envelope of the two independent buffers"),
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
        "thread@0:1#0": _RmemExpectation(64, "two independent live 2x4 f32 buffers"),
        "cta@0:1#0": _RmemExpectation(64, "parent envelope of the independent loop buffers"),
    },
    "submesh_slice_in_sibling": {},
    "duplicate_output_storage": {},
    "loop_carry_rebind": {
        "thread@0:32#0": _RmemExpectation(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
        "cta@0:1#0": _RmemExpectation(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
    },
    "loop_carry_window": {
        "thread@0:32#0": _RmemExpectation(96, "independent live Buffers: 1 * 32B + 1 * 64B"),
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
        "thread@0:32#0": _RmemExpectation(256, "independent live Buffers: 2 * 128B"),
        "thread@0:32#1": _RmemExpectation(256, "independent live Buffers: 2 * 128B"),
        "thread@0:32#2": _RmemExpectation(256, "independent live Buffers: 2 * 128B"),
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
        "thread@0:32#0": _RmemExpectation(384, "independent live Buffers: 3 * 128B"),
        "cta@0:1#0": _RmemExpectation(384, "independent live Buffers: 3 * 128B"),
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
        "thread@0:384#0": _RmemExpectation(
            131584, "independent live Buffers: 1 * 512B + 2 * 65536B"
        ),
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
    "sm89_e4m3_gemm": {
        "thread@0:32#0": _RmemExpectation(
            1_280,
            "512-byte f32 accumulator plus 512-byte lhs and 256-byte rhs e4m3 fragments",
        ),
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
        "thread@128:128#2": _RmemExpectation(16384, "independent live Buffers: 2 * 8192B"),
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
    "wgmma_b_mn_major_window_unit_mode": {
        "thread@0:256#0": _RmemExpectation(
            32768, "two 64x64 f32 accumulator groups; MMA aliases acc"
        ),
    },
    "persistent_tile_union_count": {
        "thread@128:256#0": _RmemExpectation(131_072, "128x256 f32 zero accumulator"),
        "thread@128:256#1": _RmemExpectation(131_072, "f32 phi/mma alias chain"),
        "thread@128:256#2": _RmemExpectation(131_072, "f32 loop result/bf16 cast alias"),
        "thread@0:384#0": _RmemExpectation(131_072, "parent envelope of one accumulator chain"),
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
    return next(
        value for value in vars(loaded).values() if isinstance(value, (Module, PrimFunction))
    )


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
            importlib.import_module(f"tests.nvidia.cuda.fixtures.schedule.plain.{name}")
        return
    module = importlib.import_module(f"tests.nvidia.cuda.fixtures.schedule.plain.{name}")
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
            schedules = [
                expr
                for expr in collect_exprs(function.body)
                if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)
            ]
            assert any(get_metadata(expr, record_type) is not None for expr in schedules)

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
        liveness = get_metadata(result.function, LivenessMetadata)
        local = {id(expr) for expr in collect_exprs(result.function.body)}
        region_windows = tuple(window for window in liveness.regions if id(window.region) in local)
        regions = tuple(window.region for window in region_windows)
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
        for window in region_windows:
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
    memory = report["functions"][0]["function_records"]["memory"]
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


@pytest.mark.parametrize("dtype", (DType.i32, DType.i64))
def test_schedule_arange_evaluates_static_coordinates(dtype) -> None:
    type_ = TensorType((8,), dtype, Layout((8,), (1,)), StorageKind.RMEM)
    torch_dtype = to_torch_dtype(dtype)
    result = _evaluate_call(
        ScheduleOp(op=TirArange(start=3, step=2)),
        ((type_, torch.empty(8, dtype=torch_dtype)),),
        type_,
    )
    assert torch.equal(result, torch.tensor([3, 5, 7, 9, 11, 13, 15, 17], dtype=torch_dtype))


def test_arange_candidates_refuse_a_symbolic_start() -> None:
    type_ = TensorType((8,), DType.i32, Layout((8,), (1,)), StorageKind.RMEM)
    call = Call(target=HirArange(type=type_, start=DimVar("start", 0, 8)), args=(), type=type_)
    with pytest.raises(ScheduleRefusal, match="start must be a static integer"):
        candidate_operands(call, TirArange)


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


def test_schedule_reduce_store_adds_to_an_explicit_nonzero_destination():
    cta = Mesh((Topology("cta", 2),), Layout((2,), (1,)))
    source_type = TensorType(
        (64, 64),
        DType.f32,
        ShardLayout(Layout((64, 64), (64, 1)), (Partial("sum"),), cta),
        StorageKind.SMEM,
    )
    result_type = TensorType((64, 64), DType.f32, Layout((64, 64), (64, 1)), StorageKind.GMEM)
    source = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    destination = torch.full_like(source, 7)
    result = _evaluate_call(
        ScheduleOp(op=CopyReduceAsyncTensor()),
        ((source_type, source), (result_type, destination)),
        result_type,
    )
    assert torch.equal(result, destination + source)


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


def _e4m3_gemm_tensor(name: str) -> torch.Tensor:
    """Integer inputs in [-4, 4] keep every partial sum of the product exact."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (16, 128), generator=generator).float().to(torch.float8_e4m3fn)
    weight = torch.randint(-4, 5, (8, 128), generator=generator).float().to(torch.float8_e4m3fn)
    return {"a": a, "b": weight.t(), "expected": a.float() @ weight.float().t()}[name]


def _check_e4m3_gemm(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    comparison = report["runs"][0]["outputs"][0]["fns"][0]
    assert comparison["fn"] == "allclose"
    assert comparison["rtol"] == 0 and comparison["atol"] == 0
    assert report["passed"] and comparison["passed"], (
        "K slicing, the (K, N) view of B and the accumulation across slices match torch"
    )


def _check_golden(out: Path, captured: pytest.CaptureResult[str], *, golden: Path) -> None:
    assert captured == ("", "")
    if golden.suffix == ".json":
        assert json.loads(out.read_text()) == json.loads(golden.read_text())
    else:
        assert out.read_bytes() == golden.read_bytes()


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
    for label, byte_count in (("12", 256), ("20", 224)):
        memory = rows[label]
        assert memory["operands"][0]["read"] == byte_count
        assert memory["operands"][1]["write"] == byte_count
    assert rows["20"]["buffer_bytes"] == 112
    assert rows["20"]["offsets"]
    for label in ("14", "15", "16", "17", "18"):
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
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/plain/reshard_locations.py",
            "{out}",
            "--memory",
            "--json",
        ),
        _check_reshard_locations,
    ),
    FixtureCase(
        "check binds two DimVars in a reshape",
        (
            "check",
            "tests/nvidia/cuda/fixtures/placed/two_dimvar_reshape.py:TwoDimVarReshape",
            "--inputs",
            "files:{tmp}/x.pt",
            "--expected",
            "{tmp}/expected.pt",
            "--dim",
            "T=8",
            "--dim",
            "C=16",
            "--device",
            "cpu",
            "--out",
            "output",
            "--fn",
            "allclose",
            "--atol",
            "0",
            "--rtol",
            "0",
            "--json",
            "{out}",
        ),
        _check_reshape_matches_torch,
        files={"x.pt": _reshape_tensor, "expected.pt": partial(_reshape_tensor, True)},
    ),
    FixtureCase(
        "finalize refuses a layout crossing logical axes where it is declared",
        ("schedule", "finalize", "tests/nvidia/cuda/fixtures/diagnostics/layout_crossing_axes.py", "{out}"),
        _check_crossing_axes_refusal,
        code=1,
    ),
    FixtureCase(
        "analyze help names only Module.function",
        ("analyze", "--help"),
        _check_selector_help,
    ),
    FixtureCase(
        "finalize binds an open row dimension",
        (
            "schedule",
            "finalize",
            "tests/nvidia/cuda/fixtures/schedule/hir/dim/row_copy.py",
            "{out}",
            "--dim=rows=128",
        ),
        _check_finalized_rows,
    ),
    FixtureCase(
        "finalize refuses an unbound row dimension without output",
        ("schedule", "finalize", "tests/nvidia/cuda/fixtures/schedule/hir/dim/row_copy.py", "{out}"),
        _check_finalize_unbound_dim,
        code=1,
    ),
    FixtureCase(
        "gather reads selected rows and caps the footprint",
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py",
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
            "tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py",
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
            "tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--json",
        ),
        _check_gmem_gather_refusal,
    ),
    FixtureCase(
        "analyzed smem peak matches the lowered placement",
        ("schedule", "finalize", "tests/nvidia/cuda/fixtures/schedule/hir/sparse_decode.py", "{out}"),
        _check_smem_placement,
    ),
    *(
        FixtureCase(
            name,
            (
                "check",
                f"tests/nvidia/cuda/fixtures/schedule/{stage}/sparse_decode.py:SparseDecode",
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
    *(
        FixtureCase(
            name,
            (
                "check",
                f"tests/nvidia/cuda/fixtures/schedule/{stage}/sm89_e4m3_gemm.py:SM89_E4M3_GEMM",
                "--inputs",
                "files:{tmp}/a.pt,{tmp}/b.pt",
                "--expected",
                "{tmp}/expected.pt",
                "--device",
                "cpu",
                "--out",
                "output",
                "--fn",
                "allclose",
                "--atol",
                "0",
                "--rtol",
                "0",
                "--json",
                "{out}",
            ),
            _check_e4m3_gemm,
            files={f"{key}.pt": partial(_e4m3_gemm_tensor, key) for key in ("a", "b", "expected")},
        )
        for name, stage in (
            ("plain e4m3 gemm matches torch", "plain"),
            ("scheduled e4m3 gemm matches torch", "hir"),
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
                "sm_89 facts list no WGMMA or TMA instruction",
                "nvidia.rtx_6000_ada",
                (),
                _SM89_FACTS_TEXT,
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
            f"tests/nvidia/cuda/fixtures/schedule/plain/{CANDIDATE_GOLDEN.name.removesuffix('.candidates.txt')}.py",
            "{out}",
        ),
        partial(_check_golden, golden=CANDIDATE_GOLDEN),
    ),
    FixtureCase(
        "analyze writes the memory report",
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/hir/gemm_8192x17408x5120_tma_store.py",
            "{out}",
            "--memory",
        ),
        partial(_check_golden, golden=ANALYZED_GOLDEN),
    ),
)


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
    assert as_script(program) == path.read_text()


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
        source = f"tests/nvidia/cuda/fixtures/schedule/plain/{name}.py"
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
    assert len(matmuls) == 9
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
    source = f"tests/nvidia/cuda/fixtures/schedule/plain/{name}.py"
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
        assert {item["id"] for item in row["refused"]} == {
            "T.cuda.sm90.Wgmma",
            "T.cuda.sm80.Mma",
            "T.cuda.sm89.Mma",
        }
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
    tuple(
        path
        for path in HIR
        if path.stem not in ("wgmma_b_mn_major_window_unit_mode", "view_visit_memo")
    ),
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


@pytest.mark.parametrize(
    "row_scale, leading, position", ((2, None, "a_stride_00"), (1, 1, "a_stride_11"))
)
def test_wgmma_stride_refusal_names_operand_and_descriptor_position(
    row_scale, leading, position
) -> None:
    function = _prim_in(
        Path(__file__).parents[1] / "fixtures" / "schedule" / "tir" / "wgmma_a_k_major.py"
    )

    class RejectOne(_OperandMatchVisitor):
        rejected = False

        def visit_Evaluate(self, stmt: Evaluate) -> None:
            if self.rejected or not isinstance(getattr(stmt.callable, "atom", None), Wgmma):
                return
            lhs = stmt.args[1].type
            shard = lhs.layout
            outer = shard.layout
            row, column = outer.strides
            column = (column[0] if leading is None else leading, column[1])
            bad = replace(outer, strides=((row[0], row[1] * row_scale), column))
            operand = Var("bad_lhs", type=replace(lhs, layout=replace(shard, layout=bad)))
            with pytest.raises(ScheduleRefusal, match=rf"lhs:.*{position}"):
                super().visit_Evaluate(replace(stmt, args=(stmt.args[0], operand, stmt.args[2])))
            self.rejected = True

    visitor = RejectOne()
    visitor.visit(function.body)
    assert visitor.rejected


def _assert_live_smem_views_do_not_overlap(function: PrimFunction, source: Path) -> None:
    """Check literal views in lowered TIR, including uses across loop backedges."""
    views: list[tuple[Var, int, int]] = []
    defined: dict[int, int] = {}
    last: dict[int, int] = {}
    aliases: dict[int, set[int]] = {}
    clock = 0

    def use(expressions) -> None:
        for expression in expressions:
            for value in collect_exprs(expression):
                if isinstance(value, Var):
                    for key in aliases.get(id(value), {id(value)}):
                        if key in defined:
                            last[key] = clock

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
            aliases[id(statement.var)] = {
                key
                for value in collect_exprs(statement.value)
                if isinstance(value, Var)
                for key in aliases.get(id(value), {id(value)})
            } or {id(statement.var)}
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

    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    lowered = _tir_in(out)
    functions = (
        tuple(frame.function for frame in list_call_paths(lowered).values())
        if isinstance(lowered, Module)
        else (lowered,)
    )
    found = []
    for function in functions:
        _assert_live_smem_views_do_not_overlap(function, source)
        found.extend(_direct_operand_matches(function))
    if source.stem == "loop_outputs":
        assert lowered.output_count == 3
        assert import_dsl(as_script(lowered)).output_count == 3
    if source.stem == "zero_trip_insert_slice":
        visitor = _RetainedStatements()
        visitor.visit(functions[0].body)
        assert len(visitor.loops) == 1
        loop = visitor.loops[0]
        assert tuple(static_dim_value(value) for value in (loop.start, loop.stop, loop.step)) == (
            0,
            0,
            1,
        )
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
    return (
        source,
        module,
        analyze(
            *check_program(module, module.entry_function()),
            analysis=("memory",),
        ),
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
    module = _module_in(four_stage)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    memory = get_metadata(result.function, RegionMemoryMetadata)
    assert "smem placement peak 256.00KiB exceeds capacity 227.00KiB" in memory.errors
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


def test_lowering_rejects_nonidentity_inverse_for_a_written_return():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class WrittenReturnWindow:
    @func
    def gemm(x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as cta:
            out = tf.zeros(Tensor[(2, 4), "f32"])
            updated = tf.insert_slice(out, x, (1, 0))
            return updated[1:2, :]
""")
    with pytest.raises(
        lowering_module.LoweringError, match="requires a non-identity inverse relation"
    ):
        finalize(module)


def test_zero_trip_carry_keeps_direct_address_proofs_and_the_loop():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ZeroTripCarry:
    @func
    def gemm(x: Tensor[(4,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                acc = tf.schedule((x, tf.empty(Tensor[(4,), "f32", "rmem"])), op=T.copy())
                for k in range(0):
                    acc = tf.schedule((x, tf.empty(Tensor[(4,), "f32", "rmem"])), op=T.copy())
                return tf.schedule((acc, tf.empty(Tensor[(4,), "f32"])), op=T.copy())
""")
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    summary = get_metadata(result.function, AllocationSummaryMetadata)
    carry = [
        proof
        for value in summary.values
        for proof in value.aliases
        if proof.kind in (AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD)
    ]
    assert {proof.kind for proof in carry} == {AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD}
    assert all(proof.relation.relation.is_identity() for proof in carry)
    lowered = finalize(module)
    verify_module(lowered)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    assert len(visitor.loops) == 1
    loop = visitor.loops[0]
    assert tuple(static_dim_value(value) for value in (loop.start, loop.stop, loop.step)) == (
        0,
        0,
        1,
    )
    body = _RetainedStatements()
    body.visit(loop.body)
    assert sum(isinstance(op, Copy) for op in body.operations) == 1


def test_dynamic_window_keeps_a_mixed_scalar_and_constant_key():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class Window:
    @func
    def gemm(x: Tensor[(64, 8), "f32"], starts: Tensor[(1,), "i64"]) -> Tensor[(8, 8), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                held = tf.schedule((starts, tf.empty(Tensor[(1,), "i64", (1,), "rmem"])), op=T.copy())
                start = tf.reshape(held, new_shape=())
                window = x[start + 1:start + 9, 0:8]
                return tf.schedule((window, tf.empty(Tensor[(8, 8), "f32", (8, 8), "rmem"])), op=T.copy())
""")
    lowered = finalize(module)
    verify_module(lowered)
    printed = as_script(lowered)
    verify_module(import_dsl(printed))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    views = [
        value
        for value in visitor.bindings.values()
        if isinstance(value, Call)
        and isinstance(value.target, TensorView)
        and value.type.shape == (8, 8)
    ]
    assert any(
        view.type.layout.strides == (8, 1)
        and any(
            isinstance(expr, Call)
            and isinstance(expr.target, Slice)
            and expr.target.sizes == (1, 1)
            and isinstance(expr.args[1].elements[0], Call)
            and expr.args[1].elements[1].value == 0
            for expr in collect_exprs(view)
        )
        for view in views
    )


def test_register_row_pointer_keeps_the_split_axis_whole():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 16)))
class RegisterRows:
    @func
    def gemm(x: Tensor[(16,16), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(16,), names=("col",)) as th:
                held = tf.schedule((x, tf.empty(Tensor[(16,16), "f32", (16,16 @ th.col), "rmem"])), op=T.copy())
                out = tf.zeros(Tensor[(16,16), "f32"])
                for i in range(16):
                    out = tf.insert_slice(out, held[i:i+1, :], (i,0))
                return out
""")
    lowered = finalize(module)
    verify_module(lowered)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    slices = [
        expr
        for value in visitor.bindings.values()
        for expr in collect_exprs(value)
        if isinstance(expr, Call)
        and isinstance(expr.target, Slice)
        and isinstance(expr.args[0].type.layout, ShardLayout)
    ]
    assert slices and all(expr.target.sizes[1] == 16 for expr in slices)


@pytest.mark.parametrize("start,offset", [(0, 0), (64, 4096)])
def test_composed_view_keeps_its_layout_when_logical_rank_changes(start, offset):
    module = import_dsl(
        """
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ComposedReshape:
    @func
    def gemm(x: Tensor[(64,128), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                layout = ComposedLayout(Swizzle(3,4,3), 0, Layout(((8,8),(2,4,16)), ((512,64),(4096,16,1))))
                shared = tf.schedule((x, tf.empty(Tensor[(64,128), "bf16", layout, "smem"])), op=T.copy())
                reshaped = tf.reshape(shared[:,START:START+64], (8,8,16,4))
                return tf.schedule((reshaped, tf.empty(Tensor[(8,8,16,4), "bf16", "rmem"])), op=T.copy())
""".replace("START", str(start))
    )
    lowered = finalize(module)
    verify_module(lowered)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    views = [
        value
        for value in visitor.bindings.values()
        if isinstance(value, Call)
        and isinstance(value.target, TensorView)
        and value.type.shape == (8, 8, 16, 4)
    ]
    assert any(
        value.type.layout
        == ComposedLayout(Swizzle(3, 4, 3), offset, Layout((8, 8, 16, 4), (512, 64, 4, 1)))
        for value in views
    )


@pytest.mark.parametrize(
    "width,start,shape,modes,strides,offset",
    [
        (256, 0, (16, 8, 16, 16), (16, 8, (4, 4), 16), (512, 64, (8192, 16), 1), 0),
        (32, 0, (16, 8, 2, 16), (16, 8, 2, 16), (512, 64, 16, 1), 0),
        (128, 128, (16, 8, 8, 16), (16, 8, (2, 4), 16), (512, 64, (8192, 16), 1), 16384),
    ],
)
def test_composed_reshape_retains_non_affine_outer_modes(
    width, start, shape, modes, strides, offset
):
    module = import_dsl(
        """
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ComposedReshape:
    @func
    def gemm(x: Tensor[(128,256), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                layout = ComposedLayout(Swizzle(3,4,3), 0, Layout((128,(4,64)), (64,(8192,1))))
                shared = tf.schedule((x, tf.empty(Tensor[(128,256), "bf16", layout, "smem"])), op=T.copy())
                reshaped = tf.reshape(shared[:,START:START+WIDTH], SHAPE)
                return tf.schedule((reshaped, tf.empty(Tensor[SHAPE, "bf16", "rmem"])), op=T.copy())
""".replace("WIDTH", str(width))
        .replace("START", str(start))
        .replace("SHAPE", repr(shape))
    )
    lowered = finalize(module)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    expected = ComposedLayout(Swizzle(3, 4, 3), offset, Layout(modes, strides))
    assert any(
        isinstance(value, Call)
        and isinstance(value.target, TensorView)
        and value.type.layout == expected
        for value in visitor.bindings.values()
    )


def test_address_origin_can_divide_a_runtime_loop_index():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ModularWindow:
    @func
    def gemm(x: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                out = tf.zeros(Tensor[(8,), "f32", "rmem"])
                for u in range(cta.b, 8, 1):
                    start = (u % 4) * 8
                    out = tf.schedule((x[start:start+8], tf.empty(Tensor[(8,), "f32", "rmem"])), op=T.copy())
                return out
""")
    verify_module(import_dsl(as_script(finalize(module))))


@pytest.mark.parametrize("coordinate_division", [False, True])
def test_view_strides_distinguish_coordinate_and_parameter_divisions(coordinate_division):
    source = Var("x", type=TensorType((64,), DType.f32, Layout((64,), (1,)), StorageKind.GMEM))
    desired = replace(source.type, shape=(8,), layout=Layout((8,), (1,)))
    relation = AccessRelation(isl.map("{ [i] -> [floor(i/2)] : 0 <= i < 8 }"))
    if not coordinate_division:
        tile = Var("tile", type=TensorType.umat_scalar(DType.i64))
        relation = AccessRelation(
            isl.map("[tile] -> { [i] -> [i + floor(tile/4)] : 0 <= i < 8 }"), {"tile": tile}
        )
    function = Function.build(name="view", params=(source,), body=source, return_type=source.type)
    storage = lowering_module.StoragePlan(function)
    values = {} if coordinate_division else {id(tile): tile}
    if coordinate_division:
        with pytest.raises(lowering_module.LoweringError, match="non-affine address"):
            storage.view(source, relation, desired, values=values)
    else:
        view = storage.view(source, relation, desired, values=values)
        assert view.type.layout == Layout((8,), (1,))


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

    _assert_cli_lowering_error(
        source, "_UnstatedInstruction states no access relations", tmp_path, capsys
    )


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
        "addressable gmem result but no offsets",
        tmp_path,
        capsys,
    )


def test_shared_projection_rejects_partial_atom_tiles() -> None:
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/wgmma_repeat_along_k.py"
    module = _module_in(source)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    schedule = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and isinstance(expr.target.op, TiledMma)
    )
    acc, lhs, rhs = (arg.type for arg in schedule.args)
    with pytest.raises(
        ScheduleRefusal, match="iteration extent 17 is not divisible by single-issue extent 16"
    ):
        project(
            schedule.target.op,
            (replace(acc, shape=(acc.shape[0], 17)), lhs, replace(rhs, shape=(rhs.shape[0], 17))),
            None,
        )


@pytest.mark.parametrize("failure", ("leftmost", "factor"))
def test_grouped_projection_names_incompatible_repeat_mode(failure):
    source = next(path for path in HIR if path.stem == "wgmma_n16_within_warpgroup_share")
    module = _module_in(source)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    schedule = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call)
        and isinstance(value.target, ScheduleOp)
        and isinstance(value.target.op, TiledMma)
    )
    acc, lhs, rhs = (arg.type for arg in schedule.args)
    if failure == "leftmost":
        layout = replace(
            acc.layout, layout=Layout((4, 2, 8, 2, 2, 4, 6), (16, 8, 1, 3072, 1024, 128, 64))
        )
        operands = (replace(acc, layout=layout), lhs, rhs)
        reason = "acc axis 1: repeat is not leftmost at mode 4"
    else:
        layout = replace(
            rhs.layout,
            outer=Layout(((3, 4, 16), (3, 2, 2, 8)), ((6144, 16, 1), (8192, 1024, 512, 64))),
        )
        operands = (acc, lhs, replace(rhs, layout=layout))
        reason = "rhs iteration axis 1: repeat mode .* cannot factor as"
    with pytest.raises(ScheduleRefusal, match=reason):
        project(schedule.target.op, operands, None)


@pytest.mark.parametrize(
    ("name", "issues"), (("wgmma_repeat_along_n_order", 4), ("wgmma_k_slices_of_wide_run", 2))
)
def test_lowering_writes_static_issues_without_readmitting(name, issues, monkeypatch):
    source = Path(__file__).parents[1] / f"fixtures/schedule/hir/{name}.py"
    module = _module_in(source)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))

    def reject(*args, **kwargs):
        raise AssertionError("lowering repeated instruction admission")

    monkeypatch.setattr(schedules, "check_call", reject)
    monkeypatch.setattr(instructions, "check_call", reject)
    lowered = lowering_module.Lowering(
        TypeInferContext(module=result.module, function=result.function, owner=result.function),
        result.function,
    ).run()
    statements = _RetainedStatements()
    statements.visit(lowered.body)
    assert not any(loop.induction_var.name.startswith("o_") for loop in statements.loops)
    issued = [stmt for stmt in statements.statements if isinstance(stmt.callable, TiledMma)]
    assert len(issued) == issues
    for stmt in issued:
        for arg in stmt.args:
            for expr in collect_exprs(statements.bindings.get(id(arg), arg)):
                if isinstance(expr, Call) and isinstance(expr.target, Slice):
                    assert all(
                        static_dim_value(start) is not None for start in expr.args[1].elements
                    )


class _RetainedStatements(StmtVisitor[None]):
    def __init__(self):
        self.calls = []
        self.loops = []
        self.operations = []
        self.statements = []
        self.bindings = {}

    def visit_LetStmt(self, stmt, ctx=None):
        self.bindings[id(stmt.var)] = stmt.value

    def storage(self, value):
        if isinstance(value, Var) and id(value) in self.bindings:
            return self.storage(self.bindings[id(value)])
        if isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf, Slice)):
            return self.storage(value.args[0])
        return value

    def visit_For(self, stmt, ctx=None):
        self.loops.append(stmt)
        self.visit(stmt.body)

    def visit_Evaluate(self, stmt, ctx=None):
        self.operations.append(stmt.callable)
        self.statements.append(stmt)
        if isinstance(stmt.callable, (PrimFunction, ModuleList, Call)):
            self.calls.append(stmt)


def test_reduce_store_stays_in_its_warp_scope():
    source = next(path for path in HIR if path.stem == "bulk_reduce_store")
    lowered = finalize(_module_in(source))

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
    visitor.visit(lowered.entry_function().body)
    assert visitor.found == 1


def test_grouped_n16_issues_keep_source_addresses():
    source = next(path for path in HIR if path.stem == "wgmma_n16_within_warpgroup_share")
    module = _module_in(source)
    checked = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    schedule = next(
        value
        for value in collect_exprs(checked.function.body)
        if isinstance(value, Call)
        and isinstance(value.target, ScheduleOp)
        and isinstance(value.target.op, TiledMma)
    )
    original_c, _, original_b = (arg.type for arg in schedule.args)
    lowered = finalize(module)
    visitor = _RetainedStatements()
    function = lowered.entry_function()
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

def test_authored_views_survive_region_forwarding_without_extra_wrappers():
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "view_visit_memo.py"
    lowered = finalize(_module_in(source))
    verify_module(lowered)
    function = lowered.entry_function()
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


@pytest.mark.parametrize(
    "name",
    (
        "wgmma_b_mn_major_grouped_n64",
        "wgmma_b_mn_major_window_unit_mode",
        "wgmma_b_mn_major_wide_lbo",
        "smem_padded_stages",
    ),
)
def test_smem_slots_reserve_all_physical_bytes(name):
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / f"{name}.py"
    module = _module_in(source)
    analyzed = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    peak = get_metadata(analyzed.function, RegionMemoryMetadata).peak_for("smem").peak_bytes
    function = finalize(module).entry_function()
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


def _returned_view(storage, chain=False):
    type_ = TensorType((32,), DType.f32, Layout((32,), (1,)), storage)
    param = Var("x", type=type_)
    function = Function.build(name="run", params=(param,), body=param, return_type=type_)
    leaf = Module("View", (function,), "run")
    slot = ModuleList(leaf)
    activation = replace(param)
    call = Call(slot, (activation,), type=type_)
    if chain:
        call = Call(slot, (call,), type=type_)
    root = Function.build(name="root", params=(activation,), body=call, return_type=type_)
    return Module(
        "ViewCaller",
        (root,),
        "root",
        children={"child": slot},
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)),
    )


@pytest.mark.parametrize("storage", [StorageKind.GMEM, StorageKind.SMEM])
@pytest.mark.parametrize("chain", [False, True])
def test_a_returned_view_passes_its_analyzed_storage_to_the_callee(storage, chain):
    lowered = finalize(_returned_view(storage, chain))
    verify_module(lowered)
    function = lowered.entry_function()
    visitor = _RetainedStatements()
    visitor.visit(function.body)
    assert len(visitor.calls) == (2 if chain else 1)
    for prior in visitor.calls[:-1]:
        assert visitor.storage(prior.args[0]) is function.params[0]
        assert visitor.storage(prior.args[-1]) is function.params[0]
    call = visitor.calls[-1]
    assert call.args[0] is not call.args[-1]
    assert visitor.storage(call.args[0]) is function.params[0]
    assert visitor.storage(call.args[-1]) is function.params[-1]
    assert not any(isinstance(op, Copy) for op in visitor.operations)
    callee = _RetainedStatements()
    callee.visit(lowered.children[call.callable.key].elements[0].entry_function().body)
    assert not callee.operations


@pytest.mark.parametrize("read_old", [False, True])
@pytest.mark.parametrize("reshape", [False, True])
def test_inplace_output_reuses_storage_or_refuses_a_nonidentity_inverse(
    tmp_path, read_old, reshape
):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    if read_old or reshape:
        text = source.read_text()
        if read_old:
            text = text.replace("(x, tf.empty", "(out[0:1, :], tf.empty")
        if reshape:
            text = text.replace(
                "return update(out, x)", "return tf.reshape(update(out, x), (4, 2))"
            )
        rewritten = tmp_path / source.name
        rewritten.write_text(text)
        source = rewritten
    if reshape:
        with pytest.raises(
            lowering_module.LoweringError, match="requires a non-identity inverse relation"
        ):
            finalize(_module_in(source))
        return
    lowered = finalize(_module_in(source))
    verify_module(lowered)
    function = lowered.entry_function()
    visitor = _RetainedStatements()
    visitor.visit(function.body)
    assert len(visitor.calls) == 1
    call = visitor.calls[0]
    assert call.args[0] is not call.args[-1]
    assert visitor.storage(call.args[0]) is function.params[0]
    assert visitor.storage(call.args[-1]) is function.params[-1]
    callee = _RetainedStatements()
    callee.visit(call.callable.body)
    copies = [stmt for stmt in callee.statements if isinstance(stmt.callable, Copy)]
    assert len(copies) == 2
    assert callee.storage(copies[0].args[0]) is call.callable.params[0 if read_old else 1]
    assert callee.storage(copies[-1].args[-1]) is call.callable.params[-1]
    assert not any(stmt.args[0].type.shape == (2, 4) for stmt in copies)


def test_read_only_call_before_an_update_keeps_the_input_storage(tmp_path):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    text = (
        source.read_text()
        .replace(
            "    @func\n    def update",
            '    @func\n    def identity(out: Tensor[(2, 4), "f32"]):\n'
            "        return out\n\n    @func\n    def update",
        )
        .replace(
            "row = tf.schedule((x,",
            "old = identity(out)\n                row = tf.schedule((old[0:1, :],",
        )
        .replace("return tf.insert_slice(out,", "return tf.insert_slice(old,")
    )
    path = tmp_path / source.name
    path.write_text(text)
    lowered = finalize(_module_in(path))
    verify_module(lowered)
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    update = visitor.calls[0].callable
    body = _RetainedStatements()
    body.visit(update.body)
    assert len(body.calls) == 1
    identity = body.calls[0]
    assert body.storage(identity.args[0]) is update.params[0]
    assert body.storage(identity.args[-1]) is update.params[0]
    copies = [stmt for stmt in body.statements if isinstance(stmt.callable, Copy)]
    assert body.storage(copies[0].args[0]) is update.params[0]
    assert body.storage(copies[-1].args[-1]) is update.params[-1]


@pytest.mark.parametrize(
    "wrong,refused",
    [
        ("count", "arg count"),
        ("type", r"arg\[0\] type"),
        ("recursion", "recursion not allowed"),
    ],
)
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
    lowered = finalize(
        _module_in(Path(__file__).parents[1] / "fixtures/schedule/hir/module_instances.py")
    )
    slot = lowered.children["layers"]
    target = Call(
        TupleGetItem(),
        (slot, Constant(0.5, type=TensorType.scalar(DType.f32))),
        type=slot.type.fields[0],
    )
    root = replace(lowered.entry_function(), body=Sequential((Evaluate(target, ()),)))
    functions = tuple(
        root if function is lowered.entry_function() else function for function in lowered.functions
    )
    malformed = replace(lowered, functions=functions)
    with pytest.raises(VerifyError, match="index must be an integer scalar"):
        verify_module(malformed)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_fixture(case, tmp_path, capsys):
    check_fixture(case, tmp_path, capsys)
