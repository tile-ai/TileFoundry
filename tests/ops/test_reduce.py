"""Reduce's Partial algebra, its cross-CTA limit, and its real reduce kernel.

Which reductions commute with which ``Partial``; that a reduced Split collapses
to ``Broadcast`` (layout positions shrunk to size 1 / stride 0) while a Split on
a non-reduced axis survives; that a reduce crossing CTAs is refused rather than
silently downgraded; and a real GPU run of the warp-only path.
"""
from __future__ import annotations

import re

import pytest
import torch

import tilefoundry
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
    split_local_extents,
)
from tilefoundry import func, module
from tilefoundry.codegen.cuda.module import emit_cuda_module
from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.passes.transforms.hir_to_tir import _analyze_cross_warp_workspace
from tilefoundry.target import CudaTarget

_RMEM = StorageKind.RMEM
_BF = DType.bf16
# Two-axis mesh; the reduce cases reuse it for input and expectation so the
# preserved mesh compares equal.
_M = make_mesh((6, 32), ("w", "t"))

_PARTIAL_MESH = make_mesh((4,))
_PSUM = make_shard_tensor_type((8, 16), mesh=_PARTIAL_MESH, attrs=(Partial("sum"),), dtype=DType.f32)
_PMAX = make_shard_tensor_type((8, 16), mesh=_PARTIAL_MESH, attrs=(Partial("max"),), dtype=DType.f32)


def test_plain_reduce_layout_describes_the_reduced_result():
    source = make_tensor_type(
        (1, 16, 1024, 128),
        _BF,
        layout=Layout(shape=(1, 16, 1024, 128), strides=(2097152, 131072, 128, 1)),
    )

    result = infer_call(
        Reduce(axes=(-1,), keepdim=False, kind=ReduceKind.SUM), source
    )

    assert result.shape == (1, 16, 1024)
    assert result.layout == Layout(
        shape=(1, 16, 1024), strides=(16384, 1024, 1)
    )

    replicated = make_shard_tensor_type(
        (1, 16, 1024, 128),
        _BF,
        mesh=make_mesh((4,)),
        attrs=(Broadcast(),),
    )
    replicated_result = infer_call(
        Reduce(axes=(-1,), keepdim=False, kind=ReduceKind.SUM), replicated
    )
    assert replicated_result.layout.layout.shape == (1, 16, 1024)
    assert replicated_result.layout.attrs == (Broadcast(),)


@pytest.mark.parametrize(
    "op,input_type,expected",
    [
        (Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM), _PSUM, _PSUM),
        (Reduce(axes=(1,), keepdim=True, kind=ReduceKind.MAX), _PMAX, _PMAX),
    ],
    ids=["sum_over_partial_sum", "max_over_partial_max"],
)
def test_reduce_partial_commutes(op, input_type, expected):
    out = infer_call(op, input_type)
    assert out.layout.attrs == expected.layout.attrs


@pytest.mark.parametrize(
    "case",
    [
        TypeInferCase(
            "sum_over_partial_max",
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
            (_PMAX,),
            ExpectedError(match="mesh axis 0"),
        ),
        TypeInferCase(
            "max_over_partial_sum",
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.MAX),
            (_PSUM,),
            ExpectedError(match="mesh axis 0"),
        ),
    ],
    ids=lambda case: case.name,
)
def test_reduce_partial_rejects_noncommuting(case):
    run_typeinfer_case(case)


# ── sharded carries ───────────────────────────────────────────────────────
# Reduce shrinks every reduced-axis layout position to size 1 / stride 0
# (broadcast view) while preserving the layout's own rank, so this checks output
# shape and which mesh axis stays genuinely `Split` vs collapses to `Broadcast`,
# not the internal layout position count a valid `Reduce` happens to produce.


def test_preserves_non_reduced_axis_split():
    """A Split on the non-reduced axis is preserved; the reduced axis ->
    Broadcast, with its layout positions shrunk to local extent 1."""
    x_ty = make_shard_tensor_type(
        (12, 32), mesh=_M, attrs=(Split(0), Split(1)), dtype=_BF, storage=_RMEM,
    )
    ty = infer_call(Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM), x_ty)
    assert tuple(ty.shape) == (12, 1)
    assert tuple(type(a).__name__ for a in ty.layout.attrs) == ("Split", "Broadcast")
    assert split_local_extents(ty) == [1]


# ── Cross-warp reduce path selection (runtime-derived, no op attribute) ──────
#
# The runtime has two sharded multi-warp templates: ``reduce_intra_cta`` (lane
# butterfly + cross-warp combine) and ``reduce_cross_warp`` (cross-warp combine
# only, each lane keeps its own output cells). Which one applies is a pure
# function of the operand layouts — a reduced Split on a lane axis vs on a
# warp-only axis. Codegen emits one uniform ``reduce`` entry; the runtime
# derives the level and its ``warps_per_group`` from ``(src, dst)`` and the
# ``Reduce`` op carries no selection attribute. The workspace *capacity* is still
# sized by the lowering (``_analyze_cross_warp_workspace``).

# rmsnorm-like: reduce the last axis, whose Split covers both the warp (w) and
# lane (t) mesh axes → a reduced lane axis → intra-cta.
_THREAD_A = Topology("thread", 6 * 32)
_MESH_A = make_mesh((6, 32), ("w", "t"), topology=_THREAD_A)
# cross-expert-like: reduce the warp axis (tk) only; the lane axis (hc) carries
# distinct output cells → no reduced lane axis → cross-warp.
_THREAD_B = Topology("thread", 4 * 32)
_MESH_B = make_mesh((4, 32), ("tk", "hc"), topology=_THREAD_B)


def _case_a_src():
    return make_shard_tensor_type(
        (1, 1536), mesh=_MESH_A, attrs=(Split(1), Split(1)), dtype=_BF, storage=_RMEM,
    )


def _case_b_src():
    return make_shard_tensor_type(
        (4, 32), mesh=_MESH_B, attrs=(Split(0), Split(1)), dtype=_BF, storage=_RMEM,
    )


def test_analyze_workspace_reports_lane_reduced_and_sizes():
    # The lowering reports only the values it needs to size the staging buffer:
    # (workspace_size, dtype, lane_reduced). warps_per_group is runtime-derived.
    # Case A: the reduced axis covers the warp mesh axis w(6) and the lane axis
    # t; the lane butterfly folds t, the 6 warps combine → total_warps=6,
    # lane_reduced.
    ws_a, _dt_a, lane_a = _analyze_cross_warp_workspace(_case_a_src(), (-1,))
    assert (ws_a, lane_a) == (6, True)
    # Case B: the reduce crosses the 4 warps only; each lane keeps its own cell →
    # total_warps=4, not lane_reduced.
    ws_b, _dt_b, lane_b = _analyze_cross_warp_workspace(_case_b_src(), (0,))
    assert (ws_b, lane_b) == (4, False)


def test_analyze_rejects_cross_cta_reduce():
    # A reduced Split on a cta-topology mesh axis spans CTAs — cross-CTA reduce
    # is not supported and MUST raise rather than fall back to intra_cta.
    mesh_cta = Mesh(
        topologies=(Topology("cta", 2), Topology("thread", 32)),
        layout=Layout(shape=(2, 32), strides=(32, 1)),
        names=("c", "t"),
    )
    src = make_shard_tensor_type(
        (2, 32), mesh=mesh_cta, attrs=(Split(0), Split(1)), dtype=_BF, storage=_RMEM,
    )
    with pytest.raises(NotImplementedError, match="cross-CTA"):
        _analyze_cross_warp_workspace(src, (0,))


# ── Cross-warp reduce end-to-end (folded from the former e2e file) ───────────
#
# A warp-only reduction (each lane keeps its own output cell) drives the runtime
# ``reduce_cross_warp`` path via the uniform ``reduce`` entry. Full GPU
# compile + run + numeric compare, plus the codegen-emit shape.

@module(entry="cross_warp_sum", topologies=(Topology("thread", 4 * 32),))
class _CrossWarpSumModule:

    @func
    def cross_warp_sum(a: Tensor[(4, 32), 'f32']):
        with Mesh(("thread",), (4, 32), ('tk', 'hc')) as m:
            # Axis 0 (tk) spans the four warps; axis 1 (hc) is the lane axis and
            # carries distinct output cells. Reducing axis 0 crosses warps only.
            a_reg = tf.reshard(a, (4 @ m.tk, 32 @ m.hc), 'rmem')
            s = tf.reduce(a_reg, (0,), True, ReduceKind.SUM)
            return tf.reshard(s, (1, 32 @ m.hc), 'gmem')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cross_warp_sum_matches_torch() -> None:
    rm = tilefoundry.compile(_CrossWarpSumModule, target=CudaTarget("nvidia.h200_sxm"))
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    out = rm(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(0, keepdim=True), rtol=1e-4, atol=1e-4)


def test_cross_warp_sum_emits_reduce() -> None:
    # Codegen emits the uniform reduce entry (no reduce_intra_cta /
    # reduce_cross_warp call, no warps_per_group argument) — the runtime derives
    # the level + wpg. The workspace capacity is still sized by the lowering:
    # per (warp, lane, cell) = 4 warps × 32 lanes × 1 cell = 128 slots.
    lowered = tilefoundry.lower(_CrossWarpSumModule, target=CudaTarget("nvidia.h200_sxm"))
    groups = group_functions_by_target(lowered)
    target, functions = next(iter(groups.items()))
    src = emit_cuda_module(lowered, functions, target).source
    assert re.search(r"\breduce<[^(]*>\([^;]*\);", src), src
    assert re.search(r"__shared__ __align__\(16\) float ws\w*\[128\];", src), src


@module(entry="max_over_nothing", topologies=())
class _EmptyAxisModule:
    """Reductions over an axis of no elements: per kind, and per keepdim."""


    @func
    def max_over_nothing(x: Tensor[(1, 0, 8), "f32"]) -> Tensor[(1, 1, 8), "f32"]:
        return tf.reduce(x, axes=(-2,), keepdim=True, kind="max")

    @func
    def abs_max_over_nothing(x: Tensor[(1, 0, 8), "f32"]) -> Tensor[(1, 1, 8), "f32"]:
        return tf.reduce(x, axes=(-2,), keepdim=True, kind="abs_max")

    @func
    def sum_over_nothing(x: Tensor[(1, 0, 8), "f32"]) -> Tensor[(1, 1, 8), "f32"]:
        return tf.reduce(x, axes=(-2,), keepdim=True, kind="sum")

    @func
    def max_over_nothing_squeezed(x: Tensor[(1, 0, 8), "f32"]) -> Tensor[(1, 8), "f32"]:
        return tf.reduce(x, axes=(-2,), keepdim=False, kind="max")

    @func
    def max_over_a_full_axis(x: Tensor[(1, 0, 8), "f32"]) -> Tensor[(1, 0, 1), "f32"]:
        return tf.reduce(x, axes=(-1,), keepdim=True, kind="max")

    @func
    def max_over_nothing_of_indices(x: Tensor[(1, 0, 8), "i64"]) -> Tensor[(1, 1, 8), "i64"]:
        return tf.reduce(x, axes=(-2,), keepdim=True, kind="max")


@pytest.mark.parametrize(
    "name, shape, identity, dtype",
    [
        ("max_over_nothing", (1, 1, 8), float("-inf"), torch.float32),
        ("abs_max_over_nothing", (1, 1, 8), 0.0, torch.float32),
        ("sum_over_nothing", (1, 1, 8), 0.0, torch.float32),
        ("max_over_nothing_squeezed", (1, 8), float("-inf"), torch.float32),
        ("max_over_nothing_of_indices", (1, 1, 8), -(2**63), torch.int64),
    ],
)
def test_a_reduction_over_no_elements_is_its_identity(name, shape, identity, dtype) -> None:
    """Per kind, per keepdim, and on a dtype that cannot hold -inf."""
    out = evaluate(_EmptyAxisModule.lookup(name), torch.zeros(1, 0, 8, dtype=dtype))

    assert tuple(out.shape) == shape
    assert out.flatten().tolist() == [identity] * (shape[-1] if len(shape) > 1 else 1)


def test_an_empty_axis_that_is_not_the_reduced_one_gets_no_identity() -> None:
    """Only a reduced axis of no elements is a reduction over nothing."""
    out = evaluate(_EmptyAxisModule.lookup("max_over_a_full_axis"), torch.zeros(1, 0, 8))

    assert tuple(out.shape) == (1, 0, 1)
    assert out.numel() == 0
