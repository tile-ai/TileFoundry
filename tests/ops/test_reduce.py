"""HIR Reduce typeinfer.

The output ``ShardLayout`` of ``Reduce`` collapses every Split that lives on a
reduced tensor axis into ``Broadcast`` and shrinks the matching layout
positions to size 1 with stride 0 (broadcast view); a Split on a non-reduced
axis is preserved. An unsharded input passes through.
"""
from __future__ import annotations

import math
import re

import pytest
import torch

import tilefoundry
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    raw_shard_tensor_type,
    run_typeinfer_case,
    split_local_extents,
)
from tilefoundry import func, module
from tilefoundry.codegen.cuda.module import emit_cuda_module
from tilefoundry.codegen.registry import group_functions_by_target
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.passes.transforms.hir_to_tir import _analyze_cross_warp_workspace

_RMEM = StorageKind.RMEM
_BF = DType.bf16
# Two-axis mesh; the reduce cases reuse it for input and expectation so the
# preserved mesh compares equal.
_M = make_mesh((6, 32), ("w", "t"))

_MEAN_LAST = Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)
_PARTIAL_MESH = make_mesh((4,))
_PSUM = make_shard_tensor_type((8, 16), mesh=_PARTIAL_MESH, attrs=(Partial("sum"),), dtype=DType.f32)
_PMAX = make_shard_tensor_type((8, 16), mesh=_PARTIAL_MESH, attrs=(Partial("max"),), dtype=DType.f32)

CASES = [
    # Unsharded input passes through (no layout).
    TypeInferCase(
        "unsharded_passes_through",
        Reduce(axes=(0,), keepdim=True, kind=ReduceKind.SUM),
        (make_tensor_type((8, 16), DType.f32, storage=_RMEM),),
        make_tensor_type((1, 16), DType.f32, storage=_RMEM),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reduce_typeinfer(case):
    run_typeinfer_case(case)


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
        TypeInferCase(
            "abs_max_over_partial",
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.ABS_MAX),
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
# (broadcast view) while preserving the layout's own rank, so these cases
# check output shape and which mesh axis stays genuinely `Split` vs collapses
# to `Broadcast`, not the internal layout position count a valid `Reduce`
# happens to produce.


def _attr_kinds(ty) -> tuple:
    return tuple(type(a).__name__ for a in ty.layout.attrs)


def test_reduced_axis_splits_become_broadcast():
    """Reduced-axis Splits become Broadcast; layout positions on the reduced
    axis shrink to size 1 / stride 0 (broadcast-view input)."""
    x_ty = raw_shard_tensor_type(
        (1, 1536), (1, 6, 32, 8), (0, 0, 0, 1), (Split(1), Split(2)), _M,
        dtype=_BF, storage=_RMEM,
    )
    ty = infer_call(_MEAN_LAST, x_ty)
    assert tuple(ty.shape) == (1, 1)
    assert _attr_kinds(ty) == ("Broadcast", "Broadcast")


def test_zeroes_reduced_positions_for_global_view():
    """Same, but the input layout carries a global (non-zero) stride view:
    reduced positions are still zeroed."""
    x_ty = make_shard_tensor_type(
        (1, 1536), mesh=_M, attrs=(Split(1), Split(1)), dtype=_BF, storage=_RMEM,
    )
    ty = infer_call(_MEAN_LAST, x_ty)
    assert tuple(ty.shape) == (1, 1)
    assert _attr_kinds(ty) == ("Broadcast", "Broadcast")


def test_preserves_non_reduced_axis_split():
    """A Split on the non-reduced axis is preserved; the reduced axis ->
    Broadcast."""
    x_ty = make_shard_tensor_type(
        (12, 32), mesh=_M, attrs=(Split(0), Split(1)), dtype=_BF, storage=_RMEM,
    )
    ty = infer_call(Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM), x_ty)
    assert tuple(ty.shape) == (12, 1)
    assert _attr_kinds(ty) == ("Split", "Broadcast")
    assert split_local_extents(ty) == [1]


def test_keepdim_false_pops_shape():
    """keepdim=False pops the reduced axis from the shape; the layout still
    broadcasts the reduced positions."""
    x_ty = make_shard_tensor_type(
        (1, 1536), mesh=_M, attrs=(Split(1), Split(1)), dtype=_BF, storage=_RMEM,
    )
    ty = infer_call(Reduce(axes=(1,), keepdim=False, kind=ReduceKind.MEAN), x_ty)
    assert tuple(ty.shape) == (1,)
    assert _attr_kinds(ty) == ("Broadcast", "Broadcast")


def test_implicit_strides_fresh_output():
    """Implicit (None) strides reduce to a fresh, concretely-strided output
    (no None indexing); the non-reduced axis's Split survives, the reduced
    axis becomes Broadcast."""
    x_ty = raw_shard_tensor_type(
        (12, 32), (12, 32), None, (Split(0), Split(1)), _M, dtype=_BF, storage=_RMEM,
    )
    ty = infer_call(Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM), x_ty)
    assert tuple(ty.shape) == (12, 1)
    assert _attr_kinds(ty) == ("Split", "Broadcast")
    assert ty.layout.layout.strides is not None
    assert math.prod(ty.layout.layout.shape) == math.prod(ty.shape)


def test_layout_axis_to_tensor_axis_factorized() -> None:
    # tensor (1, 1536) with layout (1, 6, 32, 8): layout pos 0 -> axis 0; 1/2/3 -> axis 1.
    assert layout_axis_to_tensor_axis((1, 6, 32, 8), (1, 1536)) == [0, 1, 1, 1]


def test_layout_axis_to_tensor_axis_one_to_one() -> None:
    assert layout_axis_to_tensor_axis((16, 32), (16, 32)) == [0, 1]


@pytest.mark.parametrize(
    "op,ref,atol",
    [
        (
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.MEAN),
            lambda x: x.mean(1, keepdim=True), 1e-6,
        ),
        (
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
            lambda x: x.sum(1, keepdim=True), 1e-5,
        ),
        (
            Reduce(axes=(1,), keepdim=False, kind=ReduceKind.ABS_MAX),
            lambda x: x.abs().amax(1), 1e-6,
        ),
        (
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.MAX),
            lambda x: x.amax(1, keepdim=True), 1e-6,
        ),
    ],
    ids=["mean", "sum", "abs_max", "max"],
)
def test_reduce_evaluate(op, ref, atol):
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    run_eval_case(EvalCase("", op, (x,), ref(x), atol=atol))


def test_reduce_max_is_signed_not_abs_max():
    """``ReduceKind.MAX`` is the signed max — distinct from ``ABS_MAX`` when the
    largest-magnitude element is negative."""
    x = torch.tensor([[-5.0, 1.0, 2.0]])
    run_eval_case(
        EvalCase("", Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.MAX),
                 (x,), torch.tensor([[2.0]]), atol=0.0)
    )
    run_eval_case(
        EvalCase("", Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.ABS_MAX),
                 (x,), torch.tensor([[5.0]]), atol=0.0)
    )


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
        topology=[Topology("cta", 2), Topology("thread", 32)],
        layout=Layout(shape=(2, 32), strides=(32, 1)),
        names=("c", "t"),
        topologies=(Topology("cta", 2), Topology("thread", 32)),
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

@module(entry="cross_warp_sum")
class _CrossWarpSumModule:
    @func(topologies=(Topology("thread", 4 * 32),))
    def cross_warp_sum(a: Tensor[(4, 32), 'f32']):
        with Mesh(Topology("thread", 4 * 32), (4, 32), ('tk', 'hc')) as m:
            # Axis 0 (tk) spans the four warps; axis 1 (hc) is the lane axis and
            # carries distinct output cells. Reducing axis 0 crosses warps only.
            a_reg = tf.reshard(a, (4 @ m.tk, 32 @ m.hc), 'rmem')
            s = tf.reduce(a_reg, (0,), True, ReduceKind.SUM)
            return tf.reshard(s, (1, 32 @ m.hc), 'gmem')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cross_warp_sum_matches_torch() -> None:
    rm = tilefoundry.compile(_CrossWarpSumModule, target="cuda")
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
    lowered = tilefoundry.lower(_CrossWarpSumModule, target="cuda")
    src = emit_cuda_module(group_functions_by_target(lowered)["cuda"]).source
    assert re.search(r"\breduce<[^(]*>\([^;]*\);", src), src
    assert re.search(r"__shared__ __align__\(16\) float ws\w*\[128\];", src), src
