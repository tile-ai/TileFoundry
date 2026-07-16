"""Qwen3-MoE attention main-path SSA HIR graph with 2-CTA cluster head-norm.

Selected distribution plan. One plan = one SSA Function = one DOT graph.

Per-value meshes: each value's ShardLayout.mesh is sized to its logical shape,
so Split(axis) always satisfies shape[axis] % mesh_extent == 0.

Global reference: 128 CTA, 256 thread/CTA (8 warp × 32 lane).

Main chain with per-op meshes:
  hidden (1,2048)          mesh=(1,1,8,32)     B,B,B,B
  input_rmsnorm (1,2048)   same                B,B,B,B
  q_proj (1,4096)          mesh=(1,128,8,32)   B,S(1),P,P   → 128 CTA split N
  reshape_q (32,128)       mesh=(32,2,8,32)    S(0),S(1),B,B
  q_norm (32,128)          mesh=(32,2,8,32)    S(0),S(1),P,P → 2 CTA/cluster on D
  k_proj (1,512)           mesh=(1,128,8,32)   B,S(1),P,P
  reshape_k (4,128)        mesh=(4,2,8,32)     S(0),S(1),B,B
  k_norm (4,128)           mesh=(4,2,8,32)     S(0),S(1),P,P
  v_proj (1,512)           mesh=(1,128,8,32)   B,S(1),P,P
  ordinary attention (32,1,128)                  complete q/k/v values
  reshard_attn (1,4096)    mesh=(1,128,8,32)   B,S(1),B,B   → onto proj mesh
  o_proj (1,2048)          mesh=(1,128,8,32)   B,P(),B,B   → K split on cta, partial sum
  all_reduce (1,2048)      mesh=(1,128,8,32)   B,B,B,B      → boxing
  residual_add (1,2048)    mesh=(1,128,8,32)   B,B,B,B
"""

from __future__ import annotations

from typing import ClassVar

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose


def Add() -> Binary:
    """Local helper constructing ``Binary(kind=ADD)`` — there is no
    per-name ``Add`` Op class; kinded math ops are built via helpers."""
    return Binary(kind=BinaryKind.ADD)
from tilefoundry.ir.core import Op, register_typeinfer  # noqa: E402
from tilefoundry.ir.core.param_def import ParamDef  # noqa: E402
from tilefoundry.ir.core.pattern import Tensor  # noqa: E402
from tilefoundry.ir.hir.sharding.reshard import Reshard  # noqa: E402
from tilefoundry.ir.target.storage import StorageKind  # noqa: E402
from tilefoundry.ir.types import DType, TensorType  # noqa: E402
from tilefoundry.ir.types.shard import (  # noqa: E402
    B,
    Layout,
    Mesh,
    P,
    S,
    ShardLayout,
    Topology,
)

# ── Model constants ──────────────────────────────────────────────────

HIDDEN = 2048
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
Q_PROJ_DIM = NUM_Q_HEADS * HEAD_DIM   # 4096
KV_PROJ_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
DTYPE = DType.bf16
EPS = 1e-6

# ── Per-value mesh builder ───────────────────────────────────────────


def _make_mesh(shape: tuple[int, ...]) -> Mesh:
    """Build a 4D mesh with given (cluster, cta, warp, lane) shape."""
    strides = (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)
    total = shape[0] * shape[1] * shape[2] * shape[3]
    return Mesh(
        topology=Topology("gpu", total),
        layout=Layout(shape=shape, strides=strides),
        names=("cluster", "cta", "warp", "lane"),
    )


def _corder_strides(shape):
    s = [1]
    for d in reversed(shape[1:]):
        s.insert(0, s[0] * d)
    return tuple(s)


def _sl(cluster_attr, cta_attr, warp_attr, lane_attr, logical_shape, mesh):
    """ShardLayout with per-value mesh."""
    return ShardLayout(
        layout=Layout(shape=logical_shape, strides=_corder_strides(logical_shape)),
        attrs=(cluster_attr, cta_attr, warp_attr, lane_attr),
        mesh=mesh,
    )


def _sharded(logical_shape, cluster_attr, cta_attr, warp_attr, lane_attr,
             mesh, dtype=DTYPE, storage=StorageKind.GMEM):
    return TensorType(
        shape=logical_shape, dtype=dtype,
        layout=_sl(cluster_attr, cta_attr, warp_attr, lane_attr,
                    logical_shape, mesh),
        storage=storage,
    )


# ── Local HIR sharding helper ────────────────────────────────────────


class AllReduce(Op):
    name: ClassVar[str] = "all_reduce"
    category: ClassVar[str] = "sharding"
    x = ParamDef(kind="input", pattern=Tensor)
    dst_layout = ParamDef(kind="attribute", annotation=object)


@register_typeinfer(AllReduce)
def _infer_allreduce(call, ctx):
    src_ty = ctx.type_of(call.args[0])
    return TensorType(shape=src_ty.shape, dtype=src_ty.dtype,
                      layout=call.target.dst_layout, storage=src_ty.storage)


# ── Shorthands ───────────────────────────────────────────────────────

SP = P("sum")
B4 = (B(), B(), B(), B())

# Common mesh shapes
_M_1x128 = _make_mesh((1, 128, 8, 32))    # flat 128-CTA
_M_32x2  = _make_mesh((32, 2, 8, 32))      # 32 heads × 2 CTA
_M_4x2   = _make_mesh((4, 2, 8, 32))       # 4 KV heads × 2 CTA
_M_1x1   = _make_mesh((1, 1, 8, 32))       # single-CTA (replicated)


def build_qwen3_attention_main_2cta_headnorm():
    """Build the selected distribution plan.

    Returns:
        Function named ``qwen3_attention_main_2cta_headnorm``.
    """
    # ── Params (use minimal meshes for replicated weights) ──────────
    hidden_ty   = _sharded((1, HIDDEN), *B4, mesh=_M_1x1)
    q_weight_ty = _sharded((Q_PROJ_DIM, HIDDEN), B(), S(0), B(), B(), mesh=_M_1x128)
    k_weight_ty = _sharded((KV_PROJ_DIM, HIDDEN), B(), S(0), B(), B(), mesh=_M_1x128)
    v_weight_ty = _sharded((KV_PROJ_DIM, HIDDEN), B(), S(0), B(), B(), mesh=_M_1x128)
    o_weight_ty = _sharded((HIDDEN, Q_PROJ_DIM), B(), S(1), B(), B(), mesh=_M_1x128)
    rms_w_ty    = _sharded((HIDDEN,), *B4, mesh=_M_1x1)
    q_norm_w_ty = _sharded((HEAD_DIM,), *B4, mesh=_M_32x2)
    k_norm_w_ty = _sharded((HEAD_DIM,), *B4, mesh=_M_4x2)
    kv_head_map_ty = _sharded(
        (NUM_Q_HEADS,), *B4, mesh=_M_1x1, dtype=DType.i64
    )

    hidden   = Var(type=hidden_ty, name="hidden")
    q_weight = Var(type=q_weight_ty, name="q_weight")
    k_weight = Var(type=k_weight_ty, name="k_weight")
    v_weight = Var(type=v_weight_ty, name="v_weight")
    o_weight = Var(type=o_weight_ty, name="o_weight")
    rms_w    = Var(type=rms_w_ty, name="rms_weight")
    q_norm_w = Var(type=q_norm_w_ty, name="q_norm_weight")
    k_norm_w = Var(type=k_norm_w_ty, name="k_norm_weight")
    kv_head_map = Var(type=kv_head_map_ty, name="kv_head_map")

    # ── input_rmsnorm: B,B,B,B ─────────────────────────────────────
    rms_out = Call(
        type=_sharded((1, HIDDEN), *B4, mesh=_M_1x1),
        target=RMSNorm(eps=EPS), args=(hidden, rms_w),
        loc="input_rmsnorm",
    )

    # ── q_proj: matmul(rms_out, transpose(q_weight)) ───────────────
    q_weight_t = Call(
        type=_sharded((HIDDEN, Q_PROJ_DIM), B(), S(1), B(), B(), mesh=_M_1x128),
        target=Transpose(perm=(1, 0)), args=(q_weight,),
        loc="transpose_q_weight",
    )
    q = Call(
        type=_sharded((1, Q_PROJ_DIM), B(), S(1), SP, SP, mesh=_M_1x128),
        target=MatMul(), args=(rms_out, q_weight_t),
        loc="q_proj",
    )
    # ── reshape (1,4096) → (32,128) then reshard to mesh (32,2) ─
    # The CTA-split projection layout cannot survive the reshape onto the head
    # layout (the split would straddle the new head / head-dim axes), so gather
    # it to a replicated layout first; the reshape then carries no sharding and
    # the head-parallel layout is established by the reshard that follows.
    q_replicated = Call(
        type=_sharded((1, Q_PROJ_DIM), *B4, mesh=_M_1x128),
        target=Reshard(layout=_sl(*B4, (1, Q_PROJ_DIM), _M_1x128)),
        args=(q,), loc="gather_q_for_reshape",
    )
    q_reshaped_logical = Call(
        type=TensorType(shape=(NUM_Q_HEADS, HEAD_DIM), dtype=DType.f32,
                        layout=None, storage=StorageKind.GMEM),
        target=Reshape(new_shape=(NUM_Q_HEADS, HEAD_DIM)),
        args=(q_replicated,), loc="reshape_q_to_heads",
    )
    q_reshaped = Call(
        type=_sharded((NUM_Q_HEADS, HEAD_DIM),
                       S(0), S(1), B(), B(), mesh=_M_32x2),
        target=Reshard(layout=_sl(S(0), S(1), B(), B(), (NUM_Q_HEADS, HEAD_DIM), _M_32x2)),
        args=(q_reshaped_logical,), loc="reshard_q_to_heads",
    )
    # ── q_norm: cluster:S(0) splits 32 heads, cta:S(1) splits D=128 ─
    # 32 clusters × 2 CTA = 64 CTAs used; warp/lane partial for reduction
    q_normed = Call(
        type=_sharded((NUM_Q_HEADS, HEAD_DIM),
                       S(0), S(1), SP, SP, mesh=_M_32x2),
        target=RMSNorm(eps=EPS), args=(q_reshaped, q_norm_w),
        loc="q_norm",
    )

    # ── k_proj: matmul(rms_out, transpose(k_weight)) ───────────────
    k_weight_t = Call(
        type=_sharded((HIDDEN, KV_PROJ_DIM), B(), S(1), B(), B(), mesh=_M_1x128),
        target=Transpose(perm=(1, 0)), args=(k_weight,),
        loc="transpose_k_weight",
    )
    k = Call(
        type=_sharded((1, KV_PROJ_DIM), B(), S(1), SP, SP, mesh=_M_1x128),
        target=MatMul(), args=(rms_out, k_weight_t),
        loc="k_proj",
    )
    # ── reshape (1,512) → (4,128) then reshard to mesh (4,2) ────
    # Gather the CTA-split projection to a replicated layout before the reshape
    # (see q above); the reshape carries no sharding and the head-parallel
    # layout is set by the following reshard.
    k_replicated = Call(
        type=_sharded((1, KV_PROJ_DIM), *B4, mesh=_M_1x128),
        target=Reshard(layout=_sl(*B4, (1, KV_PROJ_DIM), _M_1x128)),
        args=(k,), loc="gather_k_for_reshape",
    )
    k_reshaped_logical = Call(
        type=TensorType(shape=(NUM_KV_HEADS, HEAD_DIM), dtype=DType.f32,
                        layout=None, storage=StorageKind.GMEM),
        target=Reshape(new_shape=(NUM_KV_HEADS, HEAD_DIM)),
        args=(k_replicated,), loc="reshape_k_to_heads",
    )
    k_reshaped = Call(
        type=_sharded((NUM_KV_HEADS, HEAD_DIM),
                       S(0), S(1), B(), B(), mesh=_M_4x2),
        target=Reshard(layout=_sl(S(0), S(1), B(), B(), (NUM_KV_HEADS, HEAD_DIM), _M_4x2)),
        args=(k_reshaped_logical,), loc="reshard_k_to_heads",
    )
    # ── k_norm: cluster:S(0) splits 4 heads, cta:S(1) splits D=128 ─
    k_normed = Call(
        type=_sharded((NUM_KV_HEADS, HEAD_DIM),
                       S(0), S(1), SP, SP, mesh=_M_4x2),
        target=RMSNorm(eps=EPS), args=(k_reshaped, k_norm_w),
        loc="k_norm",
    )

    # ── v_proj: matmul(rms_out, transpose(v_weight)) ───────────────
    v_weight_t = Call(
        type=_sharded((HIDDEN, KV_PROJ_DIM), B(), S(1), B(), B(), mesh=_M_1x128),
        target=Transpose(perm=(1, 0)), args=(v_weight,),
        loc="transpose_v_weight",
    )
    v = Call(
        type=_sharded((1, KV_PROJ_DIM), B(), S(1), SP, SP, mesh=_M_1x128),
        target=MatMul(), args=(rms_out, v_weight_t),
        loc="v_proj",
    )

    # ── ordinary attention ─────────────────────────────────────────
    # Projection and head-normalization reductions are completed before the
    # nonlinear score/probability path. The attention inputs are then plain
    # values, so the ordinary HIR ops express the model computation directly.
    q_complete = Call(
        type=_sharded((NUM_Q_HEADS, HEAD_DIM), *B4, mesh=_M_32x2),
        target=Reshard(
            layout=_sl(*B4, (NUM_Q_HEADS, HEAD_DIM), _M_32x2)
        ),
        args=(q_normed,), loc="reshard_q_complete",
    )
    q_attn = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, 1, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Reshape(new_shape=(NUM_Q_HEADS, 1, HEAD_DIM)),
        args=(q_complete,), loc="reshape_q_for_attention",
    )

    k_complete = Call(
        type=_sharded((NUM_KV_HEADS, HEAD_DIM), *B4, mesh=_M_4x2),
        target=Reshard(
            layout=_sl(*B4, (NUM_KV_HEADS, HEAD_DIM), _M_4x2)
        ),
        args=(k_normed,), loc="reshard_k_complete",
    )
    k_heads = Call(
        type=TensorType(
            shape=(NUM_KV_HEADS, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Reshape(new_shape=(NUM_KV_HEADS, HEAD_DIM)),
        args=(k_complete,), loc="reshape_k_for_attention",
    )
    k_gathered = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Gather(axis=0), args=(k_heads, kv_head_map),
        loc="gather_kv_heads_k",
    )
    k_attn = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, 1, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Reshape(new_shape=(NUM_Q_HEADS, 1, HEAD_DIM)),
        args=(k_gathered,), loc="reshape_k_for_scores",
    )

    v_complete = Call(
        type=_sharded((1, KV_PROJ_DIM), *B4, mesh=_M_1x128),
        target=Reshard(
            layout=_sl(*B4, (1, KV_PROJ_DIM), _M_1x128)
        ),
        args=(v,), loc="reshard_v_complete",
    )
    v_heads = Call(
        type=TensorType(
            shape=(NUM_KV_HEADS, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Reshape(new_shape=(NUM_KV_HEADS, HEAD_DIM)),
        args=(v_complete,), loc="reshape_v_for_attention",
    )
    v_gathered = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Gather(axis=0), args=(v_heads, kv_head_map),
        loc="gather_kv_heads_v",
    )
    v_attn = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, 1, HEAD_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Reshape(new_shape=(NUM_Q_HEADS, 1, HEAD_DIM)),
        args=(v_gathered,), loc="reshape_v_for_attention",
    )
    k_transposed = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, HEAD_DIM, 1),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Transpose(perm=(0, 2, 1)), args=(k_attn,),
        loc="transpose_k_for_scores",
    )
    scores = Call(
        type=TensorType(
            shape=(NUM_Q_HEADS, 1, 1),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=MatMul(), args=(q_attn, k_transposed), loc="attention_scores",
    )
    probs = Call(
        type=scores.type,
        target=SoftMax(axis=-1), args=(scores,), loc="attention_probs",
    )
    out_heads = Call(
        type=v_attn.type,
        target=MatMul(), args=(probs, v_attn), loc="attention_out",
    )
    attn = Call(
        type=TensorType(
            shape=(1, Q_PROJ_DIM),
            dtype=DTYPE,
            layout=None,
            storage=StorageKind.GMEM,
        ),
        target=Reshape(new_shape=(1, Q_PROJ_DIM)),
        args=(out_heads,), loc="reshape_attention_out",
    )

    # ── o_proj: matmul(reshard(attn), transpose(o_weight)) → P(sum) ─
    # Reshard the complete attention result onto the flat projection mesh so
    # o_proj's contraction dim K is split on the same CTA mesh axis as its
    # weight.
    attn_proj = Call(
        type=_sharded((1, Q_PROJ_DIM), B(), S(1), B(), B(), mesh=_M_1x128),
        target=Reshard(layout=_sl(B(), S(1), B(), B(), (1, Q_PROJ_DIM), _M_1x128)),
        args=(attn,), loc="reshard_attn_to_proj",
    )
    o_weight_t = Call(
        type=_sharded((Q_PROJ_DIM, HIDDEN), B(), S(0), B(), B(), mesh=_M_1x128),
        target=Transpose(perm=(1, 0)), args=(o_weight,),
        loc="transpose_o_weight",
    )
    o_partial = Call(
        type=_sharded((1, HIDDEN), B(), SP, B(), B(), mesh=_M_1x128),
        target=MatMul(), args=(attn_proj, o_weight_t),
        loc="o_proj",
    )

    # ── all_reduce boxing: P(sum) → B ──────────────────────────
    o_b_ty = _sharded((1, HIDDEN), *B4, mesh=_M_1x128)
    o_b = Call(
        type=o_b_ty, target=AllReduce(dst_layout=o_b_ty.layout),
        args=(o_partial,), loc="all_reduce",
    )

    # ── residual_add ──────────────────────────────────────────────
    residual = Call(
        type=_sharded((1, HIDDEN), *B4, mesh=_M_1x128),
        target=Add(), args=(hidden, o_b),
        loc="residual_add",
    )

    return Function.build(
        name="qwen3_attention_main_2cta_headnorm",
        params=(hidden, q_weight, k_weight, v_weight, o_weight,
                rms_w, q_norm_w, k_norm_w, kv_head_map),
        body=residual,
        return_type=residual.type,
    )


# AllReduce lives outside the ``tilefoundry.ir.hir.*`` auto-import tree, so
# we route it through ``@register_op`` explicitly using the OpSchema path.
from tilefoundry.ir.core.register import register_op  # noqa: E402

register_op(dialect="tf", category="sharding", name="all_reduce")(AllReduce)
