from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType, reject_low_precision
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain, shape_from_relation
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

from ..math._helpers import resolve_anchor_storage


@register_op
class MatMul(Op):
    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)


def _is_one(dim) -> bool:
    """A static unit (broadcastable) dim."""
    return isinstance(dim, int) and not isinstance(dim, bool) and dim == 1


def _k_split_axes(t, k_tensor_axis: int) -> "frozenset[int]":
    """The mesh axes on which *t* splits its contraction (K) tensor axis."""
    if not isinstance(t.layout, ShardLayout):
        return frozenset()
    sl = t.layout
    la2ta = layout_axis_to_tensor_axis(sl.layout.shape, t.shape)
    return frozenset(
        p
        for p, a in enumerate(sl.attrs)
        if isinstance(a, Split) and la2ta[a.axis] == k_tensor_axis
    )


def _broadcast_batch(lhs_batch, rhs_batch):
    """Right-aligned per-dim broadcast of two batch shapes (ranks may differ —
    the shorter is padded on the left with 1s), or ``None`` when a dim pair is
    neither equal nor broadcastable."""
    n = max(len(lhs_batch), len(rhs_batch))
    lp = (1,) * (n - len(lhs_batch)) + tuple(lhs_batch)
    rp = (1,) * (n - len(rhs_batch)) + tuple(rhs_batch)
    out = []
    for a, b in zip(lp, rp):
        if a == b:
            out.append(a)
        elif _is_one(a):
            out.append(b)
        elif _is_one(b):
            out.append(a)
        else:
            return None
    return tuple(out)


@register_type_relation(MatMul)
def _matmul_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward access relation for ``(batch.., M, K) × (batch.., K, N)``.

    Iteration domain is ``[batch.., M, N, K]``; the output map drops K (the
    reduced contraction dim). A batch dim that this operand broadcasts (its
    extent is 1 while the output extent is larger) accesses a constant 0 rather
    than the iteration dim, so shard propagation treats it as a broadcast.
    """
    lhs, rhs = input_types
    lhs_batch = lhs.shape[:-2]
    rhs_batch = rhs.shape[:-2]
    out_batch = _broadcast_batch(lhs_batch, rhs_batch)
    b = len(out_batch)
    m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
    domain = build_domain((*out_batch, m, n, k))

    m_d, n_d, k_d = b, b + 1, b + 2
    in_dims = [f"d{i}" for i in range(b + 3)]

    def batch_access(in_batch):
        # The operand's map ranges over its own batch dims, right-aligned to the
        # output's: batch dim ``j`` reads iteration dim ``pad + j``, or a
        # constant 0 when that owned dim is size-1 broadcasting to a larger
        # output batch dim. Dims the operand lacks are simply absent from its
        # range (its range rank equals its own tensor rank).
        pad = b - len(in_batch)
        return [
            "0"
            if (_is_one(in_batch[j]) and not _is_one(out_batch[pad + j]))
            else f"d{pad + j}"
            for j in range(len(in_batch))
        ]

    lhs_out = batch_access(lhs_batch) + [f"d{m_d}", f"d{k_d}"]
    rhs_out = batch_access(rhs_batch) + [f"d{k_d}", f"d{n_d}"]
    out_out = [f"d{j}" for j in range(b)] + [f"d{m_d}", f"d{n_d}"]
    src = "[" + ", ".join(in_dims) + "]"
    maps = tuple(
        isl.map(f"{{ {src} -> [{', '.join(dst)}] }}")
        for dst in (lhs_out, rhs_out, out_out)
    )
    return AccessRelationResult(domain=domain, maps=maps)


@register_typeinfer(MatMul)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    reject_low_precision(ctx, call, lhs, rhs)
    if lhs.dtype != rhs.dtype:
        ctx.error(call, f"MatMul dtype mismatch: {lhs.dtype} vs {rhs.dtype}")
    if len(lhs.shape) < 2 or len(rhs.shape) < 2:
        ctx.error(call, "MatMul requires rank >= 2 on both operands")
    if _broadcast_batch(lhs.shape[:-2], rhs.shape[:-2]) is None:
        ctx.error(call, f"MatMul batch-dim mismatch {lhs.shape[:-2]} vs {rhs.shape[:-2]}")
    if lhs.shape[-1] != rhs.shape[-2]:
        ctx.error(
            call,
            f"MatMul contraction-dim mismatch: lhs K={lhs.shape[-1]} vs rhs K={rhs.shape[-2]}",
        )

    # The contraction dim K must be split on the same mesh axes for both
    # operands: a shard of lhs's K contracts against the matching shard of
    # rhs's K. Splitting K on one operand but not the other is inconsistent.
    if _k_split_axes(lhs, len(lhs.shape) - 1) != _k_split_axes(rhs, len(rhs.shape) - 2):
        ctx.error(
            call,
            "MatMul contraction dim K must be split on the same mesh axes for "
            "both operands",
        )

    relation = build_relation(call, (lhs, rhs), ctx)
    # Output shape comes from the relation (domain + output map), not a separate
    # hand-written rule: output axes are [batch.., M, N] (K reduced); the K
    # domain dim and N output axis fall out of the output shape's rank.
    out_shape = shape_from_relation((lhs, rhs), relation)
    k_domain_dim = len(out_shape)
    try:
        shard = derive_output_shard_layout(
            (lhs, rhs),
            relation,
            out_shape,
            partial_reduction_dims=frozenset({k_domain_dim}),
        )
    except ValueError as e:
        ctx.error(call, str(e))
    layout = shard if shard is not None else lhs.layout
    storage = resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    return TensorType(shape=out_shape, dtype=lhs.dtype, layout=layout, storage=storage)


@register_eval(MatMul)
def _eval_matmul(ctx):

    out = torch.matmul(ctx.args[0].data, ctx.args[1].data)
    return TensorValue(data=out, type=ctx.result_type)
