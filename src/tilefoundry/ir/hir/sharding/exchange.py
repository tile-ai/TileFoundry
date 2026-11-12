"""Matched variable-count transfers over packed local payloads."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types

from .communication import check_payload, participants_of

__all__ = ["Exchange"]


@expr.register_op(dialect="tf", category="sharding")
class Exchange(expr.CommunicationOp):
    """Transfer packed destination segments into source-ordered receive prefixes.

    See [hir §1.3](docs/spec/hir.md#13-op).
    """

    scope_inputs = ()
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    participants = expr.ParamDef(kind="attribute", annotation=types.Mesh)
    context = expr.ParamDef(kind="attribute", annotation=str)
    counts = expr.ParamDef(kind="attribute", annotation=tuple | None, default=None)
    count_matrix = expr.ParamDef(kind="input", optional=True, default=None)
    receive_capacity = expr.ParamDef(kind="attribute", annotation=int | None, default=None)


@typeinfer.register(Exchange)
def infer_exchange(call, ctx) -> types.TensorType:
    op = call.target
    members = participants_of(call, ctx)
    dynamic = op.counts is None
    if dynamic:
        if len(call.args) != 2:
            ctx.error(call, "dynamic exchange requires a runtime count matrix")
        metadata = ctx.type_of(call.args[1])
        if not isinstance(metadata, types.TensorType) or metadata.shape not in {(len(members), len(members)), (len(members) ** 2,)} or metadata.dtype not in {expr.DType.i32, expr.DType.i64}:
            ctx.error(call, "count matrix must contain N*N int32 or int64 elements")
        if not isinstance(op.receive_capacity, int) or isinstance(op.receive_capacity, bool) or op.receive_capacity < 0:
            ctx.error(call, "dynamic exchange requires a nonnegative receive capacity")
        source = check_payload(call, ctx, members)
        if len(source.shape) != 1:
            ctx.error(call, "exchange requires a flat packed local payload")
        shard = types.shard_layout_of(metadata.layout)
        if shard is not None and (any(not isinstance(attr, types.Broadcast) for attr in shard.attrs)
                                  or any(member not in types.Placement.from_mesh(shard.mesh) for member in members)):
            ctx.error(call, "runtime count matrix must be replicated over all participants")
        attrs = tuple(types.Varying() for _ in types.flatten(op.participants.layout.shape))
        return replace(source, shape=(op.receive_capacity,),
                       layout=types.ShardLayout(types.Layout((op.receive_capacity,)), attrs, op.participants))
    if len(call.args) != 1 or op.receive_capacity is not None:
        ctx.error(call, "static exchange cannot also declare runtime counts or receive capacity")
    if (not isinstance(op.counts, tuple) or len(op.counts) != len(members)
            or any(not isinstance(row, tuple) or len(row) != len(members) for row in op.counts)):
        ctx.error(call, "counts must be a square tuple matrix indexed by source and destination rank")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for row in op.counts for n in row):
        ctx.error(call, "counts must be static nonnegative integers")
    senders = tuple(member for member, row in zip(members, op.counts) if sum(row))
    source = check_payload(call, ctx, senders)
    if len(source.shape) != 1:
        ctx.error(call, "exchange requires a flat packed local payload")
    if any(sum(row) > source.shape[0] for row in op.counts):
        ctx.error(call, "exchange send counts exceed the input capacity")
    shape = (max(map(sum, zip(*op.counts))),)
    attrs = tuple(types.Varying() for _ in types.flatten(op.participants.layout.shape))
    return replace(source, shape=shape, layout=types.ShardLayout(types.Layout(shape), attrs, op.participants))


@evaluator.register(Exchange)
def eval_exchange(ctx):
    """Copy matched segments, retaining zero-filled padding outside receive prefixes."""
    op = ctx.op
    source = ctx.args[0].tensor
    members = types.Placement.from_mesh(op.participants)
    counts = op.counts
    if counts is None:
        metadata = ctx.args[1].tensor
        matrices = (metadata.tensor,) if metadata.is_global else tuple(metadata.to_local(p) for p in members)
        first = matrices[0]
        metadata_dtype = evaluator.to_torch_dtype(metadata.type.dtype)
        if any(tuple(matrix.shape) != tuple(metadata.type.shape) or matrix.dtype != metadata_dtype for matrix in matrices):
            raise evaluator.EvalError("invalid runtime count matrix shape or dtype")
        if any(not utils.torch.equal(first, matrix) for matrix in matrices[1:]):
            raise evaluator.EvalError("runtime count matrices disagree across participants")
        counts = tuple(tuple(row) for row in first.reshape(len(members), len(members)).tolist())
        if any(n < 0 for row in counts for n in row):
            raise evaluator.EvalError("runtime exchange counts must be nonnegative")
        if any(sum(row) > source.type.shape[0] for row in counts):
            raise evaluator.EvalError("runtime send counts exceed input capacity")
        if any(sum(column) > op.receive_capacity for column in zip(*counts)):
            raise evaluator.EvalError("runtime receive counts exceed output capacity")
    if source.is_global and any(sum(row) for row in counts):
        shard = types.shard_layout_of(source.type.layout)
        if shard is None or any(not isinstance(attr, types.Broadcast) for attr in shard.attrs):
            raise evaluator.EvalError("exchange requires actual per-placement payloads")
    dtype = evaluator.to_torch_dtype(source.type.dtype)
    pieces = tuple(source.to_local(p) if op.counts is None or sum(row) else None for p, row in zip(members, counts))
    for piece in pieces:
        if piece is not None and (tuple(piece.shape) != tuple(source.type.shape) or piece.dtype != dtype):
            raise evaluator.EvalError("exchange payload shape or dtype differs from its declaration")
    device = next((piece.device for piece in pieces if piece is not None), None)
    outputs = tuple(utils.torch.zeros(ctx.result_type.shape, dtype=dtype, device=device) for _ in members)
    offsets = [0] * len(members)
    for row, piece in zip(counts, pieces):
        start = 0
        for dst, count in enumerate(row):
            if count:
                outputs[dst][offsets[dst]:offsets[dst] + count].copy_(piece[start:start + count])
            start += count
            offsets[dst] += count
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, tuple(zip(members, outputs))))


@access.register_relation(Exchange)
def exchange_access(call, ctx):
    """Retain source and destination storage until the synchronous transfer completes."""
    return access.iterating((), (*[access.state(ctx.type_of(arg)) for arg in call.args], access.state(call.type)))


@cost.register(Exchange)
def exchange_cost(call, ctx):
    """Require an explicit transport realization before assigning exchange cost."""
    raise NotImplementedError("Exchange cost requires an explicit communication realization")
