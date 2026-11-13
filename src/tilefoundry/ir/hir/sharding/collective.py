"""Synchronous collectives over explicitly supplied rank-local payloads."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types

from .communication import check_payload, participants_of

__all__ = ["Collective"]

ROOTED = frozenset({"broadcast", "reduce", "gather", "scatter"})
REDUCING = frozenset({"reduce", "all_reduce", "reduce_scatter"})
GATHERING = frozenset({"gather", "all_gather"})
SCATTERING = frozenset({"scatter", "reduce_scatter"})
KINDS = ROOTED | REDUCING | GATHERING | SCATTERING


@expr.register_op(dialect="tf", category="sharding")
class Collective(expr.CommunicationOp):
    """Communicate local payloads in an isolated, ordered participant mesh.

    See [hir §1.3](docs/spec/hir.md#13-op).
    """

    scope_inputs = ()
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    participants = expr.ParamDef(kind="attribute", annotation=types.Mesh)
    context = expr.ParamDef(kind="attribute", annotation=str)
    kind = expr.ParamDef(kind="attribute", annotation=str)
    root = expr.ParamDef(kind="attribute", annotation=int | None, default=None)
    reduction = expr.ParamDef(kind="attribute", annotation=str | None, default=None)
    axis = expr.ParamDef(kind="attribute", annotation=int, default=0)


def root_mesh(mesh: types.Mesh, root: int) -> types.Mesh:
    """Select the root while retaining the participant mesh's sliced offset."""
    placement = types.Placement.from_mesh(mesh)[root]
    ids = placement.program_ids(mesh.topologies)
    parts = []
    for topology, (shape, _), position in zip(mesh.topologies, types.flattened_levels(mesh), ids, strict=True):
        layout = types.ComposedLayout(None, position, types.Layout((1,) * len(shape)))
        parts.append(types.Mesh((topology,), layout))
    selected = types.Mesh.from_meshes(*parts)
    return replace(selected, names=mesh.names)


@typeinfer.register(Collective)
def infer_collective(call, ctx) -> types.TensorType:
    op = call.target
    source = ctx.type_of(call.args[0])
    members = participants_of(call, ctx)
    if op.kind not in KINDS:
        ctx.error(call, f"unknown collective kind {op.kind!r}")
    if op.kind in ROOTED:
        if not isinstance(op.root, int) or isinstance(op.root, bool) or not 0 <= op.root < len(members):
            ctx.error(call, "root must be a valid communicator-local rank")
    elif op.root is not None:
        ctx.error(call, "root is only valid for a rooted collective")
    if op.kind in REDUCING:
        if op.reduction not in {"sum", "avg", "max"}:
            ctx.error(call, "reduction must be sum, avg or max")
        if op.reduction == "avg" and not isinstance(source.dtype, expr.FloatDType):
            ctx.error(call, "avg requires floating point payloads")
    elif op.reduction is not None:
        ctx.error(call, "reduction is only valid for a reduction collective")
    required = (members[op.root],) if op.kind in {"broadcast", "scatter"} else members
    check_payload(call, ctx, required)
    shape = tuple(source.shape)
    if op.kind in GATHERING | SCATTERING:
        if not isinstance(op.axis, int) or isinstance(op.axis, bool):
            ctx.error(call, "collective axis must be an integer")
        axis = op.axis + len(shape) if op.axis < 0 else op.axis
        if not 0 <= axis < len(shape):
            ctx.error(call, "collective axis is out of range")
        extent = shape[axis]
        if op.kind in SCATTERING and extent % len(members):
            ctx.error(call, "scatter axis must be divisible by participant count")
        extent = extent * len(members) if op.kind in GATHERING else extent // len(members)
        shape = (*shape[:axis], extent, *shape[axis + 1:])
    mesh = root_mesh(op.participants, op.root) if op.kind in {"reduce", "gather"} else op.participants
    attr = types.Varying() if op.kind in SCATTERING else types.Broadcast()
    attrs = tuple(attr for _ in types.flatten(mesh.layout.shape))
    return replace(source, shape=shape, layout=types.ShardLayout(types.Layout(shape), attrs, mesh))


@evaluator.register(Collective)
def eval_collective(ctx):
    """Read declared rank pieces, then publish fresh results at valid recipients."""
    op = ctx.op
    source = ctx.args[0].tensor
    members = types.Placement.from_mesh(op.participants)
    senders = (members[op.root],) if op.kind in {"broadcast", "scatter"} else members
    if source.is_global and op.kind not in {"broadcast", "scatter"}:
        shard = types.shard_layout_of(source.type.layout)
        if shard is None or any(not isinstance(a, types.Broadcast) for a in shard.attrs):
            raise evaluator.EvalError("collective requires actual per-placement payloads")
    pieces = [source.to_local(p) for p in senders]
    expected_dtype = evaluator.to_torch_dtype(source.type.dtype)
    for piece in pieces:
        if tuple(piece.shape) != tuple(source.type.shape) or piece.dtype != expected_dtype:
            raise evaluator.EvalError("collective payload shape or dtype differs from its declaration")
    if op.kind in GATHERING:
        result = utils.torch.cat(pieces, dim=op.axis)
    elif op.kind in REDUCING:
        stacked = utils.torch.stack(pieces)
        if op.reduction == "max":
            result = stacked.amax(dim=0)
        elif op.reduction == "avg":
            result = stacked.mean(dim=0)
        else:
            result = stacked.sum(dim=0, dtype=pieces[0].dtype)
    else:
        result = pieces[0]
    if op.kind in SCATTERING:
        width = result.shape[op.axis] // len(members)
        outputs = tuple((p, result.narrow(op.axis, i * width, width).clone()) for i, p in enumerate(members))
    else:
        recipients = (members[op.root],) if op.kind in {"reduce", "gather"} else members
        outputs = tuple((p, result.clone()) for p in recipients)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, outputs))


@access.register_relation(Collective)
def collective_access(call, ctx):
    """Retain all local source and destination cells for the communication event."""
    return access.iterating((), (access.state(ctx.type_of(call.args[0])), access.state(call.type)))


@cost.register(Collective)
def collective_cost(call, ctx):
    """Require a concrete realization before pricing collective transport."""
    raise NotImplementedError("Collective cost requires an explicit communication realization")
