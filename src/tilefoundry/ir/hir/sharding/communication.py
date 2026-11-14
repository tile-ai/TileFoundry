"""Shared validation for explicit communication participants and local payloads."""

from __future__ import annotations

from tilefoundry.ir import types

__all__ = ["participants_of", "check_payload"]


def participants_of(call, ctx):
    """Validate the isolated mesh and return its existing placement order."""
    op = call.target
    if not isinstance(op.participants, types.Mesh):
        ctx.error(call, "participants must be a Mesh")
    if not isinstance(op.context, str) or not op.context:
        ctx.error(call, "context must be a nonempty string")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
           for n in types.flatten(op.participants.layout.shape)):
        ctx.error(call, "participants require static positive extents")
    members = types.Placement.from_mesh(op.participants)
    if not members or any(p in members[:i] for i, p in enumerate(members)):
        ctx.error(call, "participants must be nonempty and unique")
    types.check_topology(op.participants)
    for member in members:
        for topology in op.participants.topologies:
            if isinstance(topology, types.Topology) and isinstance(topology.size, int):
                if not 0 <= member.ids[topology.name] < topology.size:
                    ctx.error(call, "participant position is outside its topology")
    if ctx.current_mesh is not None:
        available = types.Placement.from_mesh(ctx.current_mesh)
        if any(member not in available for member in members):
            ctx.error(call, "participants are outside the current mesh scope")
    return members


def check_payload(call, ctx, required):
    """Require statically sized local payloads at each sending placement."""
    source = ctx.type_of(call.args[0])
    shard = types.shard_layout_of(source.layout)
    if shard is not None:
        if any(isinstance(attr, (types.Split, types.Partial)) for attr in shard.attrs):
            ctx.error(call, "communication payload must be rank-local; apply Local to Split/Partial inputs")
        held = types.Placement.from_mesh(shard.mesh)
        if any(p not in held for p in required):
            ctx.error(call, "input participant domain does not contain every source rank")
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in source.shape):
        ctx.error(call, "communication payload extents must be static nonnegative integers")
    return source
