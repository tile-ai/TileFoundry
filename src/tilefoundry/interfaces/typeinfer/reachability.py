"""Whether a scoped Op's inputs are laid out so the scope it runs in can reach them."""

from __future__ import annotations

from tilefoundry.ir import expr, types, utils

__all__ = ["check_input_reachability"]


def check_input_reachability(op, operand_types, mesh, ctx=None):
    """Raise when a scoped Op's input is laid out more finely or more coarsely than *mesh*.

    A topology that holds only Broadcast or Varying limits no reader when *mesh*
    does not name it ([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics):
    "An explicit execution mesh may omit outside topologies that contain neither
    Split nor Partial"). The physical projection says where the addresses are;
    which units hold a copy is wider: with *ctx*'s target, the innermost topology of
    *mesh* holds the value when the value is Varying there, each of its units keeps
    its own copy of the storage, and every executing unit is one that holds a copy.
    """
    owners = None if ctx is None or ctx.target is None else ctx.target.storage_owners
    if mesh is None:
        return
    scope = set(utils.topology_names(mesh))
    names = getattr(op, "scope_inputs", None)
    params = getattr(expr.schema_of(type(op)), "signature", ())
    inputs = tuple(p.name for p in params if p.kind == "input")
    for index, type_ in enumerate(operand_types):
        if names is not None and (index >= len(inputs) or inputs[index] not in names):
            continue
        layout = types.shard_layout_of(getattr(type_, "layout", None))
        if layout is None:
            continue
        inner = mesh.topologies[-1].name
        holds = any(
            owners is not None
            and attr == types.Varying()
            and topology == inner
            and owners.private(type_.storage, topology)
            for attr, (topology, _) in zip(layout.attrs, utils.axis_keys(layout.mesh))
        ) and utils.units_within(utils.topology_mesh(mesh, inner), utils.topology_mesh(layout.mesh, inner))
        layout = types.shard_layout_of(utils.physical_layout(layout))
        if layout is None:
            continue
        kept = tuple(
            name
            for name, attrs in utils.attrs_by_topology(layout.mesh, layout.attrs).items()
            if any(isinstance(attr, (types.Split, types.Partial)) for attr in attrs) or name in scope
        )
        if not kept:
            continue
        layout = utils.filter_topologies(layout, kept)
        projected = utils.filter_topologies(layout)
        held = utils.topology_names(projected.mesh)
        replicated = (
            all(isinstance(attr, types.Broadcast) for attr in layout.attrs)
            and set(held) <= set(utils.topology_names(mesh))
            and utils.within_mesh_scope(mesh, projected.mesh, held)
        )
        if not replicated and not utils.equals_mesh_scope(projected.mesh, mesh):
            raise ValueError(
                f"input {index} is laid out more finely than the scope it runs in; write it inside that scope, or reshard it back first"
            )
        if not holds and not utils.storage_reaches(type_.storage, projected.mesh, mesh):
            raise ValueError(
                f"input {index} is laid out more coarsely and kept in {type_.storage.name.lower()}, which does not reach the units this runs on; reshard it to smem or gmem first"
            )
