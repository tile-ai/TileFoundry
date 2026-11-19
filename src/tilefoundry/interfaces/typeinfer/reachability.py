"""Whether a scoped Op's inputs are laid out so the scope it runs in can reach them."""

from __future__ import annotations

from tilefoundry.ir import expr, types, utils

__all__ = ["check_input_reachability"]


def check_input_reachability(op, operand_types, mesh):
    if mesh is None:
        return
    names = getattr(op, "scope_inputs", None)
    params = getattr(expr.schema_of(type(op)), "signature", ())
    inputs = tuple(p.name for p in params if p.kind == "input")
    for index, type_ in enumerate(operand_types):
        if names is not None and (index >= len(inputs) or inputs[index] not in names):
            continue
        layout = types.shard_layout_of(getattr(type_, "layout", None))
        if layout is None:
            continue
        layout = types.shard_layout_of(utils.physical_layout(layout))
        if layout is None:
            continue
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
        if not utils.storage_reaches(type_.storage, projected.mesh, mesh):
            raise ValueError(
                f"input {index} is laid out more coarsely and kept in {type_.storage.name.lower()}, which does not reach the units this runs on; reshard it to smem or gmem first"
            )
