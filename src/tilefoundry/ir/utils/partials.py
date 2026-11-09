"""Partial and Varying state checks for operand layouts, raised as ``ValueError``.

Type inference turns the error into a diagnostic on the call.
"""

from __future__ import annotations

from tilefoundry.ir import expr, types

__all__ = [
    "attrs_by_axis",
    "check_multilinear_partials",
    "reject_partials",
    "require_matching_partial_state",
    "require_uniform_partial_slices",
]


def attrs_by_axis(
    type_: types.TensorType,
    mesh: types.Mesh | None = None,
    owners: expr.StorageOwners | None = None,
) -> dict[tuple[str, int], types.ShardAttr]:
    """Read mesh-axis states keyed by level and ordinal.

    See [shard §6.1](docs/spec/shard.md#61-placement-semantics).

    Args:
        type_: Tensor whose plain layout reads as Varying.
        mesh: Frame to read; defaults to the tensor's shard mesh.
        owners: Target facts used to normalize shared copies. None reads attrs
            as written; empty facts instead normalize Broadcast and Varying alike.

    Returns:
        States in frame order; omitted shard axes and levels read as Broadcast.
    """
    shard = types.shard_layout_of(type_.layout)
    stated, counts = {}, {}
    if shard is not None:
        for axis, (level, _) in enumerate(types.axis_keys(shard.mesh)):
            ordinal = counts.get(level, 0)
            counts[level] = ordinal + 1
            stated[level, ordinal] = (
                shard.attrs[axis] if axis < len(shard.attrs) else types.Broadcast()
            )
    keys = stated
    if mesh is not None:
        keys, counts = {}, {}
        for level, _ in types.axis_keys(mesh):
            ordinal = counts.get(level, 0)
            counts[level] = ordinal + 1
            keys[level, ordinal] = None
    result = {}
    for key in keys:
        attr = stated.get(key, types.Broadcast()) if shard is not None else types.Varying()
        if (
            owners is not None
            and isinstance(attr, (types.Broadcast, types.Varying))
            and not owners.private(type_.storage, key[0])
        ):
            attr = types.Broadcast()
        result[key] = attr
    return result


def reject_partials(
    name: str, layout: types.LayoutBase | None, commutes_with: frozenset[str] = frozenset()
) -> None:
    """Reject Partial states that do not commute with this operation.

    See [shard §6.1](docs/spec/shard.md#61-placement-semantics).

    Args:
        name: Operand name used in the diagnostic.
        layout: Operand's distribution and address layout.
        commutes_with: Partial reduction kinds the operation preserves.

    Raises:
        ValueError: An axis carries a noncommuting Partial state.
    """
    for axis, reduction in enumerate(types.partial_reductions_by_axis(layout)):
        if reduction is None or reduction in commutes_with:
            continue
        raise ValueError(
            f"{name} carries Partial({reduction}) on mesh axis {axis}, "
            f"which does not commute; insert reshard({name}, @ {{B()}}) before this consumer"
        )


def check_multilinear_partials(
    operands: tuple[tuple[str, types.TensorType], ...],
    target,
    *,
    kind: expr.BinaryKind = expr.BinaryKind.MUL,
    anchor: str | None = None,
) -> None:
    """Check Partial combinations using each operand's precomputed states.

    See [shard §6.1](docs/spec/shard.md#61-placement-semantics).
    Plain operands merge states from existing meshes before checking keys;
    the state at a shared key depends only on storage and topology level.

    Args:
        operands: Operand names and types in combination order.
        target: Target holding storage owner facts, or None.
        kind: Operation's combination kind.
        anchor: Only this operand may carry Partial when specified.

    Raises:
        ValueError: Partial states cannot be preserved by the operation.
    """
    owners = target.storage_owners if target else expr.StorageOwners((), {})
    layouts = tuple(types.shard_layout_of(type_.layout) for _, type_ in operands)
    meshes = []
    for layout in layouts:
        if layout is not None and layout.mesh not in meshes:
            meshes.append(layout.mesh)
    states = tuple(
        attrs_by_axis(type_, owners=owners)
        if layout is not None
        else {
            key: attr for mesh in meshes for key, attr in attrs_by_axis(type_, mesh, owners).items()
        }
        for (_, type_), layout in zip(operands, layouts)
    )
    for (name, _), source in zip(operands, states):
        for axis, (key, attr) in enumerate(source.items()):
            if not isinstance(attr, types.Partial):
                continue
            if anchor is not None and name != anchor:
                raise ValueError(
                    f"{name} carries {attr!r} on mesh axis {axis}; "
                    f"reshard({name}, @ {{B()}}) before this consumer"
                )
            state = None
            for (other_name, _), other in zip(operands, states):
                held = other.get(key, types.Broadcast())
                try:
                    state = held if state is None else types.combine_attrs(kind, state, held)
                except ValueError as error:
                    raise ValueError(
                        f"{name} carries Partial({attr.reduction}) on mesh axis {axis}; "
                        f"operand {other_name}: operands do not commute: {error}"
                    ) from error


def require_matching_partial_state(
    dst: types.TensorType, update: types.TensorType, dst_name: str, update_name: str
) -> None:
    """Require identical Partial state for an in-place-style write.

    See [shard §8](docs/spec/shard.md#8-layout-propagation).

    Args:
        dst: Destination tensor type.
        update: Tensor type being inserted into the destination.
        dst_name: Destination name used in the diagnostic.
        update_name: Update name used in the diagnostic.

    Raises:
        ValueError: Destination and update carry different Partial states.
    """
    dst_partials = [
        (axis, reduction)
        for axis, reduction in enumerate(types.partial_reductions_by_axis(dst.layout))
        if reduction is not None
    ]
    update_partials = [
        (axis, reduction)
        for axis, reduction in enumerate(types.partial_reductions_by_axis(update.layout))
        if reduction is not None
    ]
    dst_layout = types.shard_layout_of(dst.layout)
    update_layout = types.shard_layout_of(update.layout)
    if dst_partials:
        if not (
            dst_layout is not None
            and update_layout is not None
            and update_layout.mesh == dst_layout.mesh
            and update_layout.attrs == dst_layout.attrs
        ):
            axis, reduction = dst_partials[0]
            raise ValueError(
                f"{dst_name} carries a Partial({reduction}) on mesh axis {axis}; "
                f"{update_name} must carry the identical per-mesh-axis state. "
                f"Insert reshard({update_name}, @ {{B()}}) or match {dst_name} before this consumer"
            )
    elif update_partials:
        axis, reduction = update_partials[0]
        raise ValueError(
            f"{update_name} carries Partial({reduction}) on mesh axis {axis}, "
            f"but {dst_name} is complete; insert reshard({update_name}, @ {{B()}}) before this consumer"
        )


def require_uniform_partial_slices(
    inputs: tuple[types.TensorType, ...], output: types.ShardLayout, op_name: str
) -> None:
    """Require every input slice to carry the output's Partial state.

    See [shard §8](docs/spec/shard.md#8-layout-propagation).

    Args:
        inputs: Input tensor types contributing slices.
        output: Distribution derived for their assembled result.
        op_name: Consumer name used in the diagnostic.

    Raises:
        ValueError: An input lacks the Partial state required by the output.
    """
    for axis, output_attr in enumerate(output.attrs):
        if not isinstance(output_attr, types.Partial):
            continue
        for index, type_ in enumerate(inputs):
            layout = types.shard_layout_of(type_.layout)
            input_attr = (
                layout.attrs[axis]
                if layout is not None and layout.mesh == output.mesh and axis < len(layout.attrs)
                else None
            )
            if input_attr != output_attr:
                raise ValueError(
                    f"input {index} does not carry {output_attr!r} on mesh axis "
                    f"{axis}; every input must carry the same Partial state: "
                    f"reshard the Partial inputs with @ {{B()}} before {op_name}"
                )
