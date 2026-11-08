"""Generic shard propagation over a forward access relation.

``derive_output_shard_layout`` derives an op's output ``ShardLayout`` from the
input shards and the relation access maps by one rule for every op. It reads
only the maps' affine structure (which domain dim each tensor axis uses), never
the domain bounds, so it is size-agnostic and identical for static and dynamic
shapes.
"""

from __future__ import annotations

from dataclasses import replace
from functools import reduce

import isl

from tilefoundry.ir.types import Layout, ShardLayout
from tilefoundry.ir.types.int_tuple import compatible_profile, mode_axes
from tilefoundry.ir.types.mesh import make_mesh, separate
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Partial,
    ShardAttr,
    Split,
    Varying,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageOwners
from tilefoundry.ir.types.utils import attrs_by_axis, participant_layout


def _result_access(ma: isl.multi_aff) -> dict[int, tuple[str, int | None]]:
    """How each result axis of the Op-written affine reads the domain."""
    n_in = ma.dim(isl.dim_type.IN)
    n_out = ma.dim(isl.dim_type.OUT)
    out: dict[int, tuple[str, int | None]] = {}
    for o in range(n_out):
        aff = ma.get_at(o)
        has_div = any(
            int(aff.get_coefficient_val(isl.dim_type.DIV, j).num_si()) != 0
            for j in range(aff.dim(isl.dim_type.DIV))
        )
        has_offset = not aff.get_constant_val().is_zero()
        used = [
            (j, int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()))
            for j in range(n_in)
            if int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()) != 0
        ]
        if has_div:
            out[o] = ("complex", None)
        elif not used:
            out[o] = ("const", None)
        elif has_offset:
            out[o] = ("complex", None)
        elif len(used) == 1 and used[0][1] == 1:
            out[o] = ("proj", used[0][0])
        else:
            out[o] = ("complex", None)
    return out


def involved_dims(access: "isl.multi_aff") -> "set[int]":
    """Every domain dim any result axis of *access* reads.

    Includes dims that appear only inside an access that is not a projection.
    """
    dims: set[int] = set()
    for out_axis in range(access.dim(isl.dim_type.OUT)):
        affine = access.get_at(out_axis)
        for in_dim in range(access.dim(isl.dim_type.IN)):
            if int(affine.get_coefficient_val(isl.dim_type.IN, in_dim).num_si()) != 0:
                dims.add(in_dim)
    return dims


def _carrier_layout(
    input_type,
    input_map,
    output_map,
    output_shape,
    mesh,
    propagated_attrs,
    complete_reduction_dims,
    reduction,
    fresh_strides,
    storage,
    owners,
):
    """Transform a covering input shard layout into the output layout.

    Route positions through domain projections and emit them in output-axis
    order. This carries a physical factorization across, so anything short of an
    exact coordinate projection returns ``None`` and lets synthesis build a
    canonical layout instead.
    See [shard §6](docs/spec/shard.md#6-shardattr) and
    [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    sl = input_type.layout
    layout = sl.layout
    la2ta_in = mode_axes(compatible_profile(layout.shape, input_type.shape))
    in_access = _result_access(input_map)
    out_access = _result_access(output_map)
    dom_to_out = {d: o for o, (k, d) in out_access.items() if k == "proj"}

    pos_dom: list = []
    for p in range(len(layout.shape)):
        kind, ddim = in_access[la2ta_in[p]]
        if kind != "proj":
            return None
        pos_dom.append(ddim)
    if not set(dom_to_out).issubset(set(pos_dom)):
        return None

    out_rank = len(output_shape)
    per_axis: dict[int, list] = {o: [] for o in range(out_rank)}
    for p, ddim in enumerate(pos_dom):
        if ddim in dom_to_out:
            per_axis[dom_to_out[ddim]].append(p)
        elif ddim in complete_reduction_dims:
            if out_rank:
                per_axis[out_rank - 1].append(p)
        else:
            return None

    new_shape: list = []
    new_pos_of: dict[int, int] = {}
    src_pos: list = []
    for o in range(out_rank):
        positions = per_axis[o]
        if not positions:
            new_shape.append(1)
            src_pos.append(None)
            continue
        for p in positions:
            reduced = pos_dom[p] in complete_reduction_dims
            new_pos_of[p] = len(new_shape)
            new_shape.append(1 if reduced else layout.shape[p])
            src_pos.append(None if reduced else p)
    new_strides = [0 if p is None else layout.strides[p] for p in src_pos]

    out_attrs: list = [
        attr if not isinstance(attr, Split) else Broadcast() for attr in propagated_attrs
    ]
    for p_mesh, attr in enumerate(sl.attrs):
        if isinstance(attr, Split) and attr.axis in new_pos_of:
            out_attrs[p_mesh] = (
                Partial(reduction)
                if pos_dom[attr.axis] in complete_reduction_dims
                else Split(new_pos_of[attr.axis])
            )
        elif isinstance(attr, Partial):
            out_attrs[p_mesh] = Partial(attr.reduction)

    try:
        la2ta_out = mode_axes(compatible_profile(tuple(new_shape), tuple(output_shape)))
    except ValueError:
        return None
    mapped: list = list(out_attrs)
    for p_mesh, attr in enumerate(out_attrs):
        if isinstance(attr, Split):
            mapped[p_mesh] = Split(la2ta_out[attr.axis])
        elif isinstance(attr, Partial):
            mapped[p_mesh] = Partial(attr.reduction)
    if mapped != propagated_attrs:
        return None

    if fresh_strides:
        return ShardLayout.from_modes(
            tuple(new_shape), mesh, tuple(out_attrs), storage=storage, owners=owners
        )
    carrier = ShardLayout(
        layout=Layout(
            shape=tuple(new_shape),
            strides=tuple(new_strides),
        ),
        attrs=tuple(out_attrs),
        mesh=mesh,
    )
    return carrier


def derive_output_shard_layout(
    input_types: tuple,
    relations,
    output_shape: tuple,
    *,
    partial_reduction_dims: "frozenset[int]" = frozenset(),
    complete_reduction_dims: "frozenset[int]" = frozenset(),
    reduction: str = "sum",
    fresh_strides: bool = False,
    ctx=None,
    storage=None,
):
    """Derive the output ``ShardLayout`` from the input shards and the Op's relations.

    Every input is projected onto the same participating levels -- each level
    any input distributes -- so mesh axes keep one index across the inputs.
    """
    target = getattr(ctx, "target", None)
    owners = target.storage_owners if target else StorageOwners((), {})
    states = tuple(attrs_by_axis(type_, owners=owners) for type_ in input_types)
    layouts = tuple(
        replace(layout, attrs=tuple(state.values())) if layout is not None else None
        for type_, state in zip(input_types, states, strict=True)
        for layout in (shard_layout_of(type_.layout),)
    )
    frame_levels = {level for state in states for level, _axis in state}
    required = {
        level
        for type_, layout, state in zip(input_types, layouts, states)
        for level in frame_levels
        if any(attr != Broadcast() for (at, _axis), attr in state.items() if at == level)
        or (layout is None and owners.private(type_.storage, level))
    }
    parts, active = {}, set()
    for layout, state in zip(layouts, states, strict=True):
        if layout is None:
            continue
        for part in separate(layout.mesh):
            level = part.topologies[0].name
            nonbroadcast = any(
                attr != Broadcast() for (at, _axis), attr in state.items() if at == level
            )
            if nonbroadcast and level in active and parts[level] != part:
                raise ValueError(
                    "inputs distribute over different meshes; their mesh axes do not align"
                )
            if level not in parts or (nonbroadcast and level not in active):
                parts[level] = part
            if nonbroadcast:
                active.add(level)
    if not parts:
        return None
    order = tuple(dict.fromkeys((*owners.order, *parts)))
    mesh = (
        make_mesh(*(parts[level] for level in order if level in required))
        if required
        else participant_layout(next(layout for layout in layouts if layout is not None)).mesh
    )
    mesh = next(
        (layout.mesh for layout in layouts if layout is not None and layout.mesh == mesh), mesh
    )
    keys = tuple(attrs_by_axis(input_types[0], mesh, owners))
    input_types = tuple(
        type_
        if layout is None
        else replace(
            type_,
            layout=ShardLayout(
                layout.layout, tuple(state.get(key, Broadcast()) for key in keys), mesh
            ),
        )
        for type_, layout, state in zip(input_types, layouts, states, strict=True)
    )
    sharded = [
        (i, t.layout) for i, t in enumerate(input_types) if isinstance(t.layout, ShardLayout)
    ]

    if any(access.affine is None for access in relations):
        raise ValueError("shard propagation requires a single Op-written affine per boundary")
    *input_maps, output_map = (access.affine for access in relations)
    out_access = _result_access(output_map)
    domain_to_out_axis = {
        d: o for o, (kind, d) in out_access.items() if kind == "proj"
    }
    out_all_dims = involved_dims(output_map)

    attrs = []
    states = tuple(attrs_by_axis(type_, mesh, owners) for type_ in input_types)
    for key in keys:
        held = [state[key] for state in states]
        merged = reduce(
            lambda a, b: ShardAttr.combine(None, a, b),
            (a for a in held if not isinstance(a, Partial)),
            Broadcast(),
        )
        fallback = reduce(
            lambda a, b: ShardAttr.combine(None, a, b),
            (a for a in held if not isinstance(a, (Split, Partial))),
            Broadcast(),
        )
        attrs.append(fallback if isinstance(merged, Split) else merged)
    for i, sl in sharded:
        la2ta = mode_axes(compatible_profile(sl.layout.shape, input_types[i].shape))
        in_access = _result_access(input_maps[i])
        for p, attr in enumerate(sl.attrs):
            if isinstance(attr, Partial):
                new_attr: object = Partial(attr.reduction)
                if isinstance(attrs[p], (Split, Partial)) and attrs[p] != new_attr:
                    raise ValueError(
                        f"input {i} mesh axis {p}: incompatible output shard "
                        f"{attrs[p]} vs {new_attr}"
                    )
                attrs[p] = new_attr
                continue
            if not isinstance(attr, Split):
                continue
            kind, ddim = in_access[la2ta[attr.axis]]
            if kind == "const":
                continue
            if kind == "complex":
                raise ValueError(
                    f"input {i} mesh axis {p}: Split on a non-projection access "
                    "is not supported for shard propagation"
                )
            if ddim in complete_reduction_dims or ddim in partial_reduction_dims:
                new_attr = Partial(reduction)
            elif ddim in domain_to_out_axis:
                new_attr = Split(domain_to_out_axis[ddim])
            elif ddim in out_all_dims:
                raise ValueError(
                    f"input {i} mesh axis {p}: domain dim survives only via a "
                    "non-projection output access; cannot derive output layout axis"
                )
            else:
                continue
            if isinstance(attrs[p], (Split, Partial)) and attrs[p] != new_attr:
                raise ValueError(
                    f"input {i} mesh axis {p}: incompatible output shard "
                    f"{attrs[p]} vs {new_attr}"
                )
            attrs[p] = new_attr

    if all(isinstance(attr, Varying) for attr in attrs):
        kept_levels = {level for level, _axis in keys}
        if all((level in kept_levels) == owners.private(storage, level) for level in frame_levels):
            return None

    carriers = [
        layout
        for i, sl in sharded
        if (
            layout := _carrier_layout(
                input_types[i],
                input_maps[i],
                output_map,
                output_shape,
                mesh,
                attrs,
                complete_reduction_dims,
                reduction,
                fresh_strides,
                storage,
                owners,
            )
        )
        is not None
    ]
    if carriers and all(c == carriers[0] for c in carriers):
        return carriers[0]

    return ShardLayout.default(output_shape, mesh, tuple(attrs), storage=storage, owners=owners)


__all__ = ["derive_output_shard_layout"]
