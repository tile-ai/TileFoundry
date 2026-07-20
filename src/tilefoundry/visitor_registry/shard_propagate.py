"""Generic shard propagation over a forward access relation.

``derive_output_shard_layout`` derives an op's output ``ShardLayout`` from the
input shards and the relation access maps by one rule for every op. It reads
only the maps' affine structure (which domain dim each tensor axis uses), never
the domain bounds, so it is size-agnostic and identical for static and dynamic
shapes.
"""
from __future__ import annotations

import isl

from tilefoundry.ir.types.shard import (
    Layout,
    ShardLayout,
    canonical_shard_layout,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    Split,
    layout_axis_to_tensor_axis,
)


def partial_reductions_by_axis(
    layout: object,
) -> tuple[str | None, ...]:
    """Return each mesh axis's carried Partial reduction, if any.

    The tuple index is the mesh-axis index in ``ShardLayout.attrs``. A
    ``None`` entry denotes an attr that is not a ``Partial``; a non-sharded
    layout returns an empty tuple.
    """
    if not isinstance(layout, ShardLayout):
        return ()
    return tuple(
        attr.reduction if isinstance(attr, Partial) else None
        for attr in layout.attrs
    )


def _result_access(m: "isl.map") -> dict[int, "tuple[str, int | None]"]:
    """Classify each result (out) axis of *m* by how it accesses the domain:

    - ``("proj", d)`` — a pure projection of domain dim ``d`` (single in-dim,
      unit coefficient): the access tracks that domain dim.
    - ``("const", None)`` — no domain dim involved: a constant (broadcast)
      access.
    - ``("complex", None)`` — multiple in-dims, a non-unit coefficient, or
      otherwise not a pure projection: not supported for shard propagation.
    """
    ma = m.as_pw_multi_aff().as_multi_aff()
    n_in = ma.dim(isl.dim_type.IN)
    n_out = ma.dim(isl.dim_type.OUT)
    out: dict[int, tuple[str, int | None]] = {}
    for o in range(n_out):
        aff = ma.get_at(o)
        used = [
            (j, int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()))
            for j in range(n_in)
            if int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()) != 0
        ]
        if not used:
            out[o] = ("const", None)
        elif len(used) == 1 and used[0][1] == 1:
            out[o] = ("proj", used[0][0])
        else:
            out[o] = ("complex", None)
    return out


def _involved_domain_dims(m: "isl.map") -> "set[int]":
    """All domain (in) dims referenced by any result axis of *m* — including
    those that appear only inside a non-projection (complex) access."""
    ma = m.as_pw_multi_aff().as_multi_aff()
    n_in = ma.dim(isl.dim_type.IN)
    n_out = ma.dim(isl.dim_type.OUT)
    dims: set[int] = set()
    for o in range(n_out):
        aff = ma.get_at(o)
        for j in range(n_in):
            if int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()) != 0:
                dims.add(j)
    return dims


def _carrier_layout(
    input_type,
    input_map,
    output_map,
    output_shape,
    mesh,
    mesh_rank,
    propagated_attrs,
    complete_reduction_dims,
    fresh_strides,
):
    """Transform one input's ``ShardLayout`` into the output layout when
    that input's relation covers every output axis.

    Each input layout position is routed to an output axis via its domain dim
    (input map projection → output map projection); positions are emitted in
    output-axis order (preserving identity, applying a permutation, or
    collapsing a completely-reduced axis to size 1). Strides are carried from
    the input (a view, e.g. transpose) or rebuilt C-order (a fresh buffer, e.g.
    reduce). Returns ``None`` when the input does not project cleanly, does not
    cover all output axes, or does not realise the full propagated sharding —
    so a partial contributor never wins over the synthesis path.
    """
    sl = input_type.layout
    layout = sl.layout
    la2ta_in = layout_axis_to_tensor_axis(layout.shape, input_type.shape)
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
            # Reduced dim with no kept output axis (keepdim=False): the position
            # is retained as a collapsed size-1 layout axis, trailing onto the
            # last surviving output axis (dropped only for a scalar output).
            if out_rank:
                per_axis[out_rank - 1].append(p)
        else:
            return None  # position has nowhere to go and is not reduced

    new_shape: list = []
    new_pos_of: dict[int, int] = {}
    src_pos: list = []  # the input layout position each new position came from, or None
    for o in range(out_rank):
        positions = per_axis[o]
        if not positions:
            new_shape.append(1)  # kept size-1 axis with no carried position
            src_pos.append(None)
            continue
        for p in positions:
            reduced = pos_dom[p] in complete_reduction_dims
            new_pos_of[p] = len(new_shape)
            new_shape.append(1 if reduced else layout.shape[p])
            src_pos.append(None if reduced else p)
    if fresh_strides:
        # Fresh output buffer (e.g. Reduce): C-order strides over the new shape,
        # size-1 positions zeroed; never reads the input strides.
        c = try_c_order_strides(tuple(new_shape)) or tuple(1 for _ in new_shape)
        new_strides = [
            0 if (isinstance(sz, int) and sz == 1) else cc
            for sz, cc in zip(new_shape, c)
        ]
    elif layout.strides is None:
        new_strides = None  # input is an implicit-stride layout: stay implicit
    else:
        # View transform (e.g. Transpose): carry each position's input stride;
        # collapsed / placeholder positions get stride 0.
        new_strides = [
            0 if p is None else layout.strides[p] for p in src_pos
        ]

    out_attrs: list = [Broadcast() for _ in range(mesh_rank)]
    for p_mesh, attr in enumerate(sl.attrs):
        if (
            isinstance(attr, Split)
            and attr.axis in new_pos_of
            and pos_dom[attr.axis] not in complete_reduction_dims
        ):
            out_attrs[p_mesh] = Split(new_pos_of[attr.axis])
        elif isinstance(attr, Partial):
            # A Partial is a mesh-axis value state with no layout axis; it carries
            # through on the same mesh axis unchanged.
            out_attrs[p_mesh] = Partial(attr.reduction)

    # The carrier must realise exactly the full propagated sharding: map its
    # output layout attrs back to output tensor axes and compare.
    la2ta_out = layout_axis_to_tensor_axis(tuple(new_shape), tuple(output_shape))
    mapped: list = [Broadcast() for _ in range(mesh_rank)]
    for p_mesh, attr in enumerate(out_attrs):
        if isinstance(attr, Split):
            mapped[p_mesh] = Split(la2ta_out[attr.axis])
        elif isinstance(attr, Partial):
            mapped[p_mesh] = Partial(attr.reduction)
    if mapped != propagated_attrs:
        return None

    return ShardLayout(
        layout=Layout(
            shape=tuple(new_shape),
            strides=None if new_strides is None else tuple(new_strides),
        ),
        attrs=tuple(out_attrs),
        mesh=mesh,
    )


def derive_output_shard_layout(
    input_types: tuple,
    relation,
    output_shape: tuple,
    *,
    partial_reduction_dims: "frozenset[int]" = frozenset(),
    complete_reduction_dims: "frozenset[int]" = frozenset(),
    fresh_strides: bool = False,
):
    """Derive the output ``ShardLayout`` from the input shards and the forward relation."""
    sharded = [
        (i, t.layout)
        for i, t in enumerate(input_types)
        if isinstance(t.layout, ShardLayout)
        and any(isinstance(a, (Split, Partial)) for a in t.layout.attrs)
    ]
    if not sharded:
        return None
    mesh = sharded[0][1].mesh
    for _, sl in sharded:
        if sl.mesh != mesh:
            raise ValueError("inputs reference different meshes")
    mesh_rank = len(mesh.axes)

    *input_maps, output_map = relation.maps
    out_access = _result_access(output_map)
    domain_to_out_axis = {
        d: o for o, (kind, d) in out_access.items() if kind == "proj"
    }
    out_all_dims = _involved_domain_dims(output_map)

    attrs: list = [Broadcast() for _ in range(mesh_rank)]
    for i, sl in sharded:
        la2ta = layout_axis_to_tensor_axis(sl.layout.shape, input_types[i].shape)
        in_access = _result_access(input_maps[i])
        for p, attr in enumerate(sl.attrs):
            if isinstance(attr, Partial):
                # A Partial is a mesh-axis value state (no layout axis); an ordinary
                # op carries it through on the same mesh axis (no silent loss).
                # It is discharged only by an explicit reduction/allreduce.
                new_attr: object = Partial(attr.reduction)
                if not isinstance(attrs[p], Broadcast) and attrs[p] != new_attr:
                    raise ValueError(
                        f"mesh axis {p}: incompatible output shard {attrs[p]} vs {new_attr}"
                    )
                attrs[p] = new_attr
                continue
            if not isinstance(attr, Split):
                continue  # Broadcast input — output stays Broadcast here
            kind, ddim = in_access[la2ta[attr.axis]]
            if kind == "const":
                continue  # Split on a size-1 broadcast input — no contribution
            if kind == "complex":
                raise ValueError(
                    f"input {i} mesh axis {p}: Split on a non-projection access "
                    "is not supported for shard propagation"
                )
            if ddim in complete_reduction_dims:
                # A completely-reduced dim (e.g. Reduce) collapses its split to
                # Broadcast, even when the dim is kept as a size-1 output axis.
                new_attr = Broadcast()
            elif ddim in domain_to_out_axis:
                new_attr = Split(domain_to_out_axis[ddim])
            elif ddim in out_all_dims:
                # Survives in the output but only via a non-projection access —
                # the output layout axis cannot be derived; fail closed.
                raise ValueError(
                    f"input {i} mesh axis {p}: domain dim survives only via a "
                    "non-projection output access; cannot derive output layout axis"
                )
            elif ddim in partial_reduction_dims:
                # A Split of a contraction dim becomes a mesh-axis Partial.
                new_attr = Partial("sum")
            else:
                new_attr = Broadcast()
            if not isinstance(attrs[p], Broadcast) and attrs[p] != new_attr:
                raise ValueError(
                    f"mesh axis {p}: incompatible output shard {attrs[p]} vs {new_attr}"
                )
            attrs[p] = new_attr

    # If a single input's relation covers every output axis, transform that
    # input's layout into the output (identity / permutation / reduction
    # collapse) and carry it verbatim. The layout can be tiled / padded
    # (its position sizes need not be a clean factorisation of the tensor
    # shape), so synthesising it from mesh extents would be lossy or wrong; an
    # elementwise / transpose / reduce op preserves that operand's layout. Only
    # carry when the candidates agree — if two covering operands realise the
    # same logical sharding with different layout factorisations the result is
    # ambiguous, so fall through to the order-independent synthesis rather than
    # arbitrarily picking the first operand.
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
                mesh_rank,
                attrs,
                complete_reduction_dims,
                fresh_strides,
            )
        )
        is not None
    ]
    if carriers and all(c == carriers[0] for c in carriers):
        return carriers[0]

    # Otherwise synthesise the output layout from the per-mesh-axis bindings
    # (combining partial shards from several inputs) via the same
    # canonicalizer make_shard_tensor_type uses (docs/spec/shard.md §7.1.1):
    # an output axis split by one mesh axis is factored into an extent-sized
    # position (+ residual) exactly like a from-scratch sharding, and one
    # split by several mesh axes factorizes into one sub-position per mesh
    # extent (per shard.md §6) plus a remainder — so a carried-through layout
    # and a synthesised one for the same logical sharding always compare
    # equal. `Split.axis` in `attrs` still names an *output tensor* axis here
    # (canonical_shard_layout's expected input), not yet a layout position.
    return canonical_shard_layout(output_shape, mesh, tuple(attrs))


__all__ = ["derive_output_shard_layout", "partial_reductions_by_axis"]
