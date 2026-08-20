"""Generic shard propagation over a forward access relation.

``derive_output_shard_layout`` derives an op's output ``ShardLayout`` from the
input shards and the relation access maps by one rule for every op. It reads
only the maps' affine structure (which domain dim each tensor axis uses), never
the domain bounds, so it is size-agnostic and identical for static and dynamic
shapes.
"""
from __future__ import annotations

from dataclasses import replace

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
    shard_layout_of,
)
from tilefoundry.visitor_registry.access_relation import boundary_maps


def partial_reductions_by_axis(
    layout: object,
) -> tuple[str | None, ...]:
    """Return each mesh axis's carried Partial reduction, if any.

    The tuple index is the mesh-axis index in ``ShardLayout.attrs``. A
    ``None`` entry denotes an attr that is not a ``Partial``; a non-sharded
    layout returns an empty tuple.
    """
    layout = shard_layout_of(layout)
    if layout is None:
        return ()
    return tuple(
        attr.reduction if isinstance(attr, Partial) else None
        for attr in layout.attrs
    )


def _equal_axis(m: "isl.map", out_axis: int, in_dim: int) -> "isl.map":
    """Every pair whose result axis *out_axis* is its domain dim *in_dim*."""
    params = [
        m.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(m.dim(isl.dim_type.PARAM))
    ]
    prefix = f"[{', '.join(params)}] -> " if params else ""
    reads = ", ".join(f"i{index}" for index in range(m.dim(isl.dim_type.IN)))
    writes = ", ".join(f"o{index}" for index in range(m.dim(isl.dim_type.OUT)))
    return isl.map(f"{prefix}{{ [{reads}] -> [{writes}] : o{out_axis} = i{in_dim} }}")


def _tracked_anyway(m: "isl.map", out_axis: int) -> "int | None":
    """Whether a constant-looking axis provably *is* its own domain dim.

    An access map is restricted to the space its Op walks, and within a space
    that gives a dim one value the coordinate and the constant it equals are the
    same expression: isl prints whichever it likes. Only the axis's own dim is
    accepted, so what is recovered is the projection that was written and never
    some other dim that happens to hold the same value.
    """
    if out_axis >= m.dim(isl.dim_type.IN):
        return None
    return out_axis if m.is_subset(_equal_axis(m, out_axis, out_axis)) else None


def _result_access(
    m: "isl.map", *, folded: bool = False
) -> dict[int, "tuple[str, int | None]"]:
    """How each result axis of *m* reaches the domain: proj, const, or complex.

    ``folded`` asks the relation rather than its spelling. A map restricted to
    the space its Op walks prints an axis of extent one as either the coordinate
    or the constant it equals, and a result axis has to be traced back to the
    dim it came from.
    """
    ma = _single_affine_piece(m)
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
            tracked = _tracked_anyway(m, o) if folded else None
            out[o] = ("const", None) if tracked is None else ("proj", tracked)
        elif has_offset:
            out[o] = ("complex", None)
        elif len(used) == 1 and used[0][1] == 1:
            out[o] = ("proj", used[0][0])
        else:
            out[o] = ("complex", None)
    return out


def _single_affine_piece(m: "isl.map") -> "isl.multi_aff":
    """Return one affine access even when its iteration domain is restricted."""
    pieces: list[isl.multi_aff] = []
    m.as_pw_multi_aff().foreach_piece(lambda _domain, access: pieces.append(access))
    if len(pieces) != 1:
        raise ValueError(
            f"shard propagation requires one affine access piece, got {len(pieces)}"
        )
    return pieces[0]


def _involved_domain_dims(m: "isl.map") -> "set[int]":
    """All domain (in) dims referenced by any result axis of *m*.

    All domain (in) dims referenced by any result axis of *m* -- including
    those that appear only inside a non-projection (complex) access.
    """
    ma = _single_affine_piece(m)
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
    la2ta_in = layout_axis_to_tensor_axis(layout.shape, input_type.shape)
    in_access = _result_access(input_map, folded=True)
    out_access = _result_access(output_map, folded=True)
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
    if fresh_strides:


        c = try_c_order_strides(tuple(new_shape)) or tuple(1 for _ in new_shape)
        new_strides = [
            0 if (isinstance(sz, int) and sz == 1) else cc
            for sz, cc in zip(new_shape, c)
        ]
    elif layout.strides is None:
        new_strides = None
    else:


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


            out_attrs[p_mesh] = Partial(attr.reduction)



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
    relations,
    output_shape: tuple,
    *,
    partial_reduction_dims: "frozenset[int]" = frozenset(),
    complete_reduction_dims: "frozenset[int]" = frozenset(),
    fresh_strides: bool = False,
):
    """Derive the output ``ShardLayout`` from the input shards and the Op's relations."""
    input_types = tuple(
        replace(type_, layout=layout)
        if (layout := shard_layout_of(type_.layout)) is not None
        else type_
        for type_ in input_types
    )
    sharded = [
        (i, t.layout)
        for i, t in enumerate(input_types)
        if isinstance(t.layout, ShardLayout)
        and any(isinstance(a, (Split, Partial)) for a in t.layout.attrs)
    ]
    if not sharded:
        return None
    mesh = sharded[0][1].mesh
    for i, sl in sharded:
        if sl.mesh != mesh:
            raise ValueError(f"input {i} references a different mesh")
    mesh_rank = len(mesh.layout.shape)

    *input_maps, output_map = boundary_maps(relations)
    out_access = _result_access(output_map, folded=True)
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



                new_attr: object = Partial(attr.reduction)
                if not isinstance(attrs[p], Broadcast) and attrs[p] != new_attr:
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
            if ddim in complete_reduction_dims:


                new_attr = Broadcast()
            elif ddim in domain_to_out_axis:
                new_attr = Split(domain_to_out_axis[ddim])
            elif ddim in out_all_dims:


                raise ValueError(
                    f"input {i} mesh axis {p}: domain dim survives only via a "
                    "non-projection output access; cannot derive output layout axis"
                )
            elif ddim in partial_reduction_dims:

                new_attr = Partial("sum")
            else:
                new_attr = Broadcast()
            if not isinstance(attrs[p], Broadcast) and attrs[p] != new_attr:
                raise ValueError(
                    f"input {i} mesh axis {p}: incompatible output shard "
                    f"{attrs[p]} vs {new_attr}"
                )
            attrs[p] = new_attr











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











    return canonical_shard_layout(output_shape, mesh, tuple(attrs))


__all__ = ["derive_output_shard_layout", "partial_reductions_by_axis"]
