"""Sharded layouts: how a mesh divides a tensor, and the local layout one unit holds."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry import services
from tilefoundry.ir import expr

from .int_tuple import (
    compatible_profile,
    flatten_to_tuple,
    mode_axes,
    product,
    rank,
    shape_div,
    unwrap,
)
from .int_tuple import unflatten as int_tuple_unflatten
from .layout import (
    ComposedLayout,
    Layout,
    LayoutBase,
    flatten,
    get,
    plain_layout,
    select,
    unflatten,
)
from .layout_algebra import (
    coalesce,
    composition,
    make_composed_layout,
    make_layout,
    slice_and_offset,
)
from .mesh import Mesh, topology_sizes
from .stride import compact_row_major, crd2idx, idx2crd
from .topology import topology_name

__all__ = [
    "B",
    "Broadcast",
    "P",
    "Partial",
    "S",
    "ShardAttr",
    "ShardLayout",
    "Split",
    "V",
    "Varying",
    "combine_attrs",
    "local_layout",
    "local_layout_and_offset",
    "local_readback",
    "mesh_coords",
    "normalize_layout",
    "partial_reductions_by_axis",
    "physical_attr",
    "shard_layout_local_shape",
    "shard_layout_of",
    "split_target_axes",
]


class ShardAttr:
    """Base for per-mesh-axis sharding attributes."""


@dataclass(frozen=True)
class Split(ShardAttr):
    """The mesh axis divides a layout mode."""

    axis: int


@dataclass(frozen=True)
class Partial(ShardAttr):
    """Each unit holds a partial value to be reduced with *reduction*."""

    reduction: str = "sum"


@dataclass(frozen=True)
class Broadcast(ShardAttr):
    """Every unit along the mesh axis holds the whole value."""


@dataclass(frozen=True)
class Varying(ShardAttr):
    """Every copy holds the whole axis; values may differ between units."""


def combine_attrs(kind: expr.BinaryKind | None, a: ShardAttr, b: ShardAttr) -> ShardAttr:
    """Merge value states before access relations map Split attrs.

    See [shard §6.1](docs/spec/shard.md#61-shard-attr-semantics).
    """
    if a is None or b is None:
        raise ValueError("combine requires resolved attrs; read states with attrs_by_axis first")
    if isinstance(a, Partial) or isinstance(b, Partial):
        if isinstance(a, Partial) and isinstance(b, Partial):
            if kind is expr.BinaryKind.ADD and a.reduction == b.reduction == "sum":
                return a
        else:
            partial, other = (a, b) if isinstance(a, Partial) else (b, a)
            if isinstance(other, Broadcast) and (
                (kind is expr.BinaryKind.MUL and partial.reduction == "sum")
                or (kind is expr.BinaryKind.ADD and partial.reduction in {"max", "min"})
            ):
                return partial
        raise ValueError(
            f"cannot combine {kind!r}: {a!r} and {b!r}; "
            "bitcast(..., <its layout> @ {B()}) to declare equal values where required"
        )
    if isinstance(a, Split):
        return a
    if isinstance(b, Split):
        return b
    return Varying() if isinstance(a, Varying) or isinstance(b, Varying) else Broadcast()


def S(axis: int) -> Split:
    """``Split(axis)``."""
    return Split(axis)


def P(reduction: str = "sum") -> Partial:
    """``Partial(reduction)``."""
    return Partial(reduction)


def B() -> Broadcast:
    """``Broadcast()``."""
    return Broadcast()


def V() -> Varying:
    """``Varying()``."""
    return Varying()


def factor_axis(axis_size, mesh_shape, splitting_mesh_axes, logical_axis):
    """Factor a logical axis once for default construction and taking local."""
    axis_static = isinstance(axis_size, int) and not isinstance(axis_size, bool)
    if len(splitting_mesh_axes) == 1 and not axis_static:
        return (axis_size,)
    factors = []
    if not axis_static:
        residual = axis_size
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            try:
                quotient = shape_div(residual, extent)
            except ValueError:
                raise ValueError(
                    f"default_shard_layout: logical axis {logical_axis} size "
                    f"{axis_size!r} is dynamic; cannot factorize across multiple "
                    "mesh axes (undecidable division by axis "
                    f"{mesh_axis} extent {extent!r})"
                ) from None
            factors.append(extent)
            residual = quotient
    else:
        extent_product = 1
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            if not isinstance(extent, int) or isinstance(extent, bool):
                raise ValueError(
                    f"default_shard_layout: mesh axis {mesh_axis} has a "
                    f"dynamic extent {extent!r}; cannot factorize logical "
                    f"axis {logical_axis}"
                )
            factors.append(extent)
            extent_product *= extent
        if axis_size % extent_product:
            raise ValueError(
                f"default_shard_layout: logical axis {logical_axis} size "
                f"{axis_size} is not divisible by mesh extent product "
                f"{extent_product}"
            )
        residual = axis_size // extent_product
    if residual != 1:
        factors.append(residual)
    return tuple(factors)


@dataclass(frozen=True)
class ShardLayout(LayoutBase):
    """Bind an underlying layout's domain axes to a mesh."""

    layout: LayoutBase
    attrs: tuple[ShardAttr, ...]
    mesh: Mesh

    @classmethod
    def from_logical(cls, shape, mesh, attrs, *, storage, owners):
        """Factor logical axes by their mesh splits, then construct default strides."""
        mesh_shape = mesh.shape
        bindings = {}
        for mesh_axis, attr in enumerate(attrs):
            if isinstance(attr, Split):
                if mesh_axis >= len(mesh_shape) or attr.axis >= len(shape):
                    raise ValueError(f"Split({attr.axis}) has no matching mesh axis {mesh_axis}")
                bindings.setdefault(attr.axis, []).append(mesh_axis)
        layout_shape, factor_mode = [], {}
        for logical_axis, axis_size in enumerate(shape):
            splitting = bindings.get(logical_axis, ())
            for index, mesh_axis in enumerate(splitting):
                factor_mode[mesh_axis] = len(layout_shape) + index
            layout_shape.extend(factor_axis(axis_size, mesh_shape, splitting, logical_axis) if splitting else (axis_size,))

        remapped_attrs = tuple(
            Split(factor_mode[mesh_axis]) if isinstance(attr, Split) else attr
            for mesh_axis, attr in enumerate(attrs)
        )
        layout_shape = tuple(layout_shape)

        return cls.from_modes(layout_shape, mesh, remapped_attrs, storage=storage, owners=owners)

    @classmethod
    def from_modes(cls, shape, mesh, attrs, *, storage, owners):
        """Construct authored modes with the owner's compact local strides.

        With no *owners* (no target), no storage is private to a unit.
        """
        topologies = tuple(
            topology_name(topology)
            for topology, mode in zip(mesh.topologies, plain_layout(mesh.layout).shape)
            for _ in flatten_to_tuple(mode)
        )
        private = []
        for mesh_axis, attr in enumerate(attrs):
            if isinstance(attr, Split) and (
                mesh_axis >= len(topologies) or attr.axis >= rank(flatten_to_tuple(shape))
            ):
                raise ValueError(f"Split({attr.axis}) has no matching mesh axis {mesh_axis}")
            private.append(
                attr
                if isinstance(attr, Split) and owners is not None and owners.private(storage, topologies[mesh_axis])
                else Broadcast()
            )
        local = divided_shape(flatten_to_tuple(shape), mesh, tuple(private), require_static=False)
        strides = list(Layout(local).strides)
        for attr in private:
            if isinstance(attr, Split):
                strides[attr.axis] = 0
        return cls(Layout(shape, int_tuple_unflatten(tuple(strides), shape)), attrs, mesh)

    @property
    def shape(self) -> tuple:
        """The domain shape of the underlying layout."""
        return self.layout.shape


def shard_layout_of(layout: object) -> "ShardLayout | None":
    """Return the distribution carried directly or by a static-offset view."""
    if isinstance(layout, ShardLayout):
        return layout
    if (
        isinstance(layout, ComposedLayout)
        and layout.inner is None
        and isinstance(layout.outer, ShardLayout)
    ):
        return layout.outer
    return None


def physical_attr(attr: ShardAttr) -> ShardAttr:
    """Read value variation as Broadcast for coordinate and execution queries."""
    return Broadcast() if isinstance(attr, Varying) else attr


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


def shard_layout_local_shape(sl: "ShardLayout", *, require_static: bool = True) -> tuple:
    """Derive one shard's local shape from its global layout.

    Splits divide by mesh extents: equal symbols yield one, matching products
    yield their other factor, and other symbolic splits are undecidable.
    Non-Split attributes consume no modes. ``require_static=False`` permits
    unconsumed symbols during inference; lowering and codegen require integers.

    See [shard §7](docs/spec/shard.md#7-shardlayout).
    """
    return divided_shape(sl.layout.shape, sl.mesh, sl.attrs, require_static=require_static)


def divided_shape(shape, mesh, attrs, *, require_static):
    """*shape* divided by the mesh extent of every ``Split`` in *attrs*."""
    mesh_shape = flatten_to_tuple(mesh.layout.shape)
    local = list(shape)
    for mesh_axis_idx, attr in enumerate(attrs):
        if mesh_axis_idx >= len(mesh_shape):
            break
        if isinstance(attr, Split):
            k = attr.axis
            if not (0 <= k < len(local)):
                continue
            mesh_ext = mesh_shape[mesh_axis_idx]
            if isinstance(mesh_ext, int) and isinstance(local[k], int):
                if mesh_ext != 0:
                    local[k] //= mesh_ext
            else:
                try:
                    local[k] = shape_div(local[k], mesh_ext)
                except ValueError:
                    raise ValueError(
                        f"shard_layout_local_shape: layout dim {k} ({local[k]!r}) "
                        f"and mesh axis {mesh_axis_idx} extent {mesh_ext!r} do not "
                        "have a decidable divisibility relation; bind symbolic "
                        "dimensions before local projection"
                    ) from None

    if require_static:
        for i, d in enumerate(local):
            if not isinstance(d, int):
                raise ValueError(
                    f"shard_layout_local_shape: per-shard dim {i} ({d!r}) is not "
                    "static after sharding; bind symbolic dimensions before local "
                    "projection"
                )
    return tuple(local)


def split_target_axes(sl: "ShardLayout", tensor_shape: tuple) -> tuple:
    """Per mesh axis in ``sl.attrs``, the logical ``tensor_shape`` axis its ``Split`` targets.

    Per mesh axis in ``sl.attrs``, the logical ``tensor_shape`` axis its
    ``Split`` targets (``None`` for a non-``Split`` attr).
    """
    la2ta = mode_axes(compatible_profile(sl.layout.shape, tensor_shape))
    return tuple(la2ta[a.axis] if isinstance(a, Split) else None for a in sl.attrs)


def local_layout(shard: ShardLayout, tensor_shape: tuple, coords: tuple):
    """Fix mesh-axis coordinates with CuTe slice; None keeps an axis."""
    address = shard.layout
    primitive = plain_layout(address)
    profile = compatible_profile(primitive.shape, tensor_shape)
    flat, mesh = flatten(primitive), flatten(plain_layout(shard.mesh.layout))
    coords = tuple(coords) + (None,) * (len(shard.attrs) - len(coords))
    shape, strides, fixed, modes = [], [], [], {}
    for mode, (extent, stride) in enumerate(zip(flat.shape, flat.strides, strict=True)):
        mesh_axes = [
            mesh_axis
            for mesh_axis, attr in enumerate(shard.attrs)
            if isinstance(attr, Split) and attr.axis == mode
        ]
        normalize = any(coords[mesh_axis] is not None for mesh_axis in mesh_axes)
        factors = factor_axis(extent, mesh.shape, mesh_axes, mode) if normalize else (extent,)
        steps = flatten_to_tuple(composition(Layout(extent, stride), make_layout(factors)).strides)
        for index, mesh_axis in enumerate(mesh_axes):
            modes[mesh_axis] = len(shape) + (index if normalize else 0)
        for index, (factor, step) in enumerate(zip(factors, steps, strict=True)):
            mesh_axis = mesh_axes[index] if index < len(mesh_axes) else None
            symbolic_whole = len(mesh_axes) == 1 and not isinstance(extent, int)
            fixed.append(
                coords[mesh_axis]
                if mesh_axis is not None and not symbolic_whole and factor == mesh.shape[mesh_axis]
                else None
            )
            shape.append(factor)
            strides.append(services.normalize_dim(step))
    profile = compatible_profile(tuple(shape), tensor_shape)
    grouped = unflatten(Layout(tuple(shape), tuple(strides)), profile)
    c = int_tuple_unflatten(tuple(fixed), profile)
    sliced, start = slice_and_offset(
        c, replace(address, outer=grouped) if isinstance(address, ComposedLayout) else grouped
    )
    local_shape = tuple(
        services.normalize_dim(
            product(
                tuple(
                    n
                    for n, at in zip(flatten_to_tuple(mode), flatten_to_tuple(coord), strict=True)
                    if at is None
                )
            )
        )
        for mode, coord in zip(profile, c, strict=True)
    )
    inner = plain_layout(sliced)
    inner = unflatten(flatten(inner), compatible_profile(inner.shape, local_shape))
    sliced = (
        replace(sliced, outer=inner, offset=services.normalize_dim(sliced.offset))
        if isinstance(sliced, ComposedLayout)
        else inner
    )
    kept = {old: new for new, old in enumerate(i for i, at in enumerate(fixed) if at is None)}
    remaining = [mesh_axis for mesh_axis, at in enumerate(coords) if at is None]
    if remaining:
        stated = plain_layout(shard.mesh.layout)
        keys = tuple(
            t for t in range(len(shard.mesh.topologies)) for _ in flatten_to_tuple(get(stated, t).shape)
        )
        held = {}
        for mesh_axis in remaining:
            if mesh_axis < len(keys):
                held.setdefault(keys[mesh_axis], []).append(mesh_axis - keys.index(keys[mesh_axis]))
        names = tuple(shard.mesh.names[i] for i in remaining if i < len(shard.mesh.names))
        if len(held) == len(shard.mesh.topologies):
            outer = Layout(
                tuple(select(flatten(get(stated, t)), *mesh_axes).shape for t, mesh_axes in held.items()),
                tuple(select(flatten(get(stated, t)), *mesh_axes).strides for t, mesh_axes in held.items()),
            )
            layout = (
                replace(shard.mesh.layout, outer=outer)
                if isinstance(shard.mesh.layout, ComposedLayout)
                else outer
            )
            shard_mesh = Mesh(shard.mesh.topologies, layout, names)
        else:
            shard_mesh = kept_axes(shard.mesh, held, names)
        attrs = tuple(
            Split(kept[modes[mesh_axis]])
            if isinstance(shard.attrs[mesh_axis], Split)
            else shard.attrs[mesh_axis]
            for mesh_axis in remaining
        )
        sliced = ShardLayout(sliced, attrs, shard_mesh)
    return sliced, services.normalize_dim(start)


def kept_axes(mesh: Mesh, held: dict, names: tuple) -> Mesh:
    """The mesh axes *held* keeps of some topologies of *mesh*, as a mesh of those topologies.

    *held* maps a topology's position to the mesh axes of its mode it keeps. Each kept
    mode is ``select`` of those mesh axes, moved to the index over the kept topologies
    by ``composition(Layout(sizes, u_t * e_t), ·)``: ``u_t`` is the
    ``compact_row_major`` stride of *t* among the kept topologies, ``e_t`` is 1
    at *t* and 0 elsewhere. The first index is ``crd2idx`` of the kept
    topologies' first unit ids, ``idx2crd(L(0), sizes)``.
    """
    stated = plain_layout(mesh.layout)
    sizes = topology_sizes(mesh.topologies)
    topologies = tuple(mesh.topologies[t] for t in held)
    kept_sizes = topology_sizes(topologies)
    units = compact_row_major(kept_sizes)
    kept = make_layout(
        *(
            composition(
                Layout(sizes, tuple(unit if at == t else 0 for at in range(len(sizes)))),
                select(flatten(get(stated, t)), *mesh_axes),
            )
            for (t, mesh_axes), unit in zip(held.items(), units)
        )
    )
    first_unit_ids = idx2crd(mesh.layout(0), sizes)
    index = crd2idx(tuple(first_unit_ids[t] for t in held), kept_sizes, units)
    return Mesh(topologies, make_composed_layout(None, index, kept) if index else kept, names)


def local_readback(layout):
    """Read a local mode tree as the host/HIR logical-axis Layout."""
    address = layout.layout if isinstance(layout, ShardLayout) else layout
    result = coalesce(address, (0,) * len(address.shape), major="row")
    primitive = plain_layout(result)
    shape = tuple(unwrap(mode) for mode in primitive.shape)
    strides = tuple(unwrap(mode) for mode in primitive.strides)
    plain = Layout(shape, strides)
    return replace(result, outer=plain) if isinstance(result, ComposedLayout) else plain


def mesh_coords(mesh: Mesh, ids: tuple[int | None, ...]) -> tuple[int | None, ...]:
    """Each mesh axis's coordinate under per-topology unit ids; None where one is unfixed.

    The host side of the CUDA runtime's ``mesh_coords(mesh, unit_ids())``. Each
    topology ``t`` is read alone: its id counts from the first unit's,
    ``idx2crd(L(0), sizes)[t]``, steps ``stride(T)[t]`` in the index over all
    topologies, and ``idx2crd`` over its mode's shape and strides gives the coord.
    """
    stated = plain_layout(mesh.layout)
    sizes = topology_sizes(mesh.topologies)
    units = compact_row_major(sizes) if len(sizes) > 1 else (1,)
    first_unit_ids = idx2crd(mesh.layout(0), sizes)
    coords = []
    for t in range(len(mesh.topologies)):
        mode = flatten(get(stated, t))
        unit_id = ids[t] if t < len(ids) else None
        coords.extend(
            (None,) * len(mode.shape)
            if unit_id is None
            else idx2crd((unit_id - first_unit_ids[t]) * units[t], mode.shape, mode.strides)
        )
    return tuple(coords)


def local_layout_and_offset(shard: ShardLayout, tensor_shape: tuple, coord: tuple):
    """The logical local Layout the unit at mesh *coord* holds, and where it begins.

    The host side of the CUDA runtime's ``local_layout_and_offset(sl, coord)``.
    """
    if any(not isinstance(size, int) or isinstance(size, bool) for size in tensor_shape):
        raise ValueError("local offset requires concrete bound tensor sizes")
    layout, start = local_layout(shard, tensor_shape, coord)
    return local_readback(layout), start


def unit_leaf_strides(shape, strides):
    """*strides* with a zero step wherever the matching extent is a static one."""
    if isinstance(shape, tuple):
        return tuple(
            unit_leaf_strides(one, step) for one, step in zip(shape, strides, strict=True)
        )
    return 0 if expr.static_dim_value(shape) == 1 else strides


def normalize_layout(layout: LayoutBase) -> LayoutBase:
    """Normalize unit leaves without changing modes, shard attrs or addresses."""
    if isinstance(layout, Layout):
        return replace(layout, strides=unit_leaf_strides(layout.shape, layout.strides))
    if isinstance(layout, ComposedLayout):
        return replace(
            layout,
            outer=normalize_layout(layout.outer) if layout.outer is not None else None,
            inner=normalize_layout(layout.inner)
            if isinstance(layout.inner, LayoutBase)
            else layout.inner,
        )
    if isinstance(layout, ShardLayout):
        return replace(layout, layout=normalize_layout(layout.layout))
    raise TypeError(f"not a tensor layout: {type(layout).__name__}")
