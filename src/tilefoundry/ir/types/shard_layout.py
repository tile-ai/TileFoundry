from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.types.int_tuple import compatible_profile, mode_axes
from tilefoundry.ir.types.layout import Layout, flatten
from tilefoundry.ir.types.stride import idx2crd

from .layout import ComposedLayout, LayoutBase
from .mesh import Mesh
from .stride import compact_row_major, crd2idx


class ShardAttr:
    """Base for per-mesh-axis sharding attributes."""

    pass


@dataclass(frozen=True)
class Split(ShardAttr):
    axis: int


@dataclass(frozen=True)
class Partial(ShardAttr):
    reduction: str = "sum"


@dataclass(frozen=True)
class Broadcast(ShardAttr):
    pass


@dataclass(frozen=True)
class Dynamic(ShardAttr):
    pass


def S(axis: int) -> Split:
    return Split(axis)


def P(reduction: str = "sum") -> Partial:
    return Partial(reduction)


def B() -> Broadcast:
    return Broadcast()


def _factor_axis(axis_size, mesh_shape, splitting_mesh_axes, logical_axis):
    """Factor a logical axis once for default construction and taking local."""
    from .dim import exact_quotient  # noqa: PLC0415

    axis_static = isinstance(axis_size, int) and not isinstance(axis_size, bool)
    if len(splitting_mesh_axes) == 1 and not axis_static:
        return (axis_size,)
    factors = []
    if not axis_static:
        residual = axis_size
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            quotient = exact_quotient(residual, extent)
            if quotient is None:
                raise ValueError(
                    f"default_shard_layout: logical axis {logical_axis} size "
                    f"{axis_size!r} is dynamic; cannot factorize across multiple "
                    "mesh axes (undecidable division by axis "
                    f"{mesh_axis} extent {extent!r})"
                )
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
    def default(cls, shape, mesh, attrs, *, storage, owners):
        """Factor logical axes by their mesh splits, then construct default strides."""
        mesh_shape = flatten(mesh.layout).shape
        bindings = {}
        for mesh_axis, attr in enumerate(attrs):
            if isinstance(attr, Split):
                if mesh_axis >= len(mesh_shape) or attr.axis >= len(shape):
                    raise ValueError(f"Split({attr.axis}) has no matching mesh axis {mesh_axis}")
                bindings.setdefault(attr.axis, []).append(mesh_axis)
        layout_shape, factor_position = [], {}
        for logical_axis, axis_size in enumerate(shape):
            splitting = bindings.get(logical_axis, ())
            for index, mesh_axis in enumerate(splitting):
                factor_position[mesh_axis] = len(layout_shape) + index
            layout_shape.extend(_factor_axis(axis_size, mesh_shape, splitting, logical_axis) if splitting else (axis_size,))

        remapped_attrs = tuple(
            Split(factor_position[mesh_axis]) if isinstance(attr, Split) else attr
            for mesh_axis, attr in enumerate(attrs)
        )
        layout_shape = tuple(layout_shape)

        return cls.from_modes(layout_shape, mesh, remapped_attrs, storage=storage, owners=owners)

    @classmethod
    def from_modes(cls, shape, mesh, attrs, *, storage, owners):
        """Construct authored modes with the owner's compact local strides."""
        from .int_tuple import unflatten  # noqa: PLC0415
        from .mesh import separate  # noqa: PLC0415

        topologies = tuple(
            part.topologies[0].name for part in separate(mesh) for _ in flatten(part.layout).shape
        )
        private = []
        for axis, attr in enumerate(attrs):
            if isinstance(attr, Split) and (
                axis >= len(topologies) or attr.axis >= len(flatten(shape))
            ):
                raise ValueError(f"Split({attr.axis}) has no matching mesh axis {axis}")
            private.append(
                attr
                if isinstance(attr, Split) and owners.private(storage, topologies[axis])
                else Broadcast()
            )
        local = _local_shape(tuple(flatten(shape)), mesh, tuple(private), require_static=False)
        strides = list(Layout(local).strides)
        for attr in private:
            if isinstance(attr, Split):
                strides[attr.axis] = 0
        return cls(Layout(shape, unflatten(tuple(strides), shape)), attrs, mesh)

    @property
    def shape(self) -> tuple:
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


def shard_layout_local_shape(sl: "ShardLayout", *, require_static: bool = True) -> tuple:
    """Derive one shard's local shape from its global layout.

    Splits divide by mesh extents: equal symbols yield one, matching products
    yield their other factor, and other symbolic splits are undecidable.
    Non-Split attributes consume no positions. ``require_static=False`` permits
    unconsumed symbols during inference; lowering and codegen require integers.

    Defer exact_quotient: core.op loads types.storage and staged shard_layout
    exports before core.expr defines Call, which dim needs during import.

    See [shard §7](docs/spec/shard.md#7-shardlayout).
    """
    return _local_shape(sl.layout.shape, sl.mesh, sl.attrs, require_static=require_static)


def _local_shape(shape, mesh, attrs, *, require_static):
    from .dim import exact_quotient  # noqa: PLC0415

    mesh_shape = flatten(mesh.layout).shape
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
                quotient = exact_quotient(local[k], mesh_ext)
                if quotient is None:
                    raise ValueError(
                        f"shard_layout_local_shape: layout dim {k} ({local[k]!r}) "
                        f"and mesh axis {mesh_axis_idx} extent {mesh_ext!r} do not "
                        "have a decidable divisibility relation; bind symbolic "
                        "dimensions before local projection"
                    )
                local[k] = quotient

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
    from tilefoundry.ir.isl_interop import normalize_dim  # noqa: PLC0415

    from .int_tuple import product  # noqa: PLC0415
    from .int_tuple import unflatten as unflatten_tuple  # noqa: PLC0415
    from .layout import unflatten  # noqa: PLC0415
    from .layout_algebra import slice_and_offset  # noqa: PLC0415
    from .mesh import levels  # noqa: PLC0415

    address = shard.layout
    primitive = address.outer if isinstance(address, ComposedLayout) else address
    profile = compatible_profile(primitive.shape, tensor_shape)
    flat, mesh = flatten(primitive), flatten(shard.mesh.layout)
    coords = tuple(coords) + (None,) * (len(shard.attrs) - len(coords))
    shape, strides, fixed, positions = [], [], [], {}
    for mode, (extent, stride) in enumerate(zip(flat.shape, flat.strides, strict=True)):
        axes = [
            axis
            for axis, attr in enumerate(shard.attrs)
            if isinstance(attr, Split) and attr.axis == mode
        ]
        normalize = any(coords[axis] is not None for axis in axes)
        factors = _factor_axis(extent, mesh.shape, axes, mode) if normalize else (extent,)
        steps = compact_row_major(factors)
        for index, mesh_axis in enumerate(axes):
            positions[mesh_axis] = len(shape) + (index if normalize else 0)
        for index, (factor, step) in enumerate(zip(factors, steps, strict=True)):
            axis = axes[index] if index < len(axes) else None
            symbolic_whole = len(axes) == 1 and not isinstance(extent, int)
            fixed.append(
                coords[axis]
                if axis is not None and not symbolic_whole and factor == mesh.shape[axis]
                else None
            )
            shape.append(factor)
            strides.append(normalize_dim(crd2idx(step, factor, stride)))
    profile = compatible_profile(tuple(shape), tensor_shape)
    grouped = unflatten(Layout(tuple(shape), tuple(strides)), profile)
    c = unflatten_tuple(tuple(fixed), profile)
    sliced, offset = slice_and_offset(
        c, replace(address, outer=grouped) if isinstance(address, ComposedLayout) else grouped
    )
    local_shape = tuple(
        normalize_dim(
            product(
                tuple(n for n, at in zip(flatten(mode), flatten(coord), strict=True) if at is None)
            )
        )
        for mode, coord in zip(profile, c, strict=True)
    )
    inner = sliced.outer if isinstance(sliced, ComposedLayout) else sliced
    inner = unflatten(flatten(inner), compatible_profile(inner.shape, local_shape))
    sliced = (
        replace(sliced, outer=inner, offset=normalize_dim(sliced.offset))
        if isinstance(sliced, ComposedLayout)
        else inner
    )
    kept = {old: new for new, old in enumerate(i for i, at in enumerate(fixed) if at is None)}
    remaining = [axis for axis, at in enumerate(coords) if at is None]
    if remaining:
        groups, topologies, begin = [], [], 0
        for topology, arrangement in zip(shard.mesh.topologies, levels(shard.mesh), strict=True):
            count = len(flatten(arrangement.shape))
            group = [axis for axis in remaining if begin <= axis < begin + count]
            if group:
                topologies.append(topology)
                groups.append(group)
            begin += count
        outer = Layout(
            tuple(tuple(mesh.shape[i] for i in group) for group in groups),
            tuple(tuple(mesh.strides[i] for i in group) for group in groups),
        )
        layout = (
            replace(shard.mesh.layout, outer=outer)
            if isinstance(shard.mesh.layout, ComposedLayout)
            else outer
        )
        frame = Mesh(
            tuple(topologies),
            layout,
            tuple(shard.mesh.names[i] for i in remaining if i < len(shard.mesh.names)),
        )
        attrs = tuple(
            Split(kept[positions[axis]])
            if isinstance(shard.attrs[axis], Split)
            else shard.attrs[axis]
            for axis in remaining
        )
        sliced = ShardLayout(sliced, attrs, frame)
    return sliced, normalize_dim(offset)


def _local_readback(layout):
    """Read a local mode tree as the host/HIR logical-axis Layout."""
    from .layout_algebra import coalesce  # noqa: PLC0415

    address = layout.layout if isinstance(layout, ShardLayout) else layout
    result = coalesce(address, (0,) * len(address.shape), major="row")
    primitive = result.outer if isinstance(result, ComposedLayout) else result
    shape = tuple(
        mode[0] if isinstance(mode, tuple) and len(mode) == 1 else mode for mode in primitive.shape
    )
    strides = tuple(
        mode[0] if isinstance(mode, tuple) and len(mode) == 1 else mode
        for mode in primitive.strides
    )
    plain = Layout(shape, strides)
    return replace(result, outer=plain) if isinstance(result, ComposedLayout) else plain


def local_layout_and_offset(shard: ShardLayout, tensor_shape: tuple, ids: tuple[int | None, ...]):
    """Adapt topology program ids and read back the logical local Layout."""
    from .mesh import levels  # noqa: PLC0415

    if any(not isinstance(size, int) or isinstance(size, bool) for size in tensor_shape):
        raise ValueError("local offset requires concrete bound tensor sizes")
    coords = []
    for index, arrangement in enumerate(levels(shard.mesh)):
        flat = flatten(arrangement)
        program_id = ids[index] if index < len(ids) else None
        coords.extend(
            (None,) * len(flat.shape)
            if program_id is None
            else idx2crd(program_id, flat.shape, flat.strides)
        )
    layout, offset = local_layout(shard, tensor_shape, tuple(coords))
    return _local_readback(layout), offset


__all__ = [
    "local_layout",
    "local_layout_and_offset",
    "ShardAttr",
    "Split",
    "Partial",
    "Broadcast",
    "Dynamic",
    "S",
    "P",
    "B",
    "ShardLayout",
    "shard_layout_of",
    "shard_layout_local_shape",
    "split_target_axes",
    "register_strides",
]


def register_strides(sl: ShardLayout) -> tuple[int, ...]:
    """``sl``'s strides as steps on a *register* engine, not on a shared buffer.

    Registers are the distinct-engine-per-instance case of [shard
    §7.1.2](docs/spec/shard.md#712-layoutstrides): a ``Split(k)`` axis gets
    ``0``, so the projection offsets no instance along it, and the rest get
    the array's own compact steps, assigned from smallest to largest stated
    physical stride. This preserves register numbering when modes are reordered.
    An extent-1 axis names no step.
    """
    from .utils import upper_bound  # noqa: PLC0415 - staged type imports

    local = shard_layout_local_shape(sl, require_static=False)
    split_axes = {int(a.axis) for a in sl.attrs if isinstance(a, Split)}
    physical = flatten(sl.layout)
    order = sorted(range(len(local)), key=lambda axis: int(upper_bound(physical.strides[axis])))
    strides: list[int] = [0] * len(local)
    step = 1
    for axis in order:
        width = int(upper_bound(local[axis]))
        if axis not in split_axes and width != 1:
            strides[axis] = step
            step *= width
    return tuple(strides)
