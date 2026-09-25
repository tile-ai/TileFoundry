from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.int_tuple import repeat_like
from tilefoundry.ir.types.layout import flatten, rank
from tilefoundry.ir.types.stride import idx2crd

from .layout import ComposedLayout, Layout, LayoutBase, get
from .mesh import Mesh
from .stride import compact_row_major, try_compact_major


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


@dataclass(frozen=True)
class ShardLayout(LayoutBase):
    """Bind an underlying layout's domain axes to a mesh."""

    layout: LayoutBase
    attrs: tuple[ShardAttr, ...]
    mesh: Mesh

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


def canonical_shard_layout(logical_shape: tuple, mesh: Mesh, attrs: tuple) -> "ShardLayout":
    """Bind logical axes to mesh axes in the canonical factored layout.

    Static splits produce mesh-sized positions plus a residual; dynamic
    single-axis splits remain whole. Attributes are remapped to factored
    positions and strides are rebuilt in C order when static.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    mesh_shape = flatten(mesh.layout).shape
    bindings: dict[int, list[int]] = {}
    for mesh_axis, attr in enumerate(attrs):
        if isinstance(attr, Split):
            bindings.setdefault(attr.axis, []).append(mesh_axis)

    layout_shape: list = []
    factor_position: dict[int, int] = {}
    for logical_axis, axis_size in enumerate(logical_shape):
        splitting_mesh_axes = bindings.get(logical_axis, [])
        if not splitting_mesh_axes:
            layout_shape.append(axis_size)
            continue

        axis_static = isinstance(axis_size, int) and not isinstance(axis_size, bool)
        if len(splitting_mesh_axes) == 1 and not axis_static:
            factor_position[splitting_mesh_axes[0]] = len(layout_shape)
            layout_shape.append(axis_size)
            continue
        extent_product = 1
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            if not (isinstance(extent, int) and not isinstance(extent, bool)):
                raise ValueError(
                    f"canonical_shard_layout: mesh axis {mesh_axis} has a "
                    f"dynamic extent {extent!r}; cannot factorize logical "
                    f"axis {logical_axis}"
                )
            factor_position[mesh_axis] = len(layout_shape)
            layout_shape.append(extent)
            extent_product *= extent
        if not axis_static:
            raise ValueError(
                f"canonical_shard_layout: logical axis {logical_axis} size "
                f"{axis_size!r} is dynamic; cannot factorize across multiple "
                f"mesh axes"
            )
        if axis_size % extent_product != 0:
            raise ValueError(
                f"canonical_shard_layout: logical axis {logical_axis} size "
                f"{axis_size} is not divisible by mesh extent product "
                f"{extent_product}"
            )
        residual = axis_size // extent_product
        if residual != 1:
            layout_shape.append(residual)

    remapped_attrs = tuple(
        Split(factor_position[mesh_axis]) if isinstance(attr, Split) else attr
        for mesh_axis, attr in enumerate(attrs)
    )
    layout_shape = tuple(layout_shape)
    return ShardLayout(
        layout=Layout(shape=layout_shape, strides=try_compact_major(layout_shape)),
        attrs=remapped_attrs,
        mesh=mesh,
    )


def shard_layout_local_shape(
    sl: "ShardLayout", *, require_static: bool = True
) -> tuple:
    """Derive one shard's local shape from a global ``ShardLayout``.

    Each ``Split`` divides its bound layout position by the mesh extent;
    repeated splits multiply their divisors. Equal symbolic extents yield one;
    other symbolic splits are undecidable before binding. Other attributes do
    not consume a layout position. ``require_static`` keeps lowering and
    codegen on their concrete-shape boundary while type inference may retain an
    unconsumed symbolic extent.

    See [shard §7](docs/spec/shard.md#7-shardlayout).
    """
    mesh_shape = flatten(sl.mesh.layout).shape
    local = list(sl.layout.shape)
    for mesh_axis_idx, attr in enumerate(sl.attrs):
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
            elif local[k] == mesh_ext:
                local[k] = 1
            else:
                raise ValueError(
                    f"shard_layout_local_shape: layout dim {k} ({local[k]!r}) "
                    f"and mesh axis {mesh_axis_idx} extent {mesh_ext!r} do not "
                    "have a decidable divisibility relation; bind symbolic "
                    "dimensions before local projection"
                )

    if require_static:
        for i, d in enumerate(local):
            if not isinstance(d, int):
                raise ValueError(
                    f"shard_layout_local_shape: per-shard dim {i} ({d!r}) is not "
                    "static after sharding; bind symbolic dimensions before local "
                    "projection"
                )
    return tuple(local)


def layout_axis_to_tensor_axis(layout_shape: tuple, tensor_shape: tuple) -> list[int]:
    """Map factored layout positions to their logical tensor axes.

    Positions are consumed left-to-right until their product reaches each
    tensor extent. Singleton tensor axes claim one singleton position; trailing
    positions attach to the final tensor axis.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    from .utils import static_dim_value  # noqa: PLC0415 - cycle guard

    result: list[int] = []
    layout_idx = 0
    for t_axis, t_dim in enumerate(tensor_shape):
        t_dim_int = static_dim_value(t_dim)
        if t_dim_int is None:
            if layout_idx < len(layout_shape) and layout_shape[layout_idx] == t_dim:
                result.append(t_axis)
                layout_idx += 1
                continue
            while layout_idx < len(layout_shape):
                result.append(t_axis)
                layout_idx += 1
            return result
        if t_dim_int == 1:
            if layout_idx < len(layout_shape):
                result.append(t_axis)
                layout_idx += 1
            continue
        running = 1
        while layout_idx < len(layout_shape) and running < t_dim_int:
            sh = static_dim_value(layout_shape[layout_idx])
            running *= 1 if sh is None else sh
            result.append(t_axis)
            layout_idx += 1
    while layout_idx < len(layout_shape):
        result.append(len(tensor_shape) - 1)
        layout_idx += 1
    return result


def split_target_axes(sl: "ShardLayout", tensor_shape: tuple) -> tuple:
    """Per mesh axis in ``sl.attrs``, the logical ``tensor_shape`` axis its ``Split`` targets.

    Per mesh axis in ``sl.attrs``, the logical ``tensor_shape`` axis its
    ``Split`` targets (``None`` for a non-``Split`` attr).
    """
    la2ta = layout_axis_to_tensor_axis(sl.layout.shape, tensor_shape)
    return tuple(la2ta[a.axis] if isinstance(a, Split) else None for a in sl.attrs)


def _extents(shard: ShardLayout) -> tuple[int, ...]:
    """Each mesh axis's extent, flat and in the order the attrs index them."""
    values = tuple(flatten(flatten(shard.mesh.layout).shape))
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"local: this mesh states a symbolic extent {value!r}; which part is "
                f"one instance's is a number, so bind the mesh first"
            )
    return values


def _cutting(shard: ShardLayout, tensor_shape: tuple) -> tuple[int | None, ...]:
    """Per mesh axis, the tensor axis its attr cuts, or ``None`` when none."""
    return split_target_axes(shard, tuple(tensor_shape))


def _strides(shard: ShardLayout, tensor_shape: tuple) -> tuple[int, ...]:
    """Each tensor axis's own stride, read off the factored layout.

    A split axis is stated as mesh-sized positions plus a residual, so the
    step of the tensor axis itself is the last of them; a layout that
    materialized no strides is the C order its canonical form would have.
    """
    fallback = compact_row_major(tuple(tensor_shape))
    stated = shard.layout.strides
    if stated is None:
        return fallback
    axes = layout_axis_to_tensor_axis(shard.layout.shape, tuple(tensor_shape))
    found = dict(zip(axes, stated, strict=False))
    return tuple(found.get(axis, fallback[axis]) for axis in range(len(tensor_shape)))


def _narrowed(
    shard: ShardLayout, tensor_shape: tuple, axis: int, dividing: Iterable[int]
) -> int:
    """Tensor *axis* divided by each mesh axis in *dividing* that cuts it.

    Several may. Two axes of one level cut a tensor axis into a grid, and so
    do two levels, one taking a block of what the other left; each division is
    of what the previous one left. An extent its mesh axis does not divide is
    refused by name: a shard would then not be one slice.
    """
    extents = _extents(shard)
    allowed = set(dividing)
    held = tensor_shape[axis]
    for mesh_axis, target in enumerate(_cutting(shard, tensor_shape)):
        if target != axis or mesh_axis >= len(extents) or mesh_axis not in allowed:
            continue
        count = extents[mesh_axis]
        if held % count:
            raise ValueError(
                f"local: axis {axis} has extent {held}, which its mesh axis of "
                f"{count} positions does not divide; a shard would not be one slice"
            )
        held //= count
    return held


def _inner(shard: ShardLayout, tensor_shape: tuple, mesh_axis: int, axis: int) -> int:
    """How much of tensor *axis* one step of *mesh_axis* steps over.

    The axes cutting one tensor axis are ordered outermost first, so a step of
    one of them clears everything the axes inside it hold.
    """
    extents = _extents(shard)
    product = 1
    for other, target in enumerate(_cutting(shard, tensor_shape)):
        if other > mesh_axis and target == axis and other < len(extents):
            product *= extents[other]
    return product


def _stride(shard: ShardLayout, tensor_shape: tuple, mesh_axis: int, axis: int) -> int:
    """What one step of *mesh_axis* costs the slice's origin, in elements.

    As the device's ``stride<Ax>(sl)`` has it: the extent one instance is left
    of the tensor axis, times what the axes inside this one hold of it, times
    the tensor axis's own stride. Every axis cutting it divides here, unfixed
    or not, because a step must clear what the device will divide too.
    """
    return (
        _narrowed(shard, tensor_shape, axis, range(len(_extents(shard))))
        * _inner(shard, tensor_shape, mesh_axis, axis)
        * _strides(shard, tensor_shape)[axis]
    )


def _positions(shard: ShardLayout, ids: tuple[int | None, ...]) -> dict[int, int]:
    """Each mesh axis's coordinate, for the levels an id was given for.

    A level with no id is left unfixed and names no coordinate, so the axes it
    owns divide nothing and the whole of them stays. Each level reads its own
    id into a coordinate shaped like its own modes, so flattening what the
    levels read together gives one entry per mesh axis.
    """
    mesh_layout = shard.mesh.layout
    stated = mesh_layout.outer if isinstance(mesh_layout, ComposedLayout) else mesh_layout
    read: list = []
    for index in range(rank(stated)):
        arrangement = get(stated, index)
        program_id = ids[index] if index < len(ids) else None
        if program_id is None:
            read.append(repeat_like(arrangement.shape, None))
            continue
        read.append(
            idx2crd(
                program_id,
                tuple(flatten(arrangement.shape)),
                tuple(flatten(arrangement.strides)),
            )
        )
    return {
        axis: position
        for axis, position in enumerate(flatten(tuple(read)))
        if position is not None
    }


def local_layout(
    shard: ShardLayout, tensor_shape: tuple, ids: tuple[int | None, ...]
) -> Layout:
    """What one instance holds: each tensor axis narrowed by what cuts it.

    Narrowed by the axes *ids* names, and no others: an axis nothing cuts is
    held whole, and so is one cut only by a level the placement left unfixed --
    ``cta`` and ``thread`` are the device's to divide. The strides stay the
    whole tensor's, as they do on the device: a slice of it is the same rows,
    the same distance apart.
    """
    dividing = _positions(shard, ids).keys()
    return Layout(
        shape=tuple(
            _narrowed(shard, tensor_shape, axis, dividing)
            for axis in range(len(tensor_shape))
        ),
        strides=_strides(shard, tensor_shape),
    )


def local_layout_and_offset(
    shard: ShardLayout, tensor_shape: tuple, ids: tuple[int | None, ...]
) -> tuple[Layout, int]:
    """That layout, and how far into the tensor this instance's part begins.

    What ``cute::slice_and_offset`` is to a Layout, as on the device: the
    offset is the instance's mesh coordinate dotted with one stride per mesh
    axis. It is counted in the strides ``local_layout`` reports, so a reader
    laid out otherwise would read the wrong elements.
    """
    cutting = _cutting(shard, tensor_shape)
    offset = 0
    for mesh_axis, position in _positions(shard, ids).items():
        axis = cutting[mesh_axis] if mesh_axis < len(cutting) else None
        if axis is None:
            continue
        offset += position * _stride(shard, tensor_shape, mesh_axis, axis)
    return local_layout(shard, tensor_shape, ids), offset


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
    "canonical_shard_layout",
    "shard_layout_local_shape",
    "layout_axis_to_tensor_axis",
    "split_target_axes",
]
