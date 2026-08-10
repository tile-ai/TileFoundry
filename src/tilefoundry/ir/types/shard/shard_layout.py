from __future__ import annotations

from dataclasses import dataclass

from .layout import Layout, LayoutBase
from .layout_algebra import try_c_order_strides
from .mesh import Mesh


class ShardAttr:
    """Base for per-mesh-axis sharding attributes."""


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


def canonical_shard_layout(logical_shape: tuple, mesh: Mesh, attrs: tuple) -> "ShardLayout":
    """Bind logical axes to mesh axes in the canonical factored layout.

    Static splits produce mesh-sized positions plus a residual; dynamic or
    launch-provided single-axis splits remain whole. Attributes are remapped to
    factored positions and strides are rebuilt in C order when static.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    mesh_shape = mesh.layout.shape
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
        if len(splitting_mesh_axes) == 1 and (
            mesh_shape[splitting_mesh_axes[0]] is None or not axis_static
        ):
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
        layout=Layout(shape=layout_shape, strides=try_c_order_strides(layout_shape)),
        attrs=remapped_attrs,
        mesh=mesh,
    )


def shard_layout_local_shape(sl: "ShardLayout") -> tuple[int, ...]:
    """Derive one thread's local shape from a global ``ShardLayout``.

    Each ``Split`` divides its bound layout position by the mesh extent;
    repeated splits multiply their divisors. Other attributes do not consume a
    layout position.

    See [shard §7](docs/spec/shard.md#7-shardlayout).
    """
    mesh_shape = sl.mesh.layout.shape
    local = list(sl.layout.shape)
    for mesh_axis_idx, attr in enumerate(sl.attrs):
        if mesh_axis_idx >= len(mesh_shape):
            break
        if isinstance(attr, Split):
            k = attr.axis
            if not (0 <= k < len(local)):
                continue
            mesh_ext = mesh_shape[mesh_axis_idx]
            if mesh_ext is None:
                local[k] = 1
            elif isinstance(mesh_ext, int) and isinstance(local[k], int):
                if mesh_ext != 0:
                    local[k] //= mesh_ext

    for i, d in enumerate(local):
        if not isinstance(d, int):
            raise ValueError(
                f"shard_layout_local_shape: per-shard dim {i} ({d!r}) is not "
                f"static after sharding; only a launch-provided CTA split "
                f"(per-shard 1) may consume a dynamic axis"
            )
    return tuple(local)


def layout_axis_to_tensor_axis(layout_shape: tuple, tensor_shape: tuple) -> list[int]:
    """Map factored layout positions to their logical tensor axes.

    Positions are consumed left-to-right until their product reaches each
    tensor extent. Singleton tensor axes claim one singleton position; trailing
    positions attach to the final tensor axis.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    from ..shape_helpers import static_dim_value  # noqa: PLC0415 - cycle guard

    result: list[int] = []
    layout_idx = 0
    for t_axis, t_dim in enumerate(tensor_shape):
        t_dim_int = static_dim_value(t_dim)
        if t_dim_int is None:
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


__all__ = [
    "ShardAttr",
    "Split",
    "Partial",
    "Broadcast",
    "Dynamic",
    "S",
    "P",
    "B",
    "ShardLayout",
    "canonical_shard_layout",
    "shard_layout_local_shape",
    "layout_axis_to_tensor_axis",
    "split_target_axes",
]
