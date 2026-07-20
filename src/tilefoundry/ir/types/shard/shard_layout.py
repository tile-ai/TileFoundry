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
    """Build the canonical ``ShardLayout`` (``docs/spec/shard.md`` §7.1.1)
    binding ``attrs`` (one entry per mesh axis; each ``Split`` names a
    ``logical_shape`` axis) to ``mesh``.

    Every logical axis split by one or more mesh axes is factored, in
    mesh-axis order, into one position per splitting mesh axis (sized to
    that axis's extent) plus a residual position
    (``logical_size // Π(extents)``, omitted when 1, an error when the
    division is not exact) — every ``Split``-bound position in the
    resulting ``Layout`` ends up sized exactly to its mesh extent
    (``G[k] == mesh.shape[a]``), the §7.1.1 canonical form, *even when a
    single mesh axis splits the axis*. A single mesh-axis split is instead
    kept whole (one Split-bound position, no residual) when it cannot be
    factored into a static extent + residual: either the mesh extent is
    launch-provided (``mesh.shape[a] is None``, only a ``cta`` topology, its
    runtime count unknown at compile time) or the logical axis size is itself
    dynamic (the residual ``size // extent`` would be symbolic) — each shard
    then owns one runtime-determined 1/extent slice. (A dynamic logical axis
    split by *several* mesh axes cannot be represented and is an error.)
    ``Split`` attrs are remapped from the logical axis to that
    position; ``Broadcast`` / ``Partial`` / ``Dynamic`` pass through
    unchanged. Strides are freshly built C-order (``None`` when the
    resulting shape is dynamic).

    This is the single canonicalizer for a §7.1.1 layout: both
    ``make_shard_tensor_type`` (a from-scratch sharding) and
    ``derive_output_shard_layout``'s synthesis fallback (a propagated one)
    call it, so a layout built by either for the same logical sharding
    always compares structurally equal.
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
        # A single mesh-axis split keeps the whole logical axis as one
        # Split-bound position when it cannot be factored into a static extent
        # + residual: either the mesh extent is launch-provided (dynamic,
        # ``None``) or the logical axis size is itself dynamic (the residual
        # ``size // extent`` would be symbolic). Each shard then owns a
        # runtime-determined 1/extent slice. This matches the pre-canonicalizer
        # single-split synthesis in ``derive_output_shard_layout`` (a dynamic
        # axis split by *several* mesh axes still errors, below).
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
    """Derive the per-thread local layout shape from a global
    ``ShardLayout``.

. ``sl.layout.shape`` is the global / unsharded
    layout shape; this helper divides each layout dim by its bound
    ``Split`` mesh extent to produce the per-thread local layout shape.

    - ``Split(k)`` on mesh axis ``i`` divides ``layout.shape[k]`` by
      ``mesh.layout.shape[i]``.
    - ``Partial`` / ``Broadcast`` / ``Dynamic`` attrs do not consume any
      layout dim. A ``Partial`` mesh axis holds an un-reduced partial of the
      full value, so each shard keeps the full local layout shape.

    Multiple ``Split`` attrs on the same layout dim multiply their
    divisors together.
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
                # Launch-provided (dynamic) CTA extent: each CTA owns one
                # slice, so the per-shard extent on this axis is a static 1.
                local[k] = 1
            elif isinstance(mesh_ext, int) and isinstance(local[k], int):
                if mesh_ext != 0:
                    local[k] //= mesh_ext
    # The per-shard shape sizes a static (register / shared) buffer, so every
    # entry must be a static int once the splits are applied. A dim left
    # dynamic — a non-split dynamic axis, or a dynamic mesh extent other than
    # the launch-provided CTA count — cannot size such a buffer.
    for i, d in enumerate(local):
        if not isinstance(d, int):
            raise ValueError(
                f"shard_layout_local_shape: per-shard dim {i} ({d!r}) is not "
                f"static after sharding; only a launch-provided CTA split "
                f"(per-shard 1) may consume a dynamic axis"
            )
    return tuple(local)


def layout_axis_to_tensor_axis(
    layout_shape: tuple, tensor_shape: tuple
) -> list[int]:
    """Map each ``Layout`` position to the logical tensor axis it lives
    within.

    Convention: layout positions are consumed left-to-right; each tensor
    axis ``k`` claims as many layout positions as needed to accumulate to
    ``tensor_shape[k]``. Trailing layout positions (if any) attach to the last
    tensor axis. Singleton tensor axes (``tensor_shape[k] == 1``) claim exactly
    one layout position (which must also be size 1 by construction).

    Example — ``tensor_shape=(1, 1536)`` + ``layout_shape=(1, 6, 32, 8)``::

        layout pos 0 (size 1)  -> tensor axis 0
        layout pos 1 (size 6)  -> tensor axis 1 (running 6)
        layout pos 2 (size 32) -> tensor axis 1 (running 192)
        layout pos 3 (size 8)  -> tensor axis 1 (running 1536)
    """
    from ..shape_helpers import static_dim_value  # noqa: PLC0415 - cycle guard

    result: list[int] = []
    layout_idx = 0
    for t_axis, t_dim in enumerate(tensor_shape):
        t_dim_int = static_dim_value(t_dim)
        if t_dim_int is None:
            # Symbolic dim — attach all remaining layout positions here and stop.
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
    """Per mesh axis in ``sl.attrs``, the logical ``tensor_shape`` axis its
    ``Split`` targets (``None`` for a non-``Split`` attr)."""
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
