from __future__ import annotations

from dataclasses import dataclass

from .layout import ComposedLayout, Layout
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
class ShardLayout:
    layout: "Layout | ComposedLayout"
    attrs: tuple[ShardAttr, ...]
    mesh: Mesh


def partial_reductions(layout: object) -> frozenset[str]:
    """The distinct ``Partial.reduction`` values carried by *layout*'s attrs,
    or an empty set when *layout* isn't a ``ShardLayout`` or carries none."""
    if not isinstance(layout, ShardLayout):
        return frozenset()
    return frozenset(a.reduction for a in layout.attrs if isinstance(a, Partial))


def shard_layout_local_shape(sl: "ShardLayout") -> tuple[int, ...]:
    """Derive the per-thread local cute shape from a global
    ``ShardLayout``.

. ``sl.layout.shape`` is the global / unsharded
    cute layout shape; this helper divides each cute dim by its bound
    ``Split`` mesh extent to produce the per-thread local cute shape.

    - ``Split(k)`` on mesh axis ``i`` divides ``layout.shape[k]`` by
      ``mesh.layout.shape[i]``.
    - ``Partial`` / ``Broadcast`` / ``Dynamic`` attrs do not consume any
      cute dim. A ``Partial`` mesh axis holds an un-reduced partial of the
      full value, so each shard keeps the full local cute shape.

    Multiple ``Split`` attrs on the same cute dim multiply their
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
    """Map each cute ``Layout`` position to the logical tensor axis it lives
    within.

    Convention: cute layout positions are consumed left-to-right; each tensor
    axis ``k`` claims as many cute positions as needed to accumulate to
    ``tensor_shape[k]``. Trailing cute positions (if any) attach to the last
    tensor axis. Singleton tensor axes (``tensor_shape[k] == 1``) claim exactly
    one cute position (which must also be size 1 by construction).

    Example — ``tensor_shape=(1, 1536)`` + ``layout_shape=(1, 6, 32, 8)``::

        cute pos 0 (size 1)  -> tensor axis 0
        cute pos 1 (size 6)  -> tensor axis 1 (running 6)
        cute pos 2 (size 32) -> tensor axis 1 (running 192)
        cute pos 3 (size 8)  -> tensor axis 1 (running 1536)
    """
    result: list[int] = []
    layout_idx = 0
    for t_axis, t_dim in enumerate(tensor_shape):
        try:
            t_dim_int = int(t_dim.value) if hasattr(t_dim, "value") else int(t_dim)
        except (TypeError, ValueError):
            # Symbolic dim — attach all remaining cute positions here and stop.
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
            try:
                sh = int(layout_shape[layout_idx])
            except (TypeError, ValueError):
                sh = 1
            running *= sh
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
    "shard_layout_local_shape",
    "layout_axis_to_tensor_axis",
    "partial_reductions",
    "split_target_axes",
]
