from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout

# The local validation exception is defined by docs/spec/target.md § Topology levels.
_LAUNCH_PROVIDED_TOPOLOGY = "cta"


@dataclass(frozen=True)
class Topology:
    name: str
    # ``None`` reaches the local launch-provided extent check.
    size: "ShapeDim | None"

    def __post_init__(self) -> None:
        if self.size is None and self.name != _LAUNCH_PROVIDED_TOPOLOGY:
            raise ValueError(
                f"Topology {self.name!r}: only a {_LAUNCH_PROVIDED_TOPOLOGY!r} "
                "topology may have a launch-provided (None) extent. The rule: "
                "tilefoundry spec target topology-levels"
            )


@dataclass(frozen=True)
class Mesh:
    """Logical mesh record: hardware levels, logical positions, and axis names.

    ``layout`` is a plain ``Layout`` for an un-sliced mesh. A constant slice
    (``m[1:3, :]``, used by ``T.sync``) replaces ``layout`` with a
    ``ComposedLayout`` recording the participating sub-box
    (``image(c) = offset + outer(c)``, the affine "mesh scope" case): the
    slice origin in ``offset`` and the selected per-axis extents over the
    parent strides in ``outer`` (identity ``inner``). The slice stays a
    compile-time descriptor; it never enters the IR/SSA graph. The enclosing
    full mesh supplies the parent shape when a slice is verified.
    """
    topologies: tuple[Topology, ...]
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()

    def __getitem__(self, key) -> "Mesh":
        """Slice this mesh into a constant sub-mesh (used by ``T.sync(m[...])``).

        ``key`` is an int / slice, or a tuple of them, one per layout axis
        (missing trailing axes default to a full ``:`` slice). An int selects a
        single coordinate (extent 1); a slice selects ``[start, stop)`` with no
        step. The sub-mesh keeps the parent topology and names; its ``layout``
        becomes a ``ComposedLayout`` recording the participating sub-box:
        ``outer`` is the selected per-axis extents over the parent strides and
        ``offset = Σ start_i · stride_i`` is the linear thread index of the
        slice origin (``inner`` identity). Only static extents/strides may be
        sliced; nesting a slice raises.
        """
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        shape = self.layout.shape
        strides = self.layout.strides
        rank = len(shape)
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > rank:
            raise ValueError(
                f"mesh slice has {len(keys)} indices but the mesh has {rank} axes"
            )
        keys = keys + (slice(None),) * (rank - len(keys))

        sub_shape: list[int] = []
        offset = 0
        for axis, (k, extent, stride) in enumerate(zip(keys, shape, strides)):
            if not isinstance(extent, int) or not isinstance(stride, int):
                raise ValueError(
                    f"cannot slice mesh axis {axis} with a dynamic extent/stride"
                )
            if isinstance(k, int):
                start = k + extent if k < 0 else k
                if not (0 <= start < extent):
                    raise ValueError(f"mesh slice index {k} out of range for axis {axis} (extent {extent})")
                sel = 1
            elif isinstance(k, slice):
                if k.step not in (None, 1):
                    raise ValueError(f"mesh slice step must be 1 (axis {axis})")
                start = 0 if k.start is None else (k.start + extent if k.start < 0 else k.start)
                stop = extent if k.stop is None else (k.stop + extent if k.stop < 0 else k.stop)
                if not (0 <= start <= stop <= extent):
                    raise ValueError(f"mesh slice {k.start}:{k.stop} out of range for axis {axis} (extent {extent})")
                sel = stop - start
                if sel == 0:
                    raise ValueError(f"mesh slice selects an empty range on axis {axis}")
            else:
                raise ValueError(f"mesh slice index must be int or slice, got {type(k).__name__}")
            offset += start * stride
            sub_shape.append(sel)

        sliced = ComposedLayout(
            inner=None,
            offset=offset,
            outer=Layout(shape=tuple(sub_shape), strides=strides),
        )
        return Mesh(
            topologies=self.topologies,
            layout=sliced,
            names=self.names,
        )


__all__ = ["Topology", "Mesh"]
