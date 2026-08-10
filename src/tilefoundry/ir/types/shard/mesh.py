from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout

_LAUNCH_PROVIDED_TOPOLOGY = "cta"


@dataclass(frozen=True)
class Topology:
    """Name one hardware level and its static or launch-provided size.

    Only CTA topology may use ``None`` for launch-provided extent. See
    [target §4](docs/spec/target.md#4-cudatarget).
    """

    name: str

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
    """Describe hardware levels, logical positions, and axis names.

    A constant slice replaces ``layout`` with a ``ComposedLayout`` whose
    ``offset`` and ``outer`` describe the selected sub-box. It remains a
    compile-time descriptor outside the IR/SSA graph.

    See [shard §5](docs/spec/shard.md#5-mesh).
    """

    topologies: tuple[Topology, ...]
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()

    def __getitem__(self, key) -> "Mesh":
        """Return a constant sub-mesh selected by integers or unit-step slices.

        Missing axes are full slices; integers select extent one. The result
        preserves topology and names while recording the sub-box as a
        ``ComposedLayout``. Dynamic layouts and nested slices are rejected.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        shape = self.layout.shape
        strides = self.layout.strides
        rank = len(shape)
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > rank:
            raise ValueError(f"mesh slice has {len(keys)} indices but the mesh has {rank} axes")
        keys = keys + (slice(None),) * (rank - len(keys))

        sub_shape: list[int] = []
        offset = 0
        for axis, (k, extent, stride) in enumerate(zip(keys, shape, strides)):
            if not isinstance(extent, int) or not isinstance(stride, int):
                raise ValueError(f"cannot slice mesh axis {axis} with a dynamic extent/stride")
            if isinstance(k, int):
                start = k + extent if k < 0 else k
                if not (0 <= start < extent):
                    raise ValueError(
                        f"mesh slice index {k} out of range for axis {axis} (extent {extent})"
                    )
                sel = 1
            elif isinstance(k, slice):
                if k.step not in (None, 1):
                    raise ValueError(f"mesh slice step must be 1 (axis {axis})")
                start = 0 if k.start is None else (k.start + extent if k.start < 0 else k.start)
                stop = extent if k.stop is None else (k.stop + extent if k.stop < 0 else k.stop)
                if not (0 <= start <= stop <= extent):
                    raise ValueError(
                        f"mesh slice {k.start}:{k.stop} out of range for axis {axis} (extent {extent})"
                    )
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
