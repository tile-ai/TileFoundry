from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout

# A ``cta`` topology may declare ``size=None`` to mean its extent is provided by
# the host launch (a runtime grid); only that topology kind supports it.
_LAUNCH_PROVIDED_TOPOLOGY = "cta"


@dataclass(frozen=True)
class Topology:
    name: str
    # ``int`` for a static extent, a scalar ``ShapeDim`` expression, or ``None``
    # for a launch-provided (dynamic) ``cta`` extent.
    size: "ShapeDim | None"

    def __post_init__(self) -> None:
        if self.size is None and self.name != _LAUNCH_PROVIDED_TOPOLOGY:
            raise ValueError(
                f"Topology {self.name!r}: only a {_LAUNCH_PROVIDED_TOPOLOGY!r} "
                f"topology may have a launch-provided (None) extent"
            )


@dataclass(frozen=True)
class MeshAxis:
    mesh: "Mesh"
    index: int
    size: "ShapeDim"


@dataclass(frozen=True)
class Mesh:
    """Logical mesh — one or more Topologies subdivided by a layout.

    multi-Topology + named-axes Meshes are first-class:

    - The Topology product gives the mesh's *domain* (e.g.
      ``warp(4) × thread(32) = 128`` threads).
    - ``layout`` (a ``Layout``) subdivides the domain into logical
      axes; the layout shape's product must equal the domain size.
    - ``names`` labels each layout axis.

    ``Mesh.topology`` is the primary (first) Topology; the full tuple is
    ``topologies``. ``__post_init__`` normalizes ``topologies`` to a
    non-empty ``tuple[Topology, ...]`` (a single Topology or a non-empty
    tuple accepted; a raw string rejected) and coerces a ``(s0, s1, ...)``
    layout shorthand to a C-order ``Layout`` — see ``docs/spec/shard.md`` §5.

    ``layout`` is a plain ``Layout`` for an un-sliced mesh. A constant slice
    (``m[1:3, :]``, used by ``T.sync``) replaces ``layout`` with a
    ``ComposedLayout`` recording the participating sub-box
    (``image(c) = offset + outer(c)``, the affine "mesh scope" case): the
    slice origin in ``offset`` and the selected per-axis extents over the
    parent strides in ``outer`` (identity ``inner``). The slice stays a
    compile-time descriptor; it never enters the IR/SSA graph. The enclosing
    full mesh supplies the parent shape when a slice is verified.
    """
    topology: Topology
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()
    topologies: tuple[Topology, ...] = ()

    def __post_init__(self) -> None:
        # Normalize ``topology`` (list / tuple of Topology accepted at
        # construction time) into the primary + the full, non-empty tuple.
        topo = self.topology
        if isinstance(topo, str):
            raise TypeError(
                f"Mesh.topology must be a Topology (or a non-empty tuple of "
                f"them), got a raw string {topo!r}; resolve it to a Topology "
                f"first (e.g. via make_mesh)"
            )
        if isinstance(topo, (list, tuple)):
            if not topo:
                raise ValueError("Mesh.topology tuple must be non-empty")
            primary = topo[0]
            full = tuple(topo)
            object.__setattr__(self, "topology", primary)
            object.__setattr__(self, "topologies", full)
        elif isinstance(topo, Topology):
            if not self.topologies:
                object.__setattr__(self, "topologies", (topo,))
        else:
            raise TypeError(
                f"Mesh.topology must be a Topology or a tuple of Topology, "
                f"got {type(topo).__name__}"
            )

        # Coerce raw ``(s0, s1, ...)`` layout into a ``Layout`` with
        # C-order strides.
        ly = self.layout
        if isinstance(ly, tuple):
            if not all(isinstance(s, int) for s in ly):
                raise ValueError(
                    "dynamic layout requires explicit strides: a layout "
                    "shorthand tuple with a non-integer (dynamic) extent cannot "
                    "auto-derive C-order strides; pass an explicit Layout"
                )
            from .layout_algebra import c_order_strides  # noqa: PLC0415 - cycle guard
            object.__setattr__(
                self, "layout", Layout(shape=ly, strides=c_order_strides(ly))
            )

    def topology_domain(self) -> "int | None":
        """Total device-domain extent = product of every ``Topology`` in
        ``topologies``; ``None`` if any extent is dynamic (launch-provided)."""
        domain = 1
        for t in self.topologies:
            if not isinstance(t.size, int):
                return None
            domain *= t.size
        return domain


    @property
    def axes(self) -> tuple[MeshAxis, ...]:
        return tuple(
            MeshAxis(mesh=self, index=i, size=sz) for i, sz in enumerate(self.layout.shape)
        )

    @property
    def x(self) -> MeshAxis:
        return self.axes[0]

    @property
    def y(self) -> MeshAxis:
        return self.axes[1]

    @property
    def z(self) -> MeshAxis:
        return self.axes[2]

    def axis_named(self, name: str) -> MeshAxis | None:
        """Find a named axis, or None."""
        for i, n in enumerate(self.names):
            if n == name:
                return MeshAxis(mesh=self, index=i, size=self.layout.shape[i])
        return None

    def __getattr__(self, name: str) -> MeshAxis:
        """Expose declared axis names with the same surface the parser accepts."""
        axis = self.axis_named(name)
        if axis is not None:
            return axis
        raise AttributeError(f"Mesh has no axis named {name!r}")

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
            topology=self.topology,
            layout=sliced,
            names=self.names,
            topologies=self.topologies,
        )


__all__ = ["Topology", "MeshAxis", "Mesh"]
