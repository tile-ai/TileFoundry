"""Say where a function's values live, and which part one participant owns.

Addresses are decided once, by the placement memory already ran; this reads that
decision back. ``BufferPlan.project`` narrows what a participant sees of a value
to an origin and a domain and leaves the buffer alone, because a shard is a view
of one allocation and not an allocation of its own.

Those coordinates are the layout's positions, where an access pattern's image
lands: a participant given the finer of two positions of one factored axis owns
a stride of that axis, which no origin in that axis can describe.
"""

from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.ir.core import get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import shard_layout_of
from tilefoundry.ir.types.shard.layout import ComposedLayout
from tilefoundry.ir.types.shard.layout_algebra import (
    apply,
    idx2crd,
    prefix_product,
    project,
    size,
)
from tilefoundry.ir.types.shard.shard_layout import Split

from .errors import AnalysisError
from .metadata import BufferAllocationMetadata, BufferRef
from .walk import postorder


@dataclass(frozen=True)
class PlannedBuffer:
    """One value field's buffer, and the part of it a reader owns.

    ``expr_id`` is the value this entry belongs to, which is how a reader that
    holds the value finds it; ``binding`` is for saying which one it was.
    ``origin`` and ``extents`` are stated in the buffer's layout positions. A
    value with no layout has one position per logical axis. An unprojected plan
    owns everything, so the origin is zero and the extents are the whole space.
    """

    expr_id: int
    binding: str
    field: int
    ref: BufferRef
    origin: tuple[int, ...]
    extents: tuple[int, ...]

    @property
    def domain(self) -> isl.set:
        """The owned coordinates, as a set to intersect an access with."""
        names = [f"i{axis}" for axis in range(len(self.extents))]
        if not names:
            return isl.set("{ [] }")
        guards = " and ".join(
            f"{start} <= {name} < {start + extent}"
            for name, start, extent in zip(names, self.origin, self.extents)
        )
        return isl.set(f"{{ [{', '.join(names)}] : {guards} }}")


@dataclass(frozen=True)
class BufferPlan:
    """Every addressed value field of one function, at one topology level."""

    level: str | None
    buffers: tuple[PlannedBuffer, ...] = ()

    def of(self, expr) -> tuple[PlannedBuffer, ...]:
        """The fields planned for one value, in the order its type states them.

        A value with no entry has none rather than an empty one: nothing is not
        the same answer as no bytes, and a reader that needs an address has to
        be able to tell them apart.
        """
        return tuple(item for item in self.buffers if item.expr_id == id(expr))

    def owned(self, expr, participant: int, field: int = 0) -> "PlannedBuffer | None":
        """What *participant* owns of one field of one value.

        The same answer ``project`` gives for that field, without narrowing the
        whole function to ask about one buffer.
        """
        item = next(
            (
                held
                for held in self.buffers
                if held.expr_id == id(expr) and held.field == field
            ),
            None,
        )
        return None if item is None else _owned(item, participant, self.level)

    def project(self, participant: int) -> "BufferPlan":
        """The same buffers, narrowed to what *participant* owns of each.

        A value this participant holds no part of is left out: the plan says
        what can be seen, and nothing is not a part.
        """
        narrowed = []
        for item in self.buffers:
            owned = _owned(item, participant, self.level)
            if owned is not None:
                narrowed.append(owned)
        return BufferPlan(level=self.level, buffers=tuple(narrowed))


def build_buffer_plan(fn: Function, level: str | None = None) -> BufferPlan:
    """Read back the addresses *fn*'s placement decided, as one plan."""
    planned: list[PlannedBuffer] = []
    for expr in _values_of(fn):
        record = get_metadata(expr, BufferAllocationMetadata)
        if record is None:
            continue
        binding = _binding_of(expr)
        for field, ref in enumerate(record.fields):
            whole = _position_shape(ref)
            planned.append(
                PlannedBuffer(
                    expr_id=id(expr),
                    binding=binding,
                    field=field,
                    ref=ref,
                    origin=(0,) * len(whole),
                    extents=whole,
                )
            )
    return BufferPlan(level=level, buffers=tuple(planned))


def _position_shape(ref: BufferRef) -> tuple:
    """The coordinates one buffer is addressed by: its layout's positions."""
    shape = getattr(ref.layout, "shape", None)
    return tuple(ref.shape if shape is None else shape)


def _values_of(fn: Function) -> "list":
    """Every value of *fn*, each once, parameters first.

    A parameter is named again wherever it is read, and one value has one
    buffer, so the walk states each the first time it reaches it.
    """
    seen: set[int] = set()
    walk = [*fn.params, *(postorder(fn.body) if fn.body is not None else ())]
    return [expr for expr in walk if id(expr) not in seen and not seen.add(id(expr))]


def _binding_of(expr) -> str:
    """How this value is named, matching the lifetimes of the same function."""
    from tilefoundry.ir.core import Var, binding_name  # noqa: PLC0415 - cycle guard

    if isinstance(expr, Var):
        return expr.name
    return binding_name(expr) or ""


def _owned(item: PlannedBuffer, participant: int, level: str | None) -> PlannedBuffer | None:
    """Which coordinates of one buffer *participant* holds.

    A value with no distribution at this level is held whole by everyone that
    reads it. One distributed at another level is not this participant's to
    narrow, so it is left whole too, and the level that owns it narrows it.
    """
    layout = shard_layout_of(item.ref.layout)
    if layout is None or level is None:
        return item
    names = tuple(topology.name for topology in layout.mesh.topologies)
    if names != (level,):
        return item
    coordinate = _mesh_coordinate(layout.mesh, participant)
    if coordinate is None:
        return None
    origin, extents = _block(item, layout, coordinate)
    return PlannedBuffer(
        expr_id=item.expr_id,
        binding=item.binding,
        field=item.field,
        ref=item.ref,
        origin=origin,
        extents=extents,
    )


def _mesh_coordinate(mesh, participant: int) -> tuple[int, ...] | None:
    """The mesh coordinate that names *participant*, or None if none does."""
    layout = mesh.layout
    if isinstance(layout, ComposedLayout):
        return project(layout, participant)
    shape = tuple(layout.shape)
    for coordinate in range(size(layout)):
        if apply(layout, coordinate) == participant:
            return idx2crd(coordinate, shape, prefix_product(shape))
    return None


def _block(
    item: PlannedBuffer, layout, coordinate: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The run of each layout position this participant holds.

    A position a mesh axis splits is held one slice at a time, and the slice a
    participant holds is the one its mesh coordinate names. Two mesh axes on one
    position name it together, coarsest first, which is the same reading the
    local shape gives when it divides that position twice.
    """
    positions = tuple(layout.layout.shape)
    mesh_shape = tuple(layout.mesh.layout.shape)
    local = list(positions)
    index = [0] * len(positions)
    for mesh_axis, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split) or mesh_axis >= len(mesh_shape):
            continue
        held = attr.axis
        if not 0 <= held < len(local):
            continue
        extent, position = mesh_shape[mesh_axis], local[held]
        if not isinstance(extent, int) or not isinstance(position, int):
            raise AnalysisError(
                f"{item.binding!r} is split by a mesh axis of extent {extent!r} "
                f"over a layout position of extent {position!r}; bind both before "
                "asking what one participant owns"
            )
        if extent == 0 or position % extent:
            raise AnalysisError(
                f"{item.binding!r} divides layout position {held} of extent "
                f"{position} across {extent} participants, which does not divide it"
            )
        local[held] = position // extent
        index[held] = index[held] * extent + coordinate[mesh_axis]
    origin = tuple(start * held for start, held in zip(index, local))
    return origin, tuple(local)


__all__ = ["BufferPlan", "PlannedBuffer", "build_buffer_plan"]
