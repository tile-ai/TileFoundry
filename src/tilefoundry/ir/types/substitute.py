"""Replace selected symbolic dimensions with concrete extents.

Substitution is partial: unbound dimensions remain symbolic. This module lives
beside tensor types to avoid making the dimension vocabulary depend on them.

See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

from collections.abc import Mapping

from tilefoundry.ir.core.expr import Call, Constant

from .dim import _DIM_OP_TYPES, DimVar, simplify_dim
from .tensor_type import TensorType, TupleType, Type


class DimSubstitutionError(ValueError):
    """A dimension was bound to something its declaration does not admit."""


def dim_vars_in(value: object) -> tuple[str, ...]:
    """Every distinct `DimVar` name reachable from *value*, in first-seen order.

    Callers use this to refuse a binding for a dimension that is not there. A
    name nobody uses is a caller who believes they specialised something, and
    silently substituting nothing would let them keep believing it.
    """
    return tuple(dim_vars_by_name(value))


def dim_vars_by_name(value: object) -> dict[str, "DimVar"]:
    """Every distinct `DimVar` reachable from *value*, by name, first-seen first.

    The declarations themselves, for a caller that needs to restate them rather
    than only check whether they are there -- printing a program back as source
    has to emit the bounds a name was declared with, and a name alone cannot say
    what they were.
    """
    found: dict[str, "DimVar"] = {}
    _collect(value, found)
    return found


def _shard_types() -> tuple[type, ...]:
    """Shard geometry descriptors, imported at call time to avoid a cycle."""
    from .shard.layout import ComposedLayout, Layout  # noqa: PLC0415
    from .shard.mesh import Mesh, Topology  # noqa: PLC0415
    from .shard.shard_layout import ShardLayout  # noqa: PLC0415

    return (Layout, ComposedLayout, ShardLayout, Mesh, Topology)


def _collect(value: object, found: dict[str, "DimVar"]) -> None:
    Layout, ComposedLayout, ShardLayout, Mesh, Topology = _shard_types()
    if isinstance(value, TensorType):
        for entry in value.shape:
            _collect(entry, found)

        _collect_layout(value.layout, found)
        return
    if isinstance(value, TupleType):
        for field in value.fields:
            _collect(field, found)
        return
    if isinstance(value, DimVar):
        found[value.name] = value
        return
    if isinstance(value, tuple):
        for entry in value:
            _collect(entry, found)
        return
    if isinstance(value, Topology):
        _collect(value.size, found)
        return
    if isinstance(value, Mesh):
        _collect(value.topologies, found)
        _collect_layout(value.layout, found)
        return
    if isinstance(value, (Layout, ComposedLayout, ShardLayout)):
        _collect_layout(value, found)
        return
    if isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES):
        for arg in value.args:
            _collect(arg, found)


def _collect_layout(layout: object, found: dict[str, "DimVar"]) -> None:
    if layout is None:
        return
    Layout, ComposedLayout, ShardLayout, _, _ = _shard_types()
    if isinstance(layout, ShardLayout):
        _collect_layout(layout.layout, found)
        _collect(layout.mesh, found)
        return
    if isinstance(layout, ComposedLayout):
        _collect_layout(layout.outer, found)
        _collect_layout(layout.inner, found)
        _collect(layout.offset, found)
        return
    if isinstance(layout, Layout):
        _collect(layout.shape, found)
        if layout.strides is not None:
            _collect(layout.strides, found)


def substitute_dims(value: Type, bindings: Mapping[str, int]) -> Type:
    """*value* with every bound `DimVar` replaced by its chosen extent.

    Unbound dimensions are left as they are. Arithmetic over dimensions folds
    as soon as its operands are known, through the same construction-time
    folding that built it, so a shape written `ctx_len // 8` becomes an integer
    rather than a division that survives into the analysis.
    """
    if isinstance(value, TensorType):
        shape = tuple(substitute_shape_dim(entry, bindings) for entry in value.shape)
        layout = substitute_layout_dims(value.layout, bindings)
        if shape == value.shape and layout is value.layout:
            return value
        return TensorType(
            shape=shape,
            dtype=value.dtype,
            layout=layout,
            storage=value.storage,
        )
    if isinstance(value, TupleType):
        fields = tuple(substitute_dims(field, bindings) for field in value.fields)
        if fields == value.fields:
            return value
        return TupleType(fields=fields)
    return value


def substitute_layout_dims(layout: object, bindings: Mapping[str, int]) -> object:
    """*layout* with its bound dimensions replaced.

    A layout restates the shape it describes, so leaving it behind produces a
    type whose shape is a number and whose layout is still a range -- concrete
    to anything that reads the shape, and not to anything that reads the layout.
    A shard layout's Mesh is part of that geometry: its layout and topology
    extents may be derived from the same dimensions as the tensor shape.
    """
    if layout is None:
        return layout
    Layout, ComposedLayout, ShardLayout, _, _ = _shard_types()
    if isinstance(layout, ShardLayout):
        inner = substitute_layout_dims(layout.layout, bindings)
        mesh = substitute_mesh_dims(layout.mesh, bindings)
        if inner is layout.layout and mesh is layout.mesh:
            return layout
        return ShardLayout(layout=inner, attrs=layout.attrs, mesh=mesh)
    if isinstance(layout, ComposedLayout):
        outer = substitute_layout_dims(layout.outer, bindings)
        inner = substitute_layout_dims(layout.inner, bindings)
        offset = substitute_shape_dim(layout.offset, bindings)
        if outer is layout.outer and inner is layout.inner and offset == layout.offset:
            return layout
        return ComposedLayout(inner=inner, offset=offset, outer=outer)
    if isinstance(layout, Layout):
        shape = _substitute_nested(layout.shape, bindings)
        strides = None if layout.strides is None else _substitute_nested(layout.strides, bindings)
        if shape == layout.shape and strides == layout.strides:
            return layout
        return Layout(shape=shape, strides=strides)
    return layout


def substitute_topology_dims(topology: object, bindings: Mapping[str, int]) -> object:
    """*topology* with its extent rebuilt from *bindings*."""
    _, _, _, _, Topology = _shard_types()
    if not isinstance(topology, Topology):
        return topology
    size = substitute_shape_dim(topology.size, bindings)
    if size == topology.size:
        return topology
    return Topology(topology.name, size)


def substitute_mesh_dims(mesh: object, bindings: Mapping[str, int]) -> object:
    """*mesh* with layout and topology dimensions rebuilt together."""
    _, _, _, Mesh, _ = _shard_types()
    if not isinstance(mesh, Mesh):
        return mesh
    topologies = tuple(substitute_topology_dims(item, bindings) for item in mesh.topologies)
    layout = substitute_layout_dims(mesh.layout, bindings)
    if topologies == mesh.topologies and layout is mesh.layout:
        return mesh
    return Mesh(topologies=topologies, layout=layout, names=mesh.names)


def _substitute_nested(entries: tuple, bindings: Mapping[str, int]) -> tuple:
    """A possibly nested tuple of shape entries, substituted throughout.

    A layout's shape is hierarchical, so an entry may itself be a tuple; and it
    may be `None`, which states an axis whose extent this layout does not fix.
    """
    return tuple(
        _substitute_nested(entry, bindings)
        if isinstance(entry, tuple)
        else (None if entry is None else substitute_shape_dim(entry, bindings))
        for entry in entries
    )


def substitute_shape_dim(entry: object, bindings: Mapping[str, int]) -> object:
    """One shape entry with its bound dimensions replaced.

    Exposed separately because a dimension is not only found in a type: a loop
    states its own extent, step and start as shape entries, and those are not
    reachable from any type the loop's body carries.
    """
    if isinstance(entry, bool):
        raise DimSubstitutionError(f"{entry!r} is not a shape entry")
    if isinstance(entry, int):
        return entry
    if isinstance(entry, DimVar):
        if entry.name not in bindings:
            return entry
        return _checked(entry, bindings[entry.name])
    if isinstance(entry, Constant):
        return entry
    if isinstance(entry, Call) and isinstance(entry.target, _DIM_OP_TYPES):
        args = tuple(substitute_shape_dim(arg, bindings) for arg in entry.args)
        if args == tuple(entry.args):
            return entry
        folded = simplify_dim(type(entry.target), args)

        if isinstance(folded, Constant) and isinstance(folded.value, int):
            return int(folded.value)
        return folded
    return entry


def _checked(variable: DimVar, extent: int) -> int:
    """*extent*, once the declaration is known to admit it.

    The bounds are half-open, matching how a specialisation states the range it
    covers, so the two cannot disagree about which side an endpoint falls on.
    """
    if isinstance(extent, bool) or not isinstance(extent, int):
        raise DimSubstitutionError(
            f"dimension {variable.name!r} takes an integer extent, got {extent!r}"
        )
    if not variable.lo <= extent < variable.hi:
        raise DimSubstitutionError(
            f"dimension {variable.name!r} was declared over "
            f"[{variable.lo}, {variable.hi}) and cannot take {extent}"
        )
    return extent


def has_symbolic_dims(value: object) -> bool:
    """Whether anything reachable from *value* is not a static dimension."""
    Layout, ComposedLayout, ShardLayout, Mesh, Topology = _shard_types()
    if isinstance(value, DimVar):
        return True
    if isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES):
        return True
    if isinstance(value, TensorType):
        return has_symbolic_dims(value.shape) or has_symbolic_dims(value.layout)
    if isinstance(value, TupleType):
        return any(has_symbolic_dims(field) for field in value.fields)
    if isinstance(value, Topology):
        return has_symbolic_dims(value.size)
    if isinstance(value, Mesh):
        return has_symbolic_dims(value.topologies) or has_symbolic_dims(value.layout)
    if isinstance(value, ShardLayout):
        return has_symbolic_dims(value.layout) or has_symbolic_dims(value.mesh)
    if isinstance(value, ComposedLayout):
        return any(
            has_symbolic_dims(entry)
            for entry in (value.inner, value.offset, value.outer)
        )
    if isinstance(value, Layout):
        return has_symbolic_dims(value.shape) or has_symbolic_dims(value.strides)
    if isinstance(value, tuple):
        return any(has_symbolic_dims(entry) for entry in value)
    return False


__all__ = [
    "DimSubstitutionError",
    "dim_vars_by_name",
    "dim_vars_in",
    "has_symbolic_dims",
    "substitute_dims",
    "substitute_layout_dims",
    "substitute_mesh_dims",
    "substitute_shape_dim",
    "substitute_topology_dims",
]
