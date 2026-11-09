"""Replace selected symbolic dimensions with concrete extents.

Substitution is partial: unbound dimensions remain symbolic.

See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

from collections.abc import Mapping

from tilefoundry.ir import expr, types

from .isl_interop import normalize
from .simplify import KIND_OF, combine

__all__ = [
    "SubstitutionError",
    "canonicalize_dims",
    "canonicalize_topology_dims",
    "collect_vars",
    "has_symbolic_dims",
    "substitute_dims",
    "substitute_layout_dims",
    "substitute_mesh_dims",
    "substitute_shape_entry",
    "substitute_topology_dims",
    "var_names",
    "vars_by_name",
]


class SubstitutionError(ValueError):
    """A dimension was bound to something its declaration does not admit."""


class ShapeEntryMutator:
    """Rebuild one shape entry with its bound dimension variables replaced.

    A node is rebuilt once, so a shared operand stays shared in the result.
    """

    def __init__(self, bindings: Mapping[str, int]) -> None:
        """Substitute the variables named in *bindings*."""
        self.bindings = bindings
        self.memo: dict[int, tuple[object, object]] = {}

    def visit(self, value):
        """The rebuilt *value*, computed once per node."""
        if isinstance(value, bool):
            raise SubstitutionError(f"{value!r} is not a shape entry")
        hit = self.memo.get(id(value))
        if hit is not None:
            return hit[1]
        result = self.rewrite(value)
        self.memo[id(value)] = (value, result)
        return result

    def rewrite(self, value):
        """Replace a bound ``DimVar``, rebuild a dimension ``Call`` whose operands changed."""
        if isinstance(value, expr.DimVar):
            if value.name not in self.bindings:
                return value
            return checked(value, self.bindings[value.name])
        if isinstance(value, expr.Call) and isinstance(value.target, expr.DIM_OP_TYPES):
            args = tuple(self.visit(arg) for arg in value.args)
            if args == value.args:
                return value
            folded = normalize(combine(KIND_OF[type(value.target)], args))
            if isinstance(folded, expr.Constant) and isinstance(folded.value, int):
                return int(folded.value)
            return folded
        return value


def var_names(value: object) -> tuple[str, ...]:
    """Every distinct `DimVar` name reachable from *value*, in first-seen order.

    Callers use this to refuse a binding for a dimension that is not there. A
    name nobody uses is a caller who believes they specialised something, and
    silently substituting nothing would let them keep believing it.
    """
    return tuple(vars_by_name(value))


def vars_by_name(value: object) -> dict[str, "expr.DimVar"]:
    """Every distinct `DimVar` reachable from *value*, by name, first-seen first.

    The declarations themselves, for a caller that needs to restate them rather
    than only check whether they are there -- printing a program back as source
    has to emit the bounds a name was declared with, and a name alone cannot say
    what they were.
    """
    found: dict[str, "expr.DimVar"] = {}
    collect_vars(value, found)
    return found


def collect_vars(value: object, found: dict[str, "expr.DimVar"]) -> None:
    """Add every `DimVar` reachable from *value* to *found*, by name."""
    if isinstance(value, types.TensorType):
        for entry in value.shape:
            collect_vars(entry, found)

        collect_layout_vars(value.layout, found)
        return
    if isinstance(value, types.TupleType):
        for field in value.fields:
            collect_vars(field, found)
        return
    if isinstance(value, types.PointerType):
        return
    if isinstance(value, expr.DimVar):
        found[value.name] = value
        return
    if isinstance(value, tuple):
        for entry in value:
            collect_vars(entry, found)
        return
    if isinstance(value, types.Topology):
        collect_vars(value.size, found)
        return
    if isinstance(value, types.Mesh):
        collect_vars(value.topologies, found)
        collect_layout_vars(value.layout, found)
        return
    if isinstance(value, (types.Layout, types.ComposedLayout, types.ShardLayout)):
        collect_layout_vars(value, found)
        return
    if isinstance(value, expr.Call) and isinstance(value.target, expr.DIM_OP_TYPES):
        expr.dim_variables(value, found)


def collect_layout_vars(layout: object, found: dict[str, "expr.DimVar"]) -> None:
    """Add every `DimVar` of one layout, its mesh and its offsets to *found*."""
    if isinstance(layout, types.ShardLayout):
        collect_layout_vars(layout.layout, found)
        collect_vars(layout.mesh, found)
        return
    if isinstance(layout, types.ComposedLayout):
        collect_layout_vars(layout.outer, found)
        collect_layout_vars(layout.inner, found)
        collect_vars(layout.offset, found)
        return
    if isinstance(layout, types.Layout):
        collect_vars(layout.shape, found)
        collect_vars(layout.strides, found)


def substitute_dims(value: types.Type, bindings: Mapping[str, int]) -> types.Type:
    """*value* with every bound `DimVar` replaced by its chosen extent.

    Unbound dimensions are left as they are. Arithmetic over dimensions folds
    as soon as its operands are known, through the same construction-time
    folding that built it, so a shape written `ctx_len // 8` becomes an integer
    rather than a division that survives into the analysis.
    """
    if isinstance(value, types.TensorType):
        shape = tuple(substitute_shape_entry(entry, bindings) for entry in value.shape)
        layout = substitute_layout_dims(value.layout, bindings)
        if shape == value.shape and layout is value.layout:
            return value
        return types.TensorType(
            shape=shape,
            dtype=value.dtype,
            layout=layout,
            storage=value.storage,
        )
    if isinstance(value, types.TupleType):
        fields = tuple(substitute_dims(field, bindings) for field in value.fields)
        if fields == value.fields:
            return value
        return types.TupleType(fields=fields)
    return value


def canonicalize_dims(value: types.Type) -> types.Type:
    """Return *value* with every symbolic dimension in isl normal form."""
    if not has_symbolic_dims(value):
        return value
    if isinstance(value, types.TensorType):
        shape = tuple(normalize(entry) for entry in value.shape)
        layout = canonicalize_layout_dims(value.layout)
        if shape == value.shape and layout is value.layout:
            return value
        return types.TensorType(
            shape=shape,
            dtype=value.dtype,
            layout=layout,
            storage=value.storage,
        )
    if isinstance(value, types.TupleType):
        fields = tuple(canonicalize_dims(field) for field in value.fields)
        if fields == value.fields:
            return value
        return types.TupleType(fields=fields)
    return value


def canonicalize_layout_dims(layout: object) -> object:
    """*layout* with every dimension in isl normal form."""
    if isinstance(layout, types.ShardLayout):
        inner = canonicalize_layout_dims(layout.layout)
        mesh = canonicalize_mesh_dims(layout.mesh)
        if inner is layout.layout and mesh is layout.mesh:
            return layout
        return types.ShardLayout(layout=inner, attrs=layout.attrs, mesh=mesh)
    if isinstance(layout, types.ComposedLayout):
        outer = canonicalize_layout_dims(layout.outer)
        inner = canonicalize_layout_dims(layout.inner)
        offset = normalize(layout.offset)
        if outer is layout.outer and inner is layout.inner and offset == layout.offset:
            return layout
        return types.ComposedLayout(inner=inner, offset=offset, outer=outer)
    if isinstance(layout, types.Layout):
        shape = canonicalize_nested(layout.shape)
        strides = canonicalize_nested(layout.strides)
        if shape == layout.shape and strides == layout.strides:
            return layout
        return types.Layout(shape=shape, strides=strides)
    return layout


def canonicalize_topology_dims(topology: object) -> object:
    """*topology* with its extent in isl normal form."""
    if not isinstance(topology, types.Topology):
        return topology
    size = normalize(topology.size)
    if size == topology.size:
        return topology
    return types.Topology(topology.name, size)


def map_mesh_layout(layout, rebuild):
    """Rebuild what a mesh stated, whether that is one arrangement or one per level."""
    if isinstance(layout, tuple):
        return tuple(rebuild(one) for one in layout)
    return rebuild(layout)


def canonicalize_mesh_dims(mesh: object) -> object:
    """*mesh* with its layout and topology dimensions in isl normal form."""
    if not isinstance(mesh, types.Mesh):
        return mesh
    topologies = tuple(canonicalize_topology_dims(item) for item in mesh.topologies)
    layout = map_mesh_layout(mesh.layout, canonicalize_layout_dims)
    if topologies == mesh.topologies and layout == mesh.layout:
        return mesh
    return types.Mesh(topologies=topologies, layout=layout, names=mesh.names)


def canonicalize_nested(entries: tuple) -> tuple:
    """A possibly nested tuple of shape entries, normalized throughout."""
    return tuple(
        canonicalize_nested(entry)
        if isinstance(entry, tuple)
        else (None if entry is None else normalize(entry))
        for entry in entries
    )


def substitute_layout_dims(layout: object, bindings: Mapping[str, int]) -> object:
    """*layout* with its bound dimensions replaced.

    A layout restates the shape it describes, so leaving it behind produces a
    type whose shape is a number and whose layout is still a range -- concrete
    to anything that reads the shape, and not to anything that reads the layout.
    A shard layout's Mesh is part of that geometry: its layout and topology
    extents may be derived from the same dimensions as the tensor shape.
    """
    if isinstance(layout, types.ShardLayout):
        inner = substitute_layout_dims(layout.layout, bindings)
        mesh = substitute_mesh_dims(layout.mesh, bindings)
        if inner is layout.layout and mesh is layout.mesh:
            return layout
        return types.ShardLayout(layout=inner, attrs=layout.attrs, mesh=mesh)
    if isinstance(layout, types.ComposedLayout):
        outer = substitute_layout_dims(layout.outer, bindings)
        inner = substitute_layout_dims(layout.inner, bindings)
        offset = substitute_shape_entry(layout.offset, bindings)
        if outer is layout.outer and inner is layout.inner and offset == layout.offset:
            return layout
        return types.ComposedLayout(inner=inner, offset=offset, outer=outer)
    if isinstance(layout, types.Layout):
        shape = substitute_nested(layout.shape, bindings)
        strides = substitute_nested(layout.strides, bindings)
        if shape == layout.shape and strides == layout.strides:
            return layout
        return types.Layout(shape=shape, strides=strides)
    return layout


def substitute_topology_dims(topology: object, bindings: Mapping[str, int]) -> object:
    """*topology* with its extent rebuilt from *bindings*."""
    if not isinstance(topology, types.Topology):
        return topology
    size = substitute_shape_entry(topology.size, bindings)
    if size == topology.size:
        return topology
    return types.Topology(topology.name, size)


def substitute_mesh_dims(mesh: object, bindings: Mapping[str, int]) -> object:
    """*mesh* with layout and topology dimensions rebuilt together."""
    if not isinstance(mesh, types.Mesh):
        return mesh
    topologies = tuple(substitute_topology_dims(item, bindings) for item in mesh.topologies)
    layout = map_mesh_layout(mesh.layout, lambda one: substitute_layout_dims(one, bindings))
    if topologies == mesh.topologies and layout == mesh.layout:
        return mesh
    return types.Mesh(topologies=topologies, layout=layout, names=mesh.names)


def substitute_nested(entries: tuple, bindings: Mapping[str, int]) -> tuple:
    """A possibly nested tuple of shape entries, substituted throughout.

    A layout's shape is hierarchical, so an entry may itself be a tuple; and it
    may be `None`, which states an axis whose extent this layout does not fix.
    """
    return tuple(
        substitute_nested(entry, bindings)
        if isinstance(entry, tuple)
        else (None if entry is None else substitute_shape_entry(entry, bindings))
        for entry in entries
    )


def substitute_shape_entry(entry: object, bindings: Mapping[str, int]) -> object:
    """One shape entry with its bound dimensions replaced.

    Exposed separately because a dimension is not only found in a type: a loop
    states its own extent, step and start as shape entries, and those are not
    reachable from any type the loop's body carries.
    """
    if isinstance(entry, bool):
        raise SubstitutionError(f"{entry!r} is not a shape entry")
    if isinstance(entry, int):
        return entry
    if isinstance(entry, (expr.DimVar, expr.Constant, expr.Call)):
        return ShapeEntryMutator(bindings).visit(entry)
    return entry


def checked(variable: expr.DimVar, extent: int) -> int:
    """*extent*, once the closed declaration is known to admit it."""
    if isinstance(extent, bool) or not isinstance(extent, int):
        raise SubstitutionError(
            f"dimension {variable.name!r} takes an integer extent, got {extent!r}"
        )
    if not variable.lo <= extent <= variable.hi:
        raise SubstitutionError(
            f"dimension {variable.name!r} was declared over "
            f"[{variable.lo}, {variable.hi}] and cannot take {extent}"
        )
    return extent


def has_symbolic_dims(value: object) -> bool:
    """Whether anything reachable from *value* is not a static dimension."""
    if isinstance(value, expr.DimVar):
        return True
    if isinstance(value, expr.Call) and isinstance(value.target, expr.DIM_OP_TYPES):
        return any(has_symbolic_dims(arg) for arg in value.args)
    if isinstance(value, types.TensorType):
        return has_symbolic_dims(value.shape) or has_symbolic_dims(value.layout)
    if isinstance(value, types.TupleType):
        return any(has_symbolic_dims(field) for field in value.fields)
    if isinstance(value, types.PointerType):
        return False
    if isinstance(value, types.Topology):
        return has_symbolic_dims(value.size)
    if isinstance(value, types.Mesh):
        return has_symbolic_dims(value.topologies) or has_symbolic_dims(value.layout)
    if isinstance(value, types.ShardLayout):
        return has_symbolic_dims(value.layout) or has_symbolic_dims(value.mesh)
    if isinstance(value, types.ComposedLayout):
        return any(has_symbolic_dims(entry) for entry in (value.inner, value.offset, value.outer))
    if isinstance(value, types.Layout):
        return has_symbolic_dims(value.shape) or has_symbolic_dims(value.strides)
    if isinstance(value, tuple):
        return any(has_symbolic_dims(entry) for entry in value)
    return False
