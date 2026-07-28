"""Replacing a symbolic dimension with the extent it was chosen to have.

A model authored for decode states its context length as a `DimVar` spanning
every length it will ever see. Analysis and scheduling need one length: they
count elements, compare footprints to capacities and lay work out over a mesh,
and none of that has an answer for a dimension that is still a range. So the
choice of length is made here, and what comes out states extents instead of
ranges.

Substitution is deliberately partial. Binding one dimension must leave the
others symbolic, because the choice is made one dimension at a time -- a
context length is picked while the sequence length stays open -- and an
implementation that demanded every variable at once could not express that.

This lives beside the types rather than beside `DimVar` itself: the dimension
vocabulary is what tensor types are built from, so a substituter that knows
about `TensorType` cannot live in the module `TensorType` imports.
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


def _layout_types() -> tuple[type, ...]:
    """The layout descriptors, imported at call time to avoid a cycle."""
    from .shard.layout import ComposedLayout, Layout  # noqa: PLC0415
    from .shard.shard_layout import ShardLayout  # noqa: PLC0415

    return (Layout, ComposedLayout, ShardLayout)


def _collect(value: object, found: dict[str, "DimVar"]) -> None:
    if isinstance(value, TensorType):
        for entry in value.shape:
            _collect(entry, found)
        # A layout restates the shape it describes, and a sharded one restates
        # it per position. A scan that read only the shape would call a type
        # concrete while its layout still held the range.
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
    if isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES):
        for arg in value.args:
            _collect(arg, found)


def _collect_layout(layout: object, found: dict[str, "DimVar"]) -> None:
    if layout is None:
        return
    Layout, ComposedLayout, ShardLayout = _layout_types()
    if isinstance(layout, ShardLayout):
        # The mesh states machine positions, which are not derived from the
        # program's sizes, so it holds no dimension of the program.
        _collect_layout(layout.layout, found)
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
    The mesh is untouched: it states machine positions, which no program size
    derives from.
    """
    if layout is None:
        return layout
    Layout, ComposedLayout, ShardLayout = _layout_types()
    if isinstance(layout, ShardLayout):
        inner = substitute_layout_dims(layout.layout, bindings)
        if inner is layout.layout:
            return layout
        return ShardLayout(layout=inner, attrs=layout.attrs, mesh=layout.mesh)
    if isinstance(layout, ComposedLayout):
        outer = substitute_layout_dims(layout.outer, bindings)
        inner = substitute_layout_dims(layout.inner, bindings)
        offset = substitute_shape_dim(layout.offset, bindings)
        if (
            outer is layout.outer
            and inner is layout.inner
            and offset == layout.offset
        ):
            return layout
        return ComposedLayout(inner=inner, offset=offset, outer=outer)
    if isinstance(layout, Layout):
        shape = _substitute_nested(layout.shape, bindings)
        strides = (
            None
            if layout.strides is None
            else _substitute_nested(layout.strides, bindings)
        )
        if shape == layout.shape and strides == layout.strides:
            return layout
        return Layout(shape=shape, strides=strides)
    return layout


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
        # Folding yields a wrapped integer. A tensor type canonicalises those
        # back to plain integers in its shape, but an operation's shape-valued
        # attribute is not a type and would keep the wrapper -- leaving one
        # entry of a shape a different kind of thing from its neighbours.
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
    """Whether anything reachable from *value* is still a range."""
    return bool(dim_vars_in(value))


__all__ = [
    "DimSubstitutionError",
    "dim_vars_by_name",
    "dim_vars_in",
    "has_symbolic_dims",
    "substitute_dims",
    "substitute_layout_dims",
    "substitute_shape_dim",
]
