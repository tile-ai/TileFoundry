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
    found: dict[str, None] = {}
    _collect(value, found)
    return tuple(found)


def _collect(value: object, found: dict[str, None]) -> None:
    if isinstance(value, TensorType):
        for entry in value.shape:
            _collect(entry, found)
        return
    if isinstance(value, TupleType):
        for field in value.fields:
            _collect(field, found)
        return
    if isinstance(value, DimVar):
        found[value.name] = None
        return
    if isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES):
        for arg in value.args:
            _collect(arg, found)


def substitute_dims(value: Type, bindings: Mapping[str, int]) -> Type:
    """*value* with every bound `DimVar` replaced by its chosen extent.

    Unbound dimensions are left as they are. Arithmetic over dimensions folds
    as soon as its operands are known, through the same construction-time
    folding that built it, so a shape written `ctx_len // 8` becomes an integer
    rather than a division that survives into the analysis.
    """
    if isinstance(value, TensorType):
        shape = tuple(substitute_shape_dim(entry, bindings) for entry in value.shape)
        if shape == value.shape:
            return value
        return TensorType(
            shape=shape,
            dtype=value.dtype,
            layout=value.layout,
            storage=value.storage,
        )
    if isinstance(value, TupleType):
        fields = tuple(substitute_dims(field, bindings) for field in value.fields)
        if fields == value.fields:
            return value
        return TupleType(fields=fields)
    return value


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
        return simplify_dim(type(entry.target), args)
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
    "dim_vars_in",
    "has_symbolic_dims",
    "substitute_dims",
    "substitute_shape_dim",
]
