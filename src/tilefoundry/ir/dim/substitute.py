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
    "collect_vars",
    "has_symbolic_dims",
    "substitute_dims",
    "substitute_shape_entry",
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
    """Add every `DimVar` reachable from *value* to *found*, by name.

    *value* may be a type, a tuple of entries or one entry; anything else holds none.
    """
    if isinstance(value, tuple):
        for entry in value:
            collect_vars(entry, found)
    elif isinstance(value, expr.DimVar):
        found[value.name] = value
    elif isinstance(value, expr.Call) and isinstance(value.target, expr.DIM_OP_TYPES):
        expr.dim_variables(value, found)
    else:
        DimCollector().visit_part(value, found)


class DimCollector(types.TypeVisitor[None]):
    """Read every dimension of a type into the ``found`` dict passed as context."""

    def visit_dim(self, entry, found) -> None:
        collect_vars(entry, found)


class DimSubstituter(types.TypeMutator):
    """Replace the dimensions bound in the ``bindings`` passed as context."""

    def visit_dim(self, entry, bindings):
        return substitute_shape_entry(entry, bindings)


class DimCanonicalizer(types.TypeMutator):
    """Put every dimension in isl normal form.

    A tensor or tuple type with no `DimVar` comes back unchanged; a topology,
    layout or mesh is normalized throughout, constants included.
    """

    def visit_dim(self, entry, ctx):
        return normalize(entry)

    def visit_TensorType(self, value, ctx):
        return super().visit_TensorType(value, ctx) if has_symbolic_dims(value) else value

    def visit_TupleType(self, value, ctx):
        return super().visit_TupleType(value, ctx) if has_symbolic_dims(value) else value


def substitute_dims(value: object, bindings: Mapping[str, int]) -> object:
    """*value* with every bound `DimVar` replaced by its chosen extent.

    *value* is any type; another value comes back as it is. Unbound dimensions
    are left as they are. Arithmetic over dimensions folds as soon as its
    operands are known, so a shape written `ctx_len // 8` becomes an integer. A
    layout restates the shape it describes, and a shard layout's Mesh is part of
    that geometry, so both are rebuilt with the shape.
    """
    return DimSubstituter().visit_part(value, bindings)


def canonicalize_dims(value: object) -> object:
    """*value* with every dimension in isl normal form; another value comes back as it is."""
    return DimCanonicalizer().visit_part(value, None)


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
    """Whether a DimVar is reachable through the supported types and dimension operations."""
    return bool(vars_by_name(value))
