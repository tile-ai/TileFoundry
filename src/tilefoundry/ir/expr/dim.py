"""The dimension operations: plain data, and the interned dimension variable.

An operation here is a node target and nothing else. Building arithmetic over
them, simplifying it and converting it to isl belong to :mod:`tilefoundry.ir.dim`,
which implements the ``simplify_dim`` service these classes call. The queries
read only the node classes, so every layer above can ask them without a cycle.

See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

from tilefoundry import services

from .expr import Call, Constant
from .op import Op
from .param_def import ParamDef

__all__ = [
    "DIM_OP_TYPES",
    "DimAdd",
    "DimConst",
    "DimFloorDiv",
    "DimMax",
    "DimMin",
    "DimMod",
    "DimMul",
    "DimSub",
    "DimVar",
    "dim_variables",
    "exact_quotient",
    "is_dim_op_call",
    "is_literal_shape",
    "shape_has_dim_var",
    "static_dim_value",
]


class DimConst(Op):
    """A compile-time integer dimension."""

    value = ParamDef(kind="attribute", annotation=int)


class VariableTable:
    """The dimension variables built so far, one per ``(class, name, lo, hi)``."""

    def __init__(self) -> None:
        """Start with no variable."""
        self.entries: dict[tuple, DimVar] = {}

    def get(self, key: tuple) -> DimVar | None:
        """The variable built for *key*, if any."""
        return self.entries.get(key)

    def add(self, key: tuple, variable: DimVar) -> None:
        """Remember *variable* as the one for *key*."""
        self.entries[key] = variable


variables = VariableTable()


def checked_bounds(name, lo, hi) -> tuple[str, int, int]:
    """*name*, *lo* and *hi* of a dimension variable, once they are known to be valid."""
    if name is None or not isinstance(name, str) or not name:
        raise TypeError("DimVar requires a non-empty str name=")
    if not isinstance(lo, int) or isinstance(lo, bool):
        raise TypeError(f"DimVar({name!r}): lo must be int, got {type(lo).__name__}")
    if not isinstance(hi, int) or isinstance(hi, bool):
        raise TypeError(f"DimVar({name!r}): hi must be int, got {type(hi).__name__}")
    if not (lo <= hi):
        raise ValueError(
            f"DimVar({name!r}, {lo}, {hi}): require lo <= hi "
            f"(closed envelope [lo, hi]; a fixed dim is [k, k])"
        )
    return name, lo, hi


class DimVar(Op):
    """A named dimension over the closed envelope ``[lo, hi]``.

    One ``(name, lo, hi)`` is one instance: building it twice returns the same object.
    """

    name = ParamDef(kind="attribute", annotation=str)
    lo = ParamDef(kind="attribute", annotation=int)
    hi = ParamDef(kind="attribute", annotation=int)

    def __new__(cls, name=None, lo=None, hi=None, **attrs):
        """The interned variable for ``(name, lo, hi)``, or a fresh one."""
        name, lo, hi = checked_bounds(name, lo, hi)
        held = variables.get((cls, name, lo, hi))
        if held is not None:
            return held
        return super().__new__(cls, name=name, lo=lo, hi=hi, **attrs)

    def __init__(self, name=None, lo=None, hi=None, **attrs) -> None:
        """Set the attributes of a fresh variable and intern it; an interned one is kept."""
        key = (type(self), name, lo, hi)
        if variables.get(key) is self:
            return
        super().__init__(name=name, lo=lo, hi=hi, **attrs)
        variables.add(key, self)

    def __reduce__(self):
        """Copy and pickle through the interning constructor."""
        return (type(self), (self.name, self.lo, self.hi))

    def __deepcopy__(self, memo):
        """Itself: one ``(name, lo, hi)`` is one instance, and copying keeps that."""
        return self

    def __add__(self, other):
        """Dimension sum."""
        return services.simplify_dim("add", (self, other))

    def __radd__(self, other):
        """Dimension sum with *other* on the left."""
        return services.simplify_dim("add", (other, self))

    def __sub__(self, other):
        """Dimension difference."""
        return services.simplify_dim("sub", (self, other))

    def __rsub__(self, other):
        """Dimension difference with *other* on the left."""
        return services.simplify_dim("sub", (other, self))

    def __mul__(self, other):
        """Dimension product."""
        return services.simplify_dim("mul", (self, other))

    def __rmul__(self, other):
        """Dimension product with *other* on the left."""
        return services.simplify_dim("mul", (other, self))

    def __floordiv__(self, other):
        """Dimension floor quotient."""
        return services.simplify_dim("floordiv", (self, other))

    def __rfloordiv__(self, other):
        """Dimension floor quotient with *other* on the left."""
        return services.simplify_dim("floordiv", (other, self))

    def __mod__(self, other):
        """Dimension remainder."""
        return services.simplify_dim("mod", (self, other))

    def __rmod__(self, other):
        """Dimension remainder with *other* on the left."""
        return services.simplify_dim("mod", (other, self))


class DimAdd(Op):
    """Sum of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimSub(Op):
    """Difference of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMul(Op):
    """Product of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimFloorDiv(Op):
    """Floor quotient of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMod(Op):
    """Remainder of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMin(Op):
    """Smaller of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMax(Op):
    """Larger of two dimensions."""

    dimension_arithmetic = True

    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


DIM_OP_TYPES = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)


def is_dim_op_call(value) -> bool:
    """True iff *value* is a ``Call`` over one of the dimension arithmetic ops.

    Unlike ``is_dim_expr`` this asks only what the call computes, not what
    it computes over, so it also holds for a coordinate built on a scalar ``Var``
    that only the surrounding walk can resolve. Dim arithmetic is an address
    rather than a value some kernel produces, so a walk over compute ops uses
    this to leave it alone.
    """
    return isinstance(value, Call) and isinstance(value.target, DIM_OP_TYPES)


def static_dim_value(dim):
    """Return the compile-time ``int`` value of a *static* shape dim, else ``None``.

    A static dim is a plain ``int`` or an integer-valued ``Constant`` (the latter
    only appears transiently before ``TensorType`` canonicalizes it to ``int``).
    ``DimVar`` / dynamic dim ``Call`` exprs are not static → ``None``. The
    detection is exact (real ``Constant`` with an ``int`` value), never "anything
    with a ``.value``".
    """
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    if isinstance(dim, Constant) and isinstance(dim.value, int) and not isinstance(dim.value, bool):
        return int(dim.value)
    return None


def is_literal_shape(shape) -> bool:
    """Whether every extent of *shape* is a non-negative plain ``int``.

    Stricter than ``static_dim_value``: an integer ``Constant`` does not count,
    so a reader that clips to a box only does so for a shape written in numbers.
    """
    return all(
        isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0 for extent in shape
    )


def shape_has_dim_var(shape) -> bool:
    """True iff *shape* contains at least one ``DimVar`` entry."""
    return any(isinstance(s, DimVar) for s in shape)


def exact_quotient(value, divisor):
    """Return an exact integer or structural product quotient, if decidable.

    A shard Split divides a logical axis by its mesh extent. Literal integer
    divisibility, identical dimensions, and either factor of DimMul determine
    the quotient; other relations return None for the caller to diagnose.
    """
    if isinstance(value, int) and isinstance(divisor, int):
        return value // divisor if divisor != 0 and value % divisor == 0 else None
    if value == divisor:
        return 1
    if isinstance(value, Call) and isinstance(value.target, DimMul) and divisor in value.args:
        left, right = value.args
        quotient = right if left == divisor else left
        return quotient.value if isinstance(quotient, Constant) else quotient
    return None


def dim_variables(value, found: dict[str, DimVar] | None = None) -> dict[str, DimVar]:
    """Every distinct ``DimVar`` of one dimension, by name, first-seen first.

    Only a ``DimVar`` and a ``Call`` over the dimension operations are read; any
    other value holds none. A node shared by several operands is read once.
    """
    found = {} if found is None else found
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, DimVar):
            found[item.name] = item
        elif isinstance(item, Call) and isinstance(item.target, DIM_OP_TYPES):
            pending.extend(reversed(item.args))
    return found
