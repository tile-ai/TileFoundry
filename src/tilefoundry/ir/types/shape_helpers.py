"""Handle static and symbolic entries in tensor shapes.

Upper-bound helpers produce compile-time allocation sizes. Runtime-total
helpers combine static factors with expressions for actual symbolic extents.

See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

from .dim import DimVar


def static_dim_value(dim):
    """Return the compile-time ``int`` value of a *static* shape dim, else ``None``.

    A static dim is a plain ``int`` or an integer-valued ``Constant`` (the latter
    only appears transiently before ``TensorType`` canonicalizes it to ``int``).
    ``DimVar`` / dynamic dim ``Call`` exprs are not static → ``None``. The
    detection is exact (real ``Constant`` with an ``int`` value), never "anything
    with a ``.value``"; the ``Constant`` import is deferred to dodge the
    ``ir.core.expr`` ↔ ``ir.types`` cycle and fails closed (returns ``None``).
    """
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    try:
        from tilefoundry.ir.core.expr import Constant  # noqa: PLC0415 - cycle guard
    except ImportError:  # pragma: no cover - import-cycle guard
        return None
    if isinstance(dim, Constant) and isinstance(dim.value, int) and not isinstance(dim.value, bool):
        return int(dim.value)
    return None


def i64_const(value: int) -> "Constant":
    """The canonical i64 shape-scalar ``Constant`` (meta-scalar typed)."""
    from tilefoundry.ir.core.expr import Constant  # noqa: PLC0415 - cycle guard

    from .tensor_type import TensorType  # noqa: PLC0415 - cycle guard

    return Constant(type=TensorType.meta_scalar(), value=int(value))


def upper_bound(dim) -> int:
    """Return a concrete int upper-bound element count for ``dim``."""
    if isinstance(dim, DimVar):
        return int(dim.hi) - 1
    static = static_dim_value(dim)
    if static is not None:
        return static
    return int(dim)


def shape_numel_upper_bound(shape) -> int:
    """Product of per-dim upper bounds.

    Product of per-dim upper bounds: the static element count a buffer or
    layout must hold across every runtime shape in the dispatch envelope.
    """
    n = 1
    for s in shape:
        n *= upper_bound(s)
    return n


def shape_upper_bound(shape) -> tuple[int, ...]:
    """Map ``upper_bound`` over every entry of *shape*."""
    return tuple(upper_bound(s) for s in shape)


def shape_has_dim_var(shape) -> bool:
    """True iff *shape* contains at least one ``DimVar`` entry."""
    return any(isinstance(s, DimVar) for s in shape)


def shape_runtime_total(shape, dim_var_expr: dict[str, str]) -> object:
    """Return the runtime element count of *shape*.

    All-static shape → an ``int``. Any ``DimVar`` axis pulls its
    runtime extent from ``dim_var_expr[name]``; the result is a C++
    expression string ``"(a * b * ...)"`` that the codegen splices
    verbatim into the generated source. Static dims fold into a single
    leading constant factor when present, otherwise the constant is
    elided.
    """
    if not shape:
        return 1
    static_prod = 1
    dyn_terms: list[str] = []
    for s in shape:
        if isinstance(s, DimVar):
            expr = dim_var_expr.get(s.name)
            if expr is None:
                static_prod *= upper_bound(s)
            else:
                dyn_terms.append(expr)
        else:
            static_prod *= upper_bound(s)
    if not dyn_terms:
        return static_prod
    if static_prod == 1:
        if len(dyn_terms) == 1:
            return dyn_terms[0]
        return "(" + " * ".join(dyn_terms) + ")"
    return "(" + " * ".join([str(static_prod), *dyn_terms]) + ")"


__all__ = [
    "static_dim_value",
    "i64_const",
    "upper_bound",
    "shape_numel_upper_bound",
    "shape_upper_bound",
    "shape_has_dim_var",
    "shape_runtime_total",
]
