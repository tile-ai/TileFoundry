"""Small helpers for handling mixed static / dynamic shape entries.

A tensor's ``shape`` tuple may contain static ``int`` entries or symbolic
``DimVar(name, lo, hi)`` entries (half-open envelope: ``lo <= S < hi``).
Two helper modes:

- ``upper_bound(dim)`` returns a concrete ``int`` — the maximum runtime
  value of the dim (``hi - 1`` for a ``DimVar``, since ``hi`` is exclusive),
  which is the natural allocation count for static buffer sizing.
  Used to size **compile-time** layouts and per-thread register
  buffers so a single binary covers every runtime shape that flows
  through the dispatch.

- ``shape_runtime_total(shape, dim_var_expr)`` returns either an
  ``int`` (all-static) or a C++ expression string that multiplies the
  static dims by the runtime kernel scalars given in ``dim_var_expr``
  (a mapping ``DimVar.name -> "<param>_shape_<axis>"``). Used at codegen
  call sites where the *actual* element count must drive the kernel
  loop (``tilefoundry::ops::binary`` / ``unary`` / ``fill`` / ``copy``),
  not the static envelope.
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
    ``ir.core.expr`` ↔ ``ir.types`` cycle and fails closed (returns ``None``)."""
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    try:
        from tilefoundry.ir.core.expr import Constant  # noqa: PLC0415 - cycle guard
    except ImportError:  # pragma: no cover - import-cycle guard
        return None
    if isinstance(dim, Constant) and isinstance(dim.value, int) and not isinstance(dim.value, bool):
        return int(dim.value)
    return None


def is_static_dim(dim) -> bool:
    """True iff ``dim`` is a compile-time-known integer dim (``int`` or integer
    ``Constant``). Mirrors :func:`static_dim_value` (which to use when the value
    is needed)."""
    return static_dim_value(dim) is not None


def upper_bound(dim) -> int:
    """Return a concrete int upper-bound element count for ``dim``."""
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    if isinstance(dim, DimVar):
        # Half-open envelope ``[lo, hi)``: ``hi`` is exclusive, so the maximum
        # runtime value (the element count a static buffer must hold) is hi-1.
        return int(dim.hi) - 1
    if hasattr(dim, "value"):
        return int(dim.value)
    return int(dim)


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
                # Fall back to envelope upper bound when the codegen
                # context has no runtime scalar registered for this
                # DimVar (e.g. legacy code paths without dispatch).
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
    "is_static_dim",
    "upper_bound",
    "shape_upper_bound",
    "shape_has_dim_var",
    "shape_runtime_total",
]
