"""SymbolRef — a module-symbol reference to a callee PrimFunction."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.expr import Expr


@dataclass(unsafe_hash=True)
class SymbolRef(Expr):
    """Leaf ``Expr`` naming a callee ``PrimFunction`` as a call target."""

    name: str
    nested: tuple[str, ...] = ()


def symbol_ref(callee, symbol: str | None = None) -> SymbolRef:
    """Construct a typed symbol, splitting an explicit child-slot selector."""
    from tilefoundry.ir.types.callable_type import (  # noqa: PLC0415
        CallableType,
        callable_type_for_prim_function,
    )

    type_ = getattr(callee, "type", None)
    if not isinstance(type_, CallableType):
        type_ = callable_type_for_prim_function(callee)
    parts = (callee.name,) if symbol is None else tuple(symbol.split("."))
    return SymbolRef(name=parts[-1], nested=parts[:-1], type=type_)


def symbol_call(callee, args, *, symbol: str | None = None) -> "Evaluate":  # noqa: F821
    """Build an effect call with an optional child-slot selector."""
    from tilefoundry.ir.tir.stmts import Evaluate  # noqa: PLC0415

    return Evaluate(callable=symbol_ref(callee, symbol), args=tuple(args))


__all__ = ["SymbolRef", "symbol_ref", "symbol_call"]
