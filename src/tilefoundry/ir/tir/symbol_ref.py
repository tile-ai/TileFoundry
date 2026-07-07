"""SymbolRef — a module-symbol reference to a callee PrimFunction."""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.expr import Expr


@dataclass(frozen=True)
class SymbolRef(Expr):
    """Leaf ``Expr`` naming a callee ``PrimFunction`` as a call target."""

    name: str
    nested: tuple[str, ...] = ()


def symbol_call(callee, args) -> "Evaluate":  # noqa: F821 -- lazy Evaluate
    """Build ``Evaluate(SymbolRef(callee), args)`` — a Stmt-position call of a
    callee ``PrimFunction`` by symbol.
    """
    from tilefoundry.ir.tir.stmts import Evaluate  # noqa: PLC0415
    from tilefoundry.ir.types import callable_type_for_prim_function  # noqa: PLC0415

    ref = SymbolRef(name=callee.name, type=callable_type_for_prim_function(callee))
    return Evaluate(callable=ref, args=tuple(args))


__all__ = ["SymbolRef", "symbol_call"]
