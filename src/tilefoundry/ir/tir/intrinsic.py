from __future__ import annotations

import inspect
from dataclasses import make_dataclass

from tilefoundry.ir.core import Expr, VerifyError
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.visitor_registry import register_verify_stmt

# DSL-surface name → Stmt subclass. Parser reads this to dispatch user
# intrinsic calls inside @prim_func bodies.
_intrinsic_dispatch: dict[str, type[Stmt]] = {}


def _snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def intrinsic(fn):
    """§4.3.5. Generate a Stmt subclass, register verify_stmt from the original
    function body, and wire parser dispatch."""
    sig = inspect.signature(fn)
    param_names: list[str] = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"@intrinsic {fn.__name__}: *args/**kwargs not supported")
        if p.annotation is inspect.Parameter.empty:
            raise TypeError(f"@intrinsic {fn.__name__}: parameter {p.name!r} must be annotated Expr")
        # Accept both live `Expr` and string "Expr" annotations.
        ann = p.annotation
        if ann is not Expr and ann != "Expr":
            raise TypeError(
                f"@intrinsic {fn.__name__}: parameter {p.name!r} must be annotated Expr, got {ann!r}"
            )
        param_names.append(p.name)
    ret_ann = sig.return_annotation
    if ret_ann is not inspect.Signature.empty and ret_ann is not None and ret_ann != "None":
        raise TypeError(f"@intrinsic {fn.__name__}: return annotation must be None")

    camel = _snake_to_camel(fn.__name__)
    fields = [(n, Expr) for n in param_names]
    stmt_cls = make_dataclass(camel, fields, bases=(Stmt,), frozen=True)
    stmt_cls.__module__ = fn.__module__

    def _verify(stmt, ctx):
        bound = {n: getattr(stmt, n) for n in param_names}
        try:
            fn(**bound)
        except VerifyError:
            raise
        except Exception as exc:  # AssertionError + anything else
            ctx.error(stmt, str(exc) or type(exc).__name__)

    register_verify_stmt(stmt_cls)(_verify)

    if fn.__name__ in _intrinsic_dispatch:
        raise RuntimeError(f"@intrinsic: name {fn.__name__!r} already registered")
    _intrinsic_dispatch[fn.__name__] = stmt_cls
    return stmt_cls


__all__ = ["intrinsic", "_intrinsic_dispatch"]
