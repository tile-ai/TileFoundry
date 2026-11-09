"""Generate and register effect-statement classes for TIR intrinsics.

See [tir §1.8](docs/spec/tir.md#18-intrinsic--user-defined-effect-stmts).
"""

from __future__ import annotations

import inspect
from dataclasses import make_dataclass

from tilefoundry.interfaces import verify
from tilefoundry.ir import core, expr


class IntrinsicTable:
    """The Stmt class generated for each ``@intrinsic`` function, by function name."""

    def __init__(self) -> None:
        """Start with no intrinsic."""
        self.classes: dict[str, type[core.Stmt]] = {}

    def add(self, name: str, stmt_cls: type[core.Stmt]) -> None:
        """Record *stmt_cls* under *name*, refusing a name that is already taken."""
        if name in self.classes:
            raise RuntimeError(f"@intrinsic: name {name!r} already registered")
        self.classes[name] = stmt_cls


intrinsic_dispatch = IntrinsicTable()


def snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def intrinsic(fn):
    """Generate a Stmt class and register its verifier and parser dispatch."""
    sig = inspect.signature(fn)
    param_names: list[str] = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"@intrinsic {fn.__name__}: *args/**kwargs not supported")
        if p.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"@intrinsic {fn.__name__}: parameter {p.name!r} must be annotated Expr"
            )

        ann = p.annotation
        if ann is not expr.Expr and ann != "Expr":
            raise TypeError(
                f"@intrinsic {fn.__name__}: parameter {p.name!r} must be annotated Expr, got {ann!r}"
            )
        param_names.append(p.name)
    ret_ann = sig.return_annotation
    if ret_ann is not inspect.Signature.empty and ret_ann is not None and ret_ann != "None":
        raise TypeError(f"@intrinsic {fn.__name__}: return annotation must be None")

    camel = snake_to_camel(fn.__name__)
    fields = [(n, expr.Expr) for n in param_names]
    stmt_cls = make_dataclass(camel, fields, bases=(core.Stmt,), module=fn.__module__)

    def verify_intrinsic(stmt, ctx):
        bound = {n: getattr(stmt, n) for n in param_names}
        try:
            fn(**bound)
        except expr.VerifyError:
            raise
        except Exception as exc:
            ctx.error(stmt, str(exc) or type(exc).__name__)

    verify.register(stmt_cls)(verify_intrinsic)

    intrinsic_dispatch.add(fn.__name__, stmt_cls)
    return stmt_cls


__all__ = ["IntrinsicTable", "intrinsic", "intrinsic_dispatch"]
