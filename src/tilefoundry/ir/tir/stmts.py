"""Define structural TIR statements.

Parser assignment sugar lowers to ``LetStmt``; allocations are expression Ops
anchored as its value. ``Sequential`` packs a tuple into one Stmt so every body
uses a uniform statement slot while remaining iterable.

See [tir §1.2](docs/spec/tir.md#12-structural-stmts-tirstmts).
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.types.shard.mesh import Mesh


@dataclass(frozen=True)
class Sequential(Stmt):
    """Wrap a ``tuple[Stmt, ...]`` as one TIR statement."""

    body: tuple[Stmt, ...]

    def __iter__(self):
        return iter(self.body)

    def __len__(self) -> int:
        return len(self.body)

    def __getitem__(self, idx):
        return self.body[idx]


@dataclass(frozen=True)
class LetStmt(Stmt):
    """TIR's single value-binding node."""

    var: Var
    value: Expr
    body: Sequential


@dataclass(frozen=True)
class For(Stmt):
    induction_var: Var
    start: Expr
    stop: Expr
    step: Expr
    body: Sequential


@dataclass(frozen=True)
class While(Stmt):
    cond: Expr
    body: Sequential


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: Sequential
    else_body: Sequential


@dataclass(frozen=True)
class MeshScope(Stmt):
    mesh: Mesh
    binding: Var
    body: Sequential


@dataclass(frozen=True)
class Return(Stmt):
    """Empty return; tir functions have no value return."""


@dataclass(frozen=True)
class Abort(Stmt):
    """Terminating stmt — runtime unreachable / dispatch fallback.

    A successfully-verified Abort exists in code paths the compiler
    believes are unreachable. When hit at runtime, the CUDA emitter
    produces ``__trap();`` (or equivalent abort) so failures are
    loud rather than silent.
    """

    message: str = ""


@dataclass(frozen=True)
class Evaluate(Stmt):
    """Place a value-less effect Op or symbol invocation in statement position."""

    callable: "Op | SymbolRef"  # noqa: A003 -- spec field name
    args: tuple[Expr, ...]


__all__ = [
    "Sequential",
    "LetStmt",
    "For",
    "While",
    "If",
    "MeshScope",
    "Return",
    "Abort",
    "Evaluate",
]
