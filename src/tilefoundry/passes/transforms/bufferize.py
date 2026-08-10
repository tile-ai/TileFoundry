"""BufferizePass: TIR buffer-planning pass — logical-buffer lifetime + physical placement.

Lifetime collection and scheduling are split into the ``LifetimeCollector``
and ``BufferScheduler`` hooks so a real scheduler can replace the placement
policy without touching the pass boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.passes.pass_base import PrimFuncPass


class BufferEntry(NamedTuple):
    """One logical buffer collected by ``LifetimeCollector``.

    ``var`` is the LetStmt-bound ``Var``; ``alloc`` is the anchored
    ``AllocTensor`` Op; ``defined_at`` / ``last_use_at`` are linear program
    points within the function (1-based stmt index in pre-order walk).
    """

    var: Var
    alloc: AllocTensor
    defined_at: int
    last_use_at: int


@dataclass
class Placement:
    """Physical placement decision for one logical buffer.

    MVP policy: ``offset == 0`` and ``pool_id`` is the buffer's own var, so
    every logical buffer has its own independent physical allocation. Real
    schedulers will fill ``offset`` and share ``pool_id`` across reuse
    groups.
    """

    var: Var
    pool_id: object
    offset: int = 0


class LifetimeCollector(StmtVisitor):
    """Collect allocation bindings and lifetimes from a primitive function.

    Statement traversal owns control-flow completeness. The default emits a flat
    preorder list; subclasses may replace ``collect`` with use-def liveness
    analysis without changing the bufferization pass.
    """

    def collect(self, fn: PrimFunction) -> tuple[BufferEntry, ...]:
        self._entries: list[BufferEntry] = []
        self._point = 0
        self.visit(fn.body)
        return tuple(self._entries)

    def visit(self, stmt) -> None:
        self._point += 1
        point = self._point
        if (
            isinstance(stmt, LetStmt)
            and isinstance(stmt.value, Call)
            and isinstance(stmt.value.target, AllocTensor)
        ):
            self._entries.append(
                BufferEntry(
                    var=stmt.var, alloc=stmt.value.target,
                    defined_at=point, last_use_at=point,
                )
            )
        return super().visit(stmt)


class BufferScheduler:
    """Decide a ``Placement`` for each ``BufferEntry``.

    MVP policy: every logical buffer gets its own physical allocation
    (``pool_id = entry.var``, ``offset = 0``). Real schedulers will share
    pools across non-overlapping lifetimes.
    """

    def schedule(
        self, entries: tuple[BufferEntry, ...]
    ) -> tuple[Placement, ...]:
        return tuple(
            Placement(var=e.var, pool_id=e.var, offset=0) for e in entries
        )


@dataclass
class BufferizePass(PrimFuncPass):
    """Collect lifetimes, schedule placements, and leave the ``PrimFunction`` body unchanged.

    Collect lifetimes, schedule placements, and (for the MVP trivial
    policy) leave the ``PrimFunction`` body unchanged. Real placement
    rewrites land here when the scheduler emits non-trivial pools.
    """

    collector: LifetimeCollector = None
    scheduler: BufferScheduler = None

    name: str = "bufferize"
    requires: tuple[str, ...] = ("hir_to_tir",)

    def __post_init__(self) -> None:
        if self.collector is None:
            self.collector = LifetimeCollector()
        if self.scheduler is None:
            self.scheduler = BufferScheduler()

    def run_prim_func(
        self, fn: PrimFunction, module: Module
    ) -> PrimFunction:
        entries = self.collector.collect(fn)
        self.scheduler.schedule(entries)


        return fn


__all__ = [
    "BufferizePass",
    "BufferEntry",
    "Placement",
    "LifetimeCollector",
    "BufferScheduler",
]
