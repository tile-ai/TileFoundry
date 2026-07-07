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
from tilefoundry.ir.tir.stmts import (
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    While,
)
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


class LifetimeCollector:
    """Walk a ``PrimFunction`` body and collect every ``LetStmt`` that binds
    a ``Call(AllocTensor, ...)`` together with its lifetime range.

    MVP impl emits a flat list in pre-order. Subclasses can override
    ``collect`` to do real liveness analysis (use-def dataflow). The hook
    boundary is intentionally narrow so swapping it does not ripple into
    the pass.
    """

    def collect(self, fn: PrimFunction) -> tuple[BufferEntry, ...]:
        entries: list[BufferEntry] = []
        counter = [0]

        def emit_entry(var: Var, alloc: AllocTensor, point: int) -> None:
            entries.append(
                BufferEntry(
                    var=var, alloc=alloc, defined_at=point, last_use_at=point
                )
            )

        def walk(stmt) -> None:
            counter[0] += 1
            point = counter[0]
            if isinstance(stmt, LetStmt):
                if isinstance(stmt.value, Call) and isinstance(
                    stmt.value.target, AllocTensor
                ):
                    emit_entry(stmt.var, stmt.value.target, point)
                walk(stmt.body)
                return
            if isinstance(stmt, Sequential):
                for s in stmt.body:
                    walk(s)
                return
            if isinstance(stmt, (For, While, MeshScope)):
                walk(stmt.body)
                return
            if isinstance(stmt, If):
                walk(stmt.then_body)
                walk(stmt.else_body)
                return

        walk(fn.body)
        return tuple(entries)


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
    """Collect lifetimes, schedule placements, and (for the MVP trivial
    policy) leave the ``PrimFunction`` body unchanged. Real placement
    rewrites land here when the scheduler emits non-trivial pools."""

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
        # MVP trivial policy: independent physical allocation per logical
        # buffer == the IR we already have. Return identity.
        return fn


__all__ = [
    "BufferizePass",
    "BufferEntry",
    "Placement",
    "LifetimeCollector",
    "BufferScheduler",
]
