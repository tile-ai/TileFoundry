"""Memory-family projection from shared Scope and Access records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Constant, Expr, Var, binding_name
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import bytes_by_storage
from tilefoundry.ir.visitor import ExprVisitor, expr_children
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope

from .errors import AnalysisError
from .facts import MemoryHierarchyFacts
from .metadata import (
    AllocationMetadata,
    LevelFootprint,
    LoopFootprintMetadata,
    MemoryMetadata,
    TrafficBytes,
    TrafficMetadata,
    ValueLifetime,
)
from .movement import add_traffic, call_traffic
from .visitor import AnalyzeContext

SELECTOR = "memory"


@dataclass(frozen=True)
class MemoryOptions:
    """How long the placement may look, and how to reproduce what it found."""

    timeout_seconds: float = 60.0
    workers: int = 1
    random_seed: int = 0


def _lifetimes(
    values: list[Expr], facts: MemoryHierarchyFacts
) -> tuple[ValueLifetime, ...]:
    index_by_id = {id(expr): index for index, expr in enumerate(values)}
    last_by_id = dict(index_by_id)
    for index, consumer in enumerate(values):
        for operand in expr_children(consumer):
            if id(operand) in last_by_id and index > last_by_id[id(operand)]:
                last_by_id[id(operand)] = index
    result: list[ValueLifetime] = []
    for index, expr in enumerate(values):
        for level, amount in bytes_by_storage(expr.type).items():
            if facts.explicit(level) is None:
                continue
            result.append(ValueLifetime(
                binding=(getattr(expr, "name", None) or binding_name(expr) or f"<value {index}>"),
                level=level,
                bytes=amount,
                defined_at=index,
                last_used_at=(len(values) - 1 if isinstance(expr, Var) else last_by_id[id(expr)]),
                persistent=isinstance(expr, Var),
            ))
    return tuple(result)


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    whole: CostContext | None = None
    local: CostContext | None = None
    totals: dict[str, TrafficBytes] = field(default_factory=dict)
    shares: dict[str, TrafficBytes] = field(default_factory=dict)
    values: list[Expr] = field(default_factory=list)


class MemoryVisitor(ExprVisitor[None]):
    """Attach per-Call traffic while collecting lifetime order and loop footprints."""

    def visit_GridRegionExpr(
        self, expr: GridRegionExpr, ctx: MemoryContext
    ) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.init_args:
            self.visit(operand, ctx)
        self.visit(expr.body, inner)
        for operand in expr.yield_values:
            self.visit(operand, inner)
        attach(expr, child.footprint())

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: MemoryContext
    ) -> None:
        if isinstance(expr, (Call, Constant)):
            ctx.values.append(expr)
        if not isinstance(expr, Call):
            return
        recorded = id(expr) in ctx.current.accesses["narrow"]
        if ctx.whole is None or ctx.local is None:
            raise AnalysisError("memory: visitor context is missing cost contexts")
        moved = call_traffic(expr, ctx.whole, ctx.local) if recorded else TrafficMetadata()
        attach(expr, moved)
        if not recorded:
            return
        repeats = 1
        cursor = ctx.current
        while cursor.parent is not None:
            if cursor.is_variant(expr):
                repeats *= max(1, cursor.trips())
            cursor = cursor.parent
        add_traffic(ctx.totals, ctx.shares, moved, repeats)


def analyze_memory(function: Function, context: AnalyzeContext) -> None:
    """Attach traffic and per-loop footprints from the shared Scope tree."""
    module = context.module
    level = context.level
    facts = context.target.get_facts(MemoryHierarchyFacts)
    topologies = module.effective_topologies()
    whole = CostContext(scope=FunctionScope(module, function))
    local = CostContext(
        scope=FunctionScope(module, function), level=level, topologies=topologies
    )
    memory_context = MemoryContext(
        module=module,
        target=context.target,
        level=level,
        options=context.options,
        root=context.root,
        current=context.current,
        whole=whole,
        local=local,
        values=list(function.params),
    )
    MemoryVisitor().visit(function.body, memory_context)
    attach(
        function,
        TrafficMetadata(
            whole=tuple(sorted(memory_context.totals.items())),
            per_unit=tuple(sorted(memory_context.shares.items())),
        ),
    )
    lifetimes = _lifetimes(memory_context.values, facts)
    topology_size = 1
    if topologies and isinstance(topologies[0].size, int):
        topology_size = max(1, topologies[0].size)
    levels_list: list[LevelFootprint] = []
    for name in sorted({item.level for item in lifetimes} | set(memory_context.shares)):
        declared = facts.explicit(name)
        divisor = topology_size if declared is not None and declared.owner != "target" else 1
        rows = [item for item in lifetimes if item.level == name]
        peak = 0
        for point in range(len(lifetimes) + 1):
            peak = max(
                peak,
                sum(item.bytes // divisor for item in rows if item.defined_at <= point <= item.last_used_at),
            )
        levels_list.append(LevelFootprint(
            level=name,
            peak_bytes=peak,
            persistent_bytes=sum(item.bytes // divisor for item in rows if item.persistent),
            capacity_bytes=declared.capacity_bytes if declared is not None else None,
        ))
    levels = tuple(levels_list)
    for level in levels:
        if level.capacity_bytes is not None and level.peak_bytes > level.capacity_bytes:
            raise AnalysisError(
                f"{level.level!r} holds {level.peak_bytes} B at one point of this program, "
                f"more than the {level.capacity_bytes} B the target states for that level"
            )
    attach(
        function,
        MemoryMetadata(
            footprint=levels,
            lifetimes=lifetimes,
            allocation=AllocationMetadata(solver_status="optimal"),
        ),
    )


def cache_pressure(
    record: LoopFootprintMetadata,
    facts: MemoryHierarchyFacts,
    peaks: dict[str, int],
) -> tuple[dict[str, object], ...]:
    """Compare one scope's device footprint with same-scope implicit caches."""
    rows: list[dict[str, object]] = []
    for level in facts.implicit_levels:
        backing_name = facts.backing_level(level.name)
        backing = facts.explicit(backing_name)
        if backing is None or backing.scope != level.scope:
            continue
        accesses = tuple(item for item in record.footprints if item.level == backing_name)
        if not accesses or any(item.device_bytes < item.bytes for item in accesses):
            continue
        working_set = sum(item.device_bytes for item in accesses)
        capacity = level.capacity_bytes
        for peer, shared_bytes in facts.capacity_sharers(level.name):
            if shared_bytes is None:
                continue
            remaining = shared_bytes - peaks.get(peer, 0)
            capacity = remaining if capacity is None else min(capacity, remaining)
        status = (
            "unknown" if capacity is None else
            "exceeds" if working_set > capacity else
            "fits" if record.known else "lower-bound"
        )
        rows.append({
            "cache_level": level.name,
            "backing_level": backing_name,
            "device_bytes": working_set,
            "capacity_bytes": capacity,
            "status": status,
        })
    return tuple(rows)


__all__ = ["MemoryOptions", "SELECTOR", "analyze_memory", "cache_pressure"]
