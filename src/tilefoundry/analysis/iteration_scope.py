"""Shared authored iteration scopes used by analysis families."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import isl

from tilefoundry.ir.core import Call, Expr
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.isl_interop import dim_range
from tilefoundry.ir.types import Mesh
from tilefoundry.ir.types.dim import DimSub, simplify_dim
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import expr_children
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
    projected,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

from .access import Access, AccessPrecision, resolve_access
from .errors import AnalysisError
from .loop_domain import induction_name, iteration_domain


@dataclass(eq=False)
class IterationScope:
    """One Function, authored loop, or mesh region, with all accesses below it."""

    owner: Function | LoopRegion | MeshRegion
    parent: IterationScope | None
    children: tuple[IterationScope, ...]
    depth: int
    domain: isl.set
    domain_params: dict[str, object] = field(default_factory=dict)
    accesses: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    outputs: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    relations: dict[int, tuple[Call, AccessRelations]] = field(default_factory=dict)
    refused: dict[str, frozenset[Call]] = field(default_factory=dict)
    _variance: dict[int, frozenset[int]] = field(default_factory=dict, repr=False)

    def stated_relations(self, call: Call, ctx: TypeInferContext) -> AccessRelations:
        """Return the Op-declared relations recorded in this scope chain."""
        cursor: IterationScope | None = self
        while cursor is not None:
            stored = cursor.relations.get(id(call))
            if stored is not None and stored[0] is call:
                return stored[1]
            cursor = cursor.parent
        return relations_of(call, ctx)

    def is_variant(self, value: Expr) -> bool:
        """Whether *value* depends on this loop's induction or carry values."""
        if not isinstance(self.owner, LoopRegion):
            return False
        root = self
        while root.parent is not None:
            root = root.parent
        return id(self) in root._variance.get(id(value), frozenset())

    def enclosing_loops(self) -> tuple[LoopRegion, ...]:
        """Return this scope's loop owners in outer-to-inner order."""
        loops = []
        cursor: IterationScope | None = self
        while cursor is not None:
            if isinstance(cursor.owner, LoopRegion):
                loops.append(cursor.owner)
            cursor = cursor.parent
        loops.reverse()
        return tuple(loops)

    def enclosing_mesh(self) -> Mesh | None:
        """Return the nearest enclosing mesh, or None when there is none."""
        cursor: IterationScope | None = self
        while cursor is not None:
            if isinstance(cursor.owner, MeshRegion):
                return cursor.owner.mesh
            cursor = cursor.parent
        return None

    def _counted(self) -> tuple[int, AccessPrecision]:
        """This scope's iteration count relative to its parent, and how exact it is.

        A loop runs its own span, not the grid it is entered from, so the count
        is read off ``stop - start``: exactly where that span reaches one value
        -- however it is written, so bounds naming an enclosing coordinate that
        cancel are exact -- and at its widest where it reaches several. The
        widest span is an upper bound on every coordinate's count, which is
        what every consumer of a trip count already reads it as.
        """
        if isinstance(self.owner, MeshRegion) or self.parent is None:
            return 1, AccessPrecision.EXACT
        owner = self.owner
        span = simplify_dim(DimSub, (owner.extent, owner.start))
        step = static_dim_value(owner.step)
        if step is None:
            raise AnalysisError(
                f"loop {induction_name(owner)!r} steps by {owner.step!r}, which is not "
                "a number; how often it runs is not stated"
            )
        widest = static_dim_value(span)
        precision = AccessPrecision.EXACT
        if widest is None:
            try:
                bounds = dim_range(span)
            except (ArithmeticError, TypeError, ValueError):
                bounds = None
            if bounds is None:
                raise AnalysisError(
                    f"loop {induction_name(owner)!r} runs from {owner.start!r} to "
                    f"{owner.extent!r}, which state no bounded span; its trip count "
                    "cannot be determined"
                )
            widest = bounds[1] - 1
            if bounds[1] - bounds[0] > 1:
                precision = AccessPrecision.WIDENED
        if step <= 0 or widest <= 0:
            return 1, precision
        return max(1, -(-widest // step)), precision

    def trips(self) -> int:
        """Return this scope's iteration count relative to its parent."""
        cached = getattr(self, "_trips_cache", None)
        if cached is None:
            cached = self._counted()
            self._trips_cache = cached
        return cached[0]

    @property
    def trips_precision(self) -> AccessPrecision:
        """How exactly :meth:`trips` counts this scope's iterations.

        ``EXACT`` where the span between the bounds is a number, ``WIDENED``
        where it is read at its widest because a bound names something no one
        here fixes -- a coordinate of the scope the loop is entered from.
        """
        self.trips()
        return self._trips_cache[1]


class ScopeBuilder:
    """Build one IterationScope tree and its access views for a Function."""

    def __init__(
        self,
        module: Module,
        graph: Function,
        *,
        views: Sequence[str] = ("narrow", "device"),
    ) -> None:
        self.graph = graph
        self.views = tuple(views)
        self.type_ctx = TypeInferContext(scope=FunctionScope(module, graph))
        self.seeds: dict[int, IterationScope]
        self.variance: dict[int, frozenset[int]]
        self.seen: set[int]

    def _empty_accesses(self) -> dict[str, dict[int, tuple[Call, tuple[Access, ...]]]]:
        return {view: {} for view in self.views}

    def _record_accesses(self, expr: Call, scope: IterationScope) -> None:
        if (
            isinstance(expr.target, Function)
            or access_relation_registry.lookup(type(expr.target)) is None
        ):
            return
        try:
            stated = relations_of(expr, self.type_ctx)
            scope.relations[id(expr)] = (expr, stated)
            local_relations = projected(stated, expr, self.type_ctx)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            for view in self.views:
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
            return
        for view in self.views:
            narrow = view == "narrow"
            built: list[Access] = []
            for index, boundary in enumerate(local_relations.inputs):
                if index >= len(expr.args):
                    continue
                access = resolve_access(
                    expr.args[index],
                    boundary,
                    scope,
                    self.type_ctx,
                    input_index=index,
                    narrow=narrow,
                )
                if access is not None:
                    built.append(access)
            scope.accesses.setdefault(view, {})[id(expr)] = (expr, tuple(built))
            written: list[Access] = []
            for output_index, boundary in enumerate(local_relations.outputs):
                access = resolve_access(
                    expr,
                    boundary,
                    scope,
                    self.type_ctx,
                    input_index=None,
                    output_index=output_index,
                    narrow=narrow,
                )
                if access is not None:
                    written.append(access)
            scope.outputs.setdefault(view, {})[id(expr)] = (expr, tuple(written))

    def _record_variance(self, expr: Expr, operands: tuple[Expr, ...]) -> None:
        changing: set[int] = set()
        for operand in operands:
            changing.update(self.variance.get(id(operand), frozenset()))
        if (loop := self.seeds.get(id(expr))) is not None:
            changing.add(id(loop))
        self.variance[id(expr)] = frozenset(changing)

    def _visit(self, expr: Expr, scope: IterationScope) -> None:
        if id(expr) in self.seen:
            return
        self.seen.add(id(expr))
        if isinstance(expr, LoopRegion):
            for operand in expr.init_args:
                self._visit(operand, scope)
            domain, domain_params = iteration_domain(expr, scope)
            child = IterationScope(
                owner=expr,
                parent=scope,
                children=(),
                depth=scope.depth + 1,
                domain=domain,
                domain_params=domain_params,
                accesses=self._empty_accesses(),
            )
            scope.children = (*scope.children, child)
            self.seeds[id(expr.induction_var)] = child
            for carried in expr.carried_args:
                self.seeds[id(carried)] = child
            self._visit(expr.body, child)
            for operand in expr.yield_values:
                self._visit(operand, child)
            self._record_variance(expr, expr_children(expr))
            return
        if isinstance(expr, MeshRegion):
            for operand in expr.args:
                self._visit(operand, scope)
            child = IterationScope(
                owner=expr,
                parent=scope,
                children=(),
                depth=scope.depth,
                domain=scope.domain,
                domain_params=scope.domain_params,
                accesses=self._empty_accesses(),
            )
            scope.children = (*scope.children, child)
            self._visit(expr.body, child)
            self._record_variance(expr, expr_children(expr))
            return
        operands = expr_children(expr)
        for operand in operands:
            self._visit(operand, scope)
        if isinstance(expr, Call):
            self._record_accesses(expr, scope)
        self._record_variance(expr, operands)

    def build(self) -> IterationScope:
        self.seeds = {}
        self.variance = {}
        self.seen = set()
        domain, domain_params = iteration_domain(self.graph, None)
        root = IterationScope(
            owner=self.graph,
            parent=None,
            children=(),
            depth=0,
            domain=domain,
            domain_params=domain_params,
            accesses=self._empty_accesses(),
        )
        for param in self.graph.params:
            self._visit(param, root)
        if self.graph.body is not None:
            self._visit(self.graph.body, root)
        root._variance = self.variance
        return root


def build_scopes(
    module: Module,
    graph: Function,
    *,
    views: Sequence[str] = ("narrow", "device"),
) -> IterationScope:
    """Build the iteration-scope tree and access views in one normalized walk."""
    return ScopeBuilder(module, graph, views=views).build()


def walk_scopes(root: IterationScope) -> Iterator[IterationScope]:
    """Yield an iteration scope and its descendants in lexical order."""
    yield root
    for child in root.children:
        yield from walk_scopes(child)


__all__ = [
    "IterationScope",
    "ScopeBuilder",
    "build_scopes",
    "walk_scopes",
]
