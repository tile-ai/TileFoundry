"""Shared authored iteration scopes used by analysis families."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property

import isl

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.isl_interop import dim_range
from tilefoundry.ir.types import Mesh, TensorType, Topology
from tilefoundry.ir.types.dim import DimSub, simplify_dim
from tilefoundry.ir.types.layout import ComposedLayout, get, size
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import expr_children
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    access_relation_registry,
    identity_access,
    leaves_of,
    projected,
    relations_of,
    renaming_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.call_frame import CallFrame, ValuationVisitor
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .access import (
    Access,
    AliasKind,
    BufferAlias,
    SSAValue,
    aligned_source,
    apply_source,
    buffer_alias,
    eliminate_parameters,
    forwarded_leaves,
    referenced_values,
    resolve_access,
)
from .errors import AnalysisError
from .loop_domain import induction_name, iteration_domain
from .metadata import ExecutionMeshMetadata
from .precision import AnalysisPrecision
from .visitor import AnalyzeContext


def _positions_at(mesh: Mesh, unit: str, topologies: tuple[Topology, ...]) -> int:
    """Count mesh positions at the selected unit and all coarser topology levels."""
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    selected = declared[unit]
    stated = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
    positions = 1
    for index, topology in enumerate(mesh.topologies):
        if declared[topology.name] > selected:
            continue
        count = size(get(stated, index))
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise AnalysisError(
                f"mesh level {topology.name!r} needs positive static extents, got {count!r}"
            )
        positions *= count
    return positions


@dataclass(frozen=True)
class Repeats:
    """Separate variant trips, all trips, executing positions, and trip precision."""

    varying_loop_trips: int
    loop_trips: int
    mesh_positions: int
    trips_precision: AnalysisPrecision


@dataclass(eq=False)
class IterationScope:
    """One Function, authored loop, or mesh region, with all accesses below it."""

    owner: Function | LoopRegion | MeshRegion
    parent: IterationScope | None
    children: tuple[IterationScope, ...]
    depth: int
    domain: isl.set
    context: AnalyzeContext
    domain_params: dict[str, object] = field(default_factory=dict)
    sources: dict[str, dict[SSAValue, BufferAlias]] = field(default_factory=dict)
    precisions: dict[str, dict[SSAValue, AnalysisPrecision]] = field(default_factory=dict)
    accesses: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    outputs: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    relations: dict[int, tuple[Call, tuple[AccessRelation, ...]]] = field(default_factory=dict)
    projections: dict[tuple[int, str | None], tuple[Call, tuple[AccessRelation, ...]]] = field(
        default_factory=dict
    )
    refused: dict[str, frozenset[Call]] = field(default_factory=dict)
    variance: dict[int, frozenset[int]] = field(default_factory=dict)

    @cached_property
    def capture_names(self):
        """The first authored capture name for each source in this definition."""
        names = {}
        for scope in walk_scopes(self):
            for param, _argument in scope.captures:
                names.setdefault(scope.context.capture_root(param), param)
        return names

    @property
    def captures(self):
        return self.owner.captures() if isinstance(self.owner, (MeshRegion, LoopRegion)) else ()

    def projected_relations(self, call: Call, ctx: TypeInferContext) -> tuple[AccessRelation, ...]:
        """Return *call*'s relations in *ctx*'s topology window, projecting it once.

        Cache on the scope that recorded the call's stated relations, keyed by
        the window. Contexts that substitute types must project on their own.
        """
        holder, stated = self, None
        cursor: IterationScope | None = self
        while cursor is not None:
            recorded = cursor.relations.get(id(call))
            if recorded is not None and recorded[0] is call:
                holder, stated = cursor, recorded[1]
                break
            cursor = cursor.parent
        key = (id(call), getattr(ctx, "topology_level", None))
        held = holder.projections.get(key)
        if held is not None and held[0] is call:
            return held[1]
        local = projected(stated if stated is not None else relations_of(call, ctx), call, ctx)
        holder.projections[key] = (call, local)
        return local

    def is_variant(self, value: Expr) -> bool:
        """Whether *value* depends on this loop's induction or carry values."""
        if not isinstance(self.owner, LoopRegion):
            return False
        return id(self) in self.variance.get(id(value), frozenset())

    def mesh_positions(self, unit: str | None) -> int:
        mesh = self.context.current_mesh
        return (
            1
            if mesh is None or unit is None
            else _positions_at(mesh, unit, self.context.topologies)
        )

    def repeats_of(self, call: Call, unit: str | None, stop=None) -> Repeats:
        """Return this Call's loop and mesh multiplicities in the current scope."""
        varying = loops = 1
        precision = AnalysisPrecision.EXACT
        cursor: IterationScope | None = self
        while cursor is not None and cursor is not stop:
            if isinstance(cursor.owner, LoopRegion):
                trips = cursor.trips()
                loops *= trips
                if id(cursor) in self.variance.get(id(call), frozenset()):
                    varying *= trips
                precision = precision.join(cursor.trips_precision)
            cursor = cursor.parent
        return Repeats(varying, loops, self.mesh_positions(unit), precision)

    def _counted(self) -> tuple[int, AnalysisPrecision]:
        """This scope's iteration count relative to its parent, and how exact it is.

        A loop runs its own span, not the grid it is entered from, so the count
        is read off ``stop - start``: exactly where that span reaches one value
        -- however it is written, so bounds naming an enclosing coordinate that
        cancel are exact -- and at its widest where it reaches several. The
        widest span is an upper bound on every coordinate's count, which is
        what every consumer of a trip count already reads it as.
        """
        if isinstance(self.owner, MeshRegion) or self.parent is None:
            return 1, AnalysisPrecision.EXACT
        owner = self.owner
        span = simplify_dim(DimSub, (owner.extent, owner.start))
        step = static_dim_value(owner.step)
        if step is None:
            raise AnalysisError(
                f"loop {induction_name(owner)!r} steps by {owner.step!r}, which is not "
                "a number; how often it runs is not stated"
            )
        widest = static_dim_value(span)
        precision = AnalysisPrecision.EXACT
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
                precision = AnalysisPrecision.UPPER_BOUND
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
    def trips_precision(self) -> AnalysisPrecision:
        """How exactly :meth:`trips` counts this scope's iterations.

        ``EXACT`` where the span between the bounds is a number, ``UPPER_BOUND``
        where it is read at its widest because a bound names something no one
        here fixes -- a coordinate of the scope the loop is entered from.
        """
        self.trips()
        return self._trips_cache[1]


class ScopeBuilder(ValuationVisitor[None]):
    """Build one IterationScope tree and its access views for a Function.

    Scopes record each call's logical relations, whatever level the caller's
    context reports, so the builder reads through that context without a level.
    """

    def __init__(
        self,
        module: Module,
        graph: Function,
        *,
        views: Sequence[str] = ("narrow", "device", "unit"),
        ctx: AnalyzeContext | None = None,
        is_root: bool = True,
        returns: Mapping[int, Mapping[str, tuple[tuple[BufferAlias, AnalysisPrecision], ...]]]
        | None = None,
    ) -> None:
        super().__init__(root_function=graph)
        self.graph = graph
        self.is_root = is_root
        self.returns = {} if returns is None else returns
        frame = CallFrame(module, graph) if ctx is None or ctx.frame is None else ctx.frame
        self.views = tuple(views)
        self.sources: dict[str, dict[SSAValue, BufferAlias]] = {view: {} for view in self.views}
        self.precisions: dict[str, dict[SSAValue, AnalysisPrecision]] = {
            view: {} for view in self.views
        }
        self.type_ctx = (
            AnalyzeContext(frame=frame, topologies=frame.topologies)
            if ctx is None
            else replace(ctx, topology_level=None)
        )
        self.unit_ctx = ctx if ctx is not None else self.type_ctx
        self.seeds: dict[int, IterationScope]
        self.variance: dict[int, frozenset[int]]

    def _empty_accesses(self) -> dict[str, dict[int, tuple[Call, tuple[Access, ...]]]]:
        return {view: {} for view in self.views}

    def _view_context(self, view: str, scope: IterationScope) -> TypeInferContext:
        return replace(
            scope.context,
            topology_level=self.unit_ctx.topology_level if view == "unit" else None,
            access_view=view,
        )

    def _source_relation(self, source: BufferAlias, held, scope: IterationScope) -> BufferAlias:
        """Align one cached leaf source with this region's loop coordinates."""
        rank = len(held.shape) if isinstance(held, TensorType) else 1
        relation = aligned_source(source, rank, scope.depth, scope.domain_params)
        relation = relation.intersect_domain(
            scope.domain.insert_dims(
                isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
            )
        )
        retained = {**source.relation.values, **scope.domain_params}
        values = {
            name: retained[name]
            for axis in range(relation.dim(isl.dim_type.PARAM))
            if (name := relation.get_dim_name(isl.dim_type.PARAM, axis)) in retained
        }
        return replace(source, relation=AccessRelation(relation, values))

    def _identity_source(self, value: SSAValue, held, scope: IterationScope) -> BufferAlias:
        """A material leaf names itself, with no data-coordinate renaming."""
        rank = len(held.shape) if isinstance(held, TensorType) else 1
        relation = identity_access(rank).relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
        relation = relation.intersect_domain(
            scope.domain.insert_dims(isl.dim_type.SET, scope.depth, rank)
        )
        values = referenced_values(relation, scope.domain_params)
        return BufferAlias(value, AccessRelation(relation, values), AliasKind.FORWARD)

    def _forward(self, expr, operands, scope, view):
        """Cache flattened sources once for tuple, capture and region forwarding."""
        types = leaves_of(self._view_context(view, scope).local_type_of(expr))
        for index, (operand, held) in enumerate(zip(operands, types, strict=True)):
            value = SSAValue(expr, index)
            self.sources[view][value] = self._source_relation(
                self.sources[view][operand], held, scope
            )
            self.precisions[view][value] = self.precisions[view][operand]

    def _record_source(self, expr, scope, view, local_relations=(), alias=None):
        """Cache each leaf's fully composed source before any access reads it."""
        table, precisions = self.sources[view], self.precisions[view]
        ctx = self._view_context(view, scope)
        if (operands := forwarded_leaves(expr, ctx)) is not None:
            self._forward(expr, operands, scope, view)
            return
        if alias is not None and view == "unit":
            alias = replace(alias, relation=renaming_relation(expr, ctx, local_relations))
        if alias is not None:
            source = table[alias.target]
            relation = alias.relation.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
            relation = relation.intersect_domain(
                scope.domain.insert_dims(
                    isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
                )
            )
            retained = dict(scope.domain_params)
            relation, precision = eliminate_parameters(
                relation,
                alias.relation.values,
                scope,
                ctx.local_type_of(alias.target.expr)
                if view == "unit"
                else scope.context.capture_root(alias.target.expr).type,
                narrow=view == "narrow",
                retained=retained,
                formals=() if self.is_root else self.graph.params,
            )
            held = ctx.local_type_of(alias.target.expr)
            rank = len(held.shape) if isinstance(held, TensorType) else 1
            relation = apply_source(relation, source, rank, scope.depth, scope.domain_params)
            retained.update(source.relation.values)
            values = {
                name: retained[name]
                for axis in range(relation.dim(isl.dim_type.PARAM))
                if (name := relation.get_dim_name(isl.dim_type.PARAM, axis)) in retained
            }
            table[SSAValue(expr)] = BufferAlias(
                source.target, AccessRelation(relation, values), AliasKind.VIEW
            )
            precisions[SSAValue(expr)] = precision.join(precisions[alias.target])
            return
        held = ctx.local_type_of(expr)
        for index, leaf in enumerate(leaves_of(held) or (held,)):
            value = SSAValue(expr, index)
            table[value] = self._identity_source(value, leaf, scope)
            precisions[value] = AnalysisPrecision.EXACT

    def _record_accesses(self, expr: Call, scope: IterationScope) -> None:
        if (
            isinstance(expr.target, Function)
            or not leaves_of(expr.type)
            or access_relation_registry.lookup(type(expr.target)) is None
        ):
            for view in self.views:
                self._record_source(expr, scope, view)
            return
        logical = self._view_context("narrow", scope)
        try:
            stated = relations_of(expr, logical)
            scope.relations[id(expr)] = (expr, stated)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            for view in self.views:
                self._record_source(expr, scope, view)
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
            return
        alias = (
            buffer_alias(expr, logical, scope.projected_relations(expr, logical))
            if aliased_operand(expr) is not None
            else None
        )
        for view in self.views:
            ctx = self._view_context(view, scope)
            try:
                local_relations = scope.projected_relations(expr, ctx)
            except (NotImplementedError, TypeError, ValueError, isl.Error):
                self._record_source(expr, scope, view)
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
                continue
            self._record_source(expr, scope, view, local_relations, alias)
            for inputs, boundaries, into in (
                (True, local_relations[: len(expr.args)], scope.accesses),
                (False, local_relations[len(expr.args) :], scope.outputs),
            ):
                built: list[Access] = []
                for index, boundary in enumerate(boundaries):
                    operand = expr.args[index] if inputs else expr
                    access = resolve_access(
                        operand,
                        boundary,
                        scope,
                        ctx,
                        input_index=index if inputs else None,
                        output_index=None if inputs else index,
                        narrow=view == "narrow",
                        sources=self.sources[view],
                        precisions=self.precisions[view],
                        formals=() if self.is_root else self.graph.params,
                    )
                    built.extend(access)
                into.setdefault(view, {})[id(expr)] = (expr, tuple(built))

    def _record_variance(self, expr: Expr, operands: tuple[Expr, ...]) -> None:
        changing: set[int] = set()
        for operand in operands:
            changing.update(self.variance.get(id(operand), frozenset()))
        if (loop := self.seeds.get(id(expr))) is not None:
            changing.add(id(loop))
        self.variance[id(expr)] = frozenset(changing)

    def region_context(self, expr, ctx):
        scope = ctx.current
        domain, parameters = (
            iteration_domain(expr, scope)
            if isinstance(expr, LoopRegion)
            else (scope.domain, scope.domain_params)
        )
        child = IterationScope(
            owner=expr,
            parent=scope,
            children=(),
            depth=scope.depth + isinstance(expr, LoopRegion),
            domain=domain,
            domain_params=parameters,
            context=ctx,
            accesses=self._empty_accesses(),
            variance=self.variance,
            sources=self.sources,
            precisions=self.precisions,
        )
        scope.children = (*scope.children, child)
        inner = super().region_context(expr, ctx)
        child.context = inner
        if isinstance(expr, LoopRegion):
            for param in (expr.induction_var, *expr.params[: len(expr.yield_values)]):
                self.seeds[id(param)] = child
        for param, argument in expr.captures():
            self._record_variance(param, (argument,))
            self._memo[id(param)] = (param, None)
            for view in self.views:
                self._forward(param, forwarded_leaves(param, inner), child, view)
        return inner

    def default_visit_leaf(self, expr, operands, ctx):
        scope = ctx.current
        if isinstance(expr, (LoopRegion, MeshRegion)):
            scope = scope.parent
            for view in self.views:
                self._forward(expr, forwarded_leaves(expr, ctx), scope, view)
        elif isinstance(expr, Call):
            callee = self.callee_frame(expr, ctx.frame)
            if callee is None:
                self._record_accesses(expr, scope)
            else:
                for view in self.views:
                    binding = self._view_context(view, scope).for_call(expr)
                    for index, (source, precision) in enumerate(
                        self.returns[id(callee.function)][view]
                    ):
                        value = SSAValue(expr, index)
                        projected, bounded = binding.project(source)
                        self.sources[view][value] = projected
                        self.precisions[view][value] = precision.join(bounded)
        else:
            for view in self.views:
                self._record_source(expr, scope, view)
        self._record_variance(expr, expr_children(expr))

    def build(self) -> IterationScope:
        self.seeds = {}
        self.variance = {}
        domain, domain_params = iteration_domain(self.graph, None)
        mesh = get_metadata(self.graph, ExecutionMeshMetadata)
        context = replace(
            self.type_ctx,
            owner=self.graph,
            parent=None,
            current_mesh=None if mesh is None else mesh.mesh,
        )
        root = IterationScope(
            owner=self.graph,
            parent=None,
            children=(),
            depth=0,
            domain=domain,
            domain_params=domain_params,
            context=context,
            accesses=self._empty_accesses(),
            variance=self.variance,
            sources=self.sources,
            precisions=self.precisions,
        )
        context.root = context.current = root
        for param in self.graph.params:
            self.visit(param, context)
            self.variance[id(param)] = frozenset((id(param),))
        self.visit_function_body(self.graph, context)
        return root


def build_scopes(
    module: Module,
    graph: Function,
    *,
    views: Sequence[str] = ("narrow", "device", "unit"),
    ctx: AnalyzeContext | None = None,
) -> IterationScope:
    """Build the iteration-scope tree and access views in one normalized walk."""
    return ScopeBuilder(module, graph, views=views, ctx=ctx).build()


def walk_scopes(root: IterationScope) -> Iterator[IterationScope]:
    """Yield an iteration scope and its descendants in lexical order."""
    yield root
    for child in root.children:
        yield from walk_scopes(child)


__all__ = [
    "IterationScope",
    "Repeats",
    "ScopeBuilder",
    "build_scopes",
    "walk_scopes",
]
