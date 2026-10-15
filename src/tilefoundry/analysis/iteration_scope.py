"""Shared authored iteration scopes used by analysis families."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

import isl

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import dim_range
from tilefoundry.ir.types import Mesh, TensorType, Topology, TupleType
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
    BufferAlias,
    aligned_source,
    apply_source,
    bind_parameters,
    buffer_alias,
    eliminate_parameters,
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
                f"mesh level {topology.name!r} needs positive static "
                f"extents, got {count!r}"
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
    accesses: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    outputs: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    relations: dict[int, tuple[Call, tuple[AccessRelation, ...]]] = field(default_factory=dict)
    projections: dict[tuple[int, str | None], tuple[Call, tuple[AccessRelation, ...]]] = field(
        default_factory=dict
    )
    refused: dict[str, frozenset[Call]] = field(default_factory=dict)
    variance: dict[int, frozenset[int]] = field(default_factory=dict)

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

    def capture_root(self, value: Expr) -> Expr:
        return self.context.capture_root(value)

    def is_variant(self, value: Expr) -> bool:
        """Whether *value* depends on this loop's induction or carry values."""
        if not isinstance(self.owner, LoopRegion):
            return False
        return id(self) in self.variance.get(id(value), frozenset())

    def enclosing_loops(self) -> tuple[LoopRegion, ...]:
        return self.context.enclosing_loops()

    def enclosing_mesh(self) -> Mesh | None:
        return self.context.enclosing_mesh()

    def execution_mesh(self) -> Mesh | None:
        return self.context.current_mesh

    def mesh_positions(self, unit: str | None, stop=None) -> int:
        mesh = self.context.current_mesh
        return 1 if mesh is None or unit is None else _positions_at(mesh, unit, self.context.topologies)

    def repeats_of(self, call: Call, unit: str | None, stop=None) -> Repeats:
        """Return this Call's loop and mesh multiplicities in the current scope."""
        varying = loops = 1
        precision = AnalysisPrecision.EXACT
        cursor: IterationScope | None = self
        while cursor is not None and cursor is not stop:
            if isinstance(cursor.owner, LoopRegion):
                trips = cursor.trips()
                loops *= trips
                if id(cursor) in self.variance.get(
                    id(call), frozenset()
                ):
                    varying *= trips
                precision = precision.join(cursor.trips_precision)
            cursor = cursor.parent
        return Repeats(varying, loops, self.mesh_positions(unit, stop=stop), precision)

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
    ) -> None:
        super().__init__(root_function=graph)
        self.graph = graph
        self.is_root = is_root
        frame = CallFrame(module, graph) if ctx is None or ctx.frame is None else ctx.frame
        self.views = tuple(views)
        self.sources: dict[str, dict[int, BufferAlias | tuple]] = {view: {} for view in self.views}
        self.precisions: dict[str, dict[int, AnalysisPrecision | tuple]] = {
            view: {} for view in self.views
        }
        self.definitions: dict[int, IterationScope] = {}
        self.aliases: dict[int, BufferAlias | None] = {}
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
            scope.context, topology_level=self.unit_ctx.topology_level if view == "unit" else None
        )

    def _source_relation(
        self, source: BufferAlias | tuple, held, scope: IterationScope
    ) -> BufferAlias | tuple:
        """Carry one operand entry into a region, or union its inner loop points."""
        if isinstance(source, tuple):
            return tuple(
                self._source_relation(item, field, scope)
                for item, field in zip(source, held.fields, strict=True)
            )
        rank = len(held.shape) if isinstance(held, TensorType) else 1
        relation = aligned_source(source, rank, scope.depth, scope.domain_params)
        domain = scope.domain.insert_dims(
            isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
        )
        relation = relation.intersect_domain(domain)
        retained = dict(source.relation.values)
        retained.update(scope.domain_params)
        values = {
            relation.get_dim_name(isl.dim_type.PARAM, axis): retained[
                relation.get_dim_name(isl.dim_type.PARAM, axis)
            ] for axis in range(relation.dim(isl.dim_type.PARAM))
        }
        return replace(source, relation=AccessRelation(relation, values))

    def _identity_source(
        self, expr: Expr, held, scope: IterationScope, index: int = 0
    ) -> BufferAlias:
        """A material result names itself, with no data-coordinate renaming."""
        rank = len(held.shape) if isinstance(held, TensorType) else 1
        relation = identity_access(rank).relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
        domain = scope.domain.insert_dims(isl.dim_type.SET, scope.depth, rank)
        relation = relation.intersect_domain(domain)
        values = {
            name: value for name, value in scope.domain_params.items()
            if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
        }
        return BufferAlias(expr, AccessRelation(relation, values), index)

    def _material_source(self, expr: Expr, held, scope: IterationScope) -> BufferAlias | tuple:
        """Number tensor leaves first, then retain the result's type structure."""
        leaves = iter(
            self._identity_source(expr, leaf, scope, index)
            for index, leaf in enumerate(leaves_of(held))
        )
        return self._structured_source(expr, held, scope, leaves)

    def _structured_source(
        self, expr: Expr, held, scope: IterationScope, leaves: Iterator[BufferAlias]
    ) -> BufferAlias | tuple:
        """Reassemble an already indexed leaf sequence into one structured source."""
        if isinstance(held, TensorType):
            return next(leaves)
        if isinstance(held, TupleType):
            return tuple(
                self._structured_source(expr, field, scope, leaves) for field in held.fields
            )
        return self._identity_source(expr, held, scope)

    def _exact_precision(self, source: BufferAlias | tuple) -> AnalysisPrecision | tuple:
        """Mirror a material source's structure with exact leaf precisions."""
        if isinstance(source, tuple):
            return tuple(self._exact_precision(item) for item in source)
        return AnalysisPrecision.EXACT

    def _view_alias(
        self, expr: Expr, scope: IterationScope, view: str, local_relations
    ) -> BufferAlias | None:
        """Prove a registered view once logically, then use this view's coordinates."""
        if not isinstance(expr, Call):
            return None
        if id(expr) not in self.aliases:
            self.aliases[id(expr)] = (
                buffer_alias(expr, self._view_context("narrow", scope), scope.projected_relations(expr, self._view_context("narrow", scope)))
                if aliased_operand(expr) is not None else None
            )
        alias = self.aliases[id(expr)]
        if alias is not None and view == "unit":
            return replace(alias, relation=renaming_relation(expr, self.unit_ctx, local_relations))
        return alias

    def _view_source(
        self, alias: BufferAlias, scope: IterationScope, view: str
    ) -> tuple[BufferAlias, AnalysisPrecision]:
        """Eliminate this view's parameters before composing its operand's source."""
        ctx = self._view_context(view, scope)
        operand = self.sources[view][id(alias.buffer)]
        relation = alias.relation.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
        domain = scope.domain.insert_dims(
            isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
        )
        relation = relation.intersect_domain(domain)
        retained = dict(scope.domain_params)
        relation, precision = eliminate_parameters(
            relation, alias.relation.values, scope,
            ctx.local_type_of(alias.buffer) if view == "unit" else scope.capture_root(alias.buffer).type,
            narrow=view == "narrow", retained=retained,
        )
        held = ctx.local_type_of(alias.buffer)
        rank = len(held.shape) if isinstance(held, TensorType) else 1
        relation = apply_source(relation, operand, rank, scope.depth, scope.domain_params)
        retained.update(operand.relation.values)
        values = {
            relation.get_dim_name(isl.dim_type.PARAM, axis): retained[
                relation.get_dim_name(isl.dim_type.PARAM, axis)
            ] for axis in range(relation.dim(isl.dim_type.PARAM))
        }
        return (
            BufferAlias(
                operand.buffer, AccessRelation(relation, values), operand.index
            ),
            precision.join(self.precisions[view][id(alias.buffer)]),
        )

    def _record_source(
        self, expr: Expr, scope: IterationScope, view: str, local_relations=()
    ) -> None:
        table = self.sources[view]
        precisions = self.precisions[view]
        ctx = self._view_context(view, scope)
        if isinstance(expr, Tuple):
            table[id(expr)] = tuple(
                self._source_relation(table[id(item)], ctx.local_type_of(item), scope)
                for item in expr.elements
            )
            precisions[id(expr)] = tuple(precisions[id(item)] for item in expr.elements)
            return
        if isinstance(expr, Call) and isinstance(expr.target, TupleGetItem) and (
            isinstance(expr.args[1], Constant) and type(expr.args[1].value) is int
        ):
            index = expr.args[1].value
            table[id(expr)] = self._source_relation(
                table[id(expr.args[0])][index], ctx.local_type_of(expr), scope
            )
            precisions[id(expr)] = precisions[id(expr.args[0])][index]
            return
        alias = self._view_alias(expr, scope, view, local_relations)
        if alias is not None:
            table[id(expr)], precisions[id(expr)] = self._view_source(alias, scope, view)
            return
        table[id(expr)] = self._material_source(expr, ctx.local_type_of(expr), scope)
        precisions[id(expr)] = self._exact_precision(table[id(expr)])

    def _record_accesses(self, expr: Call, scope: IterationScope) -> None:
        if (
            isinstance(expr.target, Function)
            or not leaves_of(expr.type)
            or access_relation_registry.lookup(type(expr.target)) is None
        ):
            for view in self.views:
                self._record_source(expr, scope, view)
            return
        try:
            stated = relations_of(expr, self._view_context("narrow", scope))
            scope.relations[id(expr)] = (expr, stated)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            for view in self.views:
                self._record_source(expr, scope, view)
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
            return
        for view in self.views:
            ctx = self._view_context(view, scope)
            try:
                local_relations = scope.projected_relations(expr, ctx)
            except (NotImplementedError, TypeError, ValueError, isl.Error):
                self._record_source(expr, scope, view)
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
                continue
            self._record_source(expr, scope, view, local_relations)
            for inputs, boundaries, into in (
                (True, local_relations[:len(expr.args)], scope.accesses),
                (False, local_relations[len(expr.args):], scope.outputs),
            ):
                built: list[Access] = []
                for index, boundary in enumerate(boundaries):
                    operand = expr.args[index] if inputs else expr
                    access = resolve_access(
                        operand, boundary, scope, ctx,
                        input_index=index if inputs else None,
                        output_index=None if inputs else index,
                        narrow=view == "narrow", sources=self.sources[view],
                        precisions=self.precisions[view], source_scope=self.definitions[id(operand)],
                    )
                    if access is not None:
                        built.append(access)
                into.setdefault(view, {})[id(expr)] = (expr, tuple(built))

    def _record_variance(self, expr: Expr, operands: tuple[Expr, ...]) -> None:
        changing: set[int] = set()
        for operand in operands:
            changing.update(self.variance.get(id(operand), frozenset()))
        if (loop := self.seeds.get(id(expr))) is not None:
            changing.add(id(loop))
        self.variance[id(expr)] = frozenset(changing)

    def _record_call_accesses(self, expr: Call, scope: IterationScope, callee: CallFrame) -> None:
        """Compose the callee's formal access summary with this call's bindings."""
        definition = self.callees[id(callee.function)]

        def leaves(held):
            return (
                tuple(item for child in held for item in leaves(child))
                if isinstance(held, tuple)
                else (held,)
            )

        formals = tuple(
            param
            for param in callee.function.params
            if isinstance(expr.target, Function) or not param.is_const
        )
        for view in self.views:
            self._record_source(expr, scope, view)
            binding = scope.context.call_context(expr, view=view)
            returned = definition.sources[view][id(callee.function.body)]
            returned_precision = definition.precisions[view][id(callee.function.body)]

            def project_return(source, precision):
                if isinstance(source, tuple):
                    projected = tuple(
                        project_return(leaf, bounded)
                        for leaf, bounded in zip(source, precision, strict=True)
                    )
                    return tuple(item[0] for item in projected), tuple(item[1] for item in projected)
                projected, bounded = binding.project(source)
                position = next(
                    (index for index, formal in enumerate(formals) if source.buffer is formal), None
                )
                if position is not None:
                    actual_precision = leaves(
                        self.precisions[view][id(expr.args[position])]
                    )[source.index]
                    bounded = bounded.join(actual_precision)
                return projected, precision.join(bounded)

            self.sources[view][id(expr)], self.precisions[view][id(expr)] = project_return(
                returned, returned_precision
            )

            inputs = []
            for access, whole_tuple in definition.access_summary.get(view, ()):
                position = next(
                    (index for index, formal in enumerate(formals) if access.buffer is formal), None
                )
                if position is None:
                    continue
                formal = formals[position]
                if whole_tuple:
                    actual = BufferAlias(expr.args[position], identity_access(1))
                    rank = 1
                else:
                    actual = binding.actual(formal, access.index)
                    held = leaves_of(formal.type)[access.index]
                    rank = len(held.shape) if isinstance(held, TensorType) else 1
                relation = access.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
                relation = relation.intersect_domain(scope.domain)
                relation, precision = bind_parameters(
                    relation,
                    definition.access_parameters,
                    binding,
                    tuple(param for param, _argument in binding.bindings),
                    scope,
                    formal.type,
                    narrow=view == "narrow",
                )
                relation = apply_source(relation, actual, rank, scope.depth, scope.domain_params)
                source_precisions = leaves(self.precisions[view][id(expr.args[position])])
                selected_precisions = (
                    source_precisions if whole_tuple else (source_precisions[access.index],)
                )
                for source_precision in selected_precisions:
                    precision = precision.join(source_precision)
                inputs.append(
                    replace(
                        access,
                        input_index=position,
                        output_index=None,
                        relation=relation,
                        buffer=actual.buffer,
                        index=actual.index,
                        precision=access.precision.join(precision),
                    )
                )
            if any(inner.refused.get(view) for inner in walk_scopes(definition.root)):
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
            scope.accesses[view][id(expr)] = (expr, tuple(inputs))
            outputs = []
            source = self.sources[view][id(expr)]

            for index, entry in enumerate(leaves(source)):
                outputs.append(
                    Access(None, entry.relation.relation, expr, output_index=index, index=index)
                )
            scope.outputs.setdefault(view, {})[id(expr)] = (expr, tuple(outputs))

    def dispatch_visit(self, expr, ctx):
        self.definitions[id(expr)] = ctx.current
        return super().dispatch_visit(expr, ctx)

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
            self.definitions[id(param)] = child
            for view in self.views:
                self.sources[view][id(param)] = self._source_relation(
                    self.sources[view][id(argument)],
                    self._view_context(view, child).local_type_of(argument),
                    child,
                )
                self.precisions[view][id(param)] = self.precisions[view][id(argument)]
        return inner

    def default_visit_leaf(self, expr, operands, ctx):
        scope = ctx.current
        if isinstance(expr, (LoopRegion, MeshRegion)):
            scope = scope.parent
            for view in self.views:
                params = expr.params[: len(expr.yield_values)] if isinstance(expr, LoopRegion) else ()
                sources = tuple(self.sources[view][id(param)] for param in params)
                precisions = tuple(self.precisions[view][id(param)] for param in params)
                source = (
                    (sources[0] if len(sources) == 1 else sources)
                    if sources
                    else self.sources[view][id(expr.body)]
                )
                precision = (
                    (precisions[0] if len(precisions) == 1 else precisions)
                    if precisions
                    else self.precisions[view][id(expr.body)]
                )
                self.sources[view][id(expr)] = self._source_relation(
                    source,
                    self._view_context(view, scope).local_type_of(expr),
                    scope,
                )
                self.precisions[view][id(expr)] = precision
        elif isinstance(expr, Call):
            callee = self.callee_frame(expr, ctx.frame)
            if callee is None:
                self._record_accesses(expr, scope)
            else:
                self._record_call_accesses(expr, scope, callee)
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
