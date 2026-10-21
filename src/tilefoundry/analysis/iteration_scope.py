"""Shared authored iteration scopes used by analysis families."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import isl

from tilefoundry.ir.core import Expr, Var, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.isl_interop import dim_range
from tilefoundry.ir.types import Mesh, Topology
from tilefoundry.ir.types.dim import DimSub, simplify_dim
from tilefoundry.ir.types.layout import ComposedLayout, get, size
from tilefoundry.ir.types.utils import leaf_rank, leaf_span, static_dim_value
from tilefoundry.ir.visitor import expr_children
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    access_relation_registry,
    identity_access,
    projected,
    relations_of,
    renaming_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.valuation import ValuationVisitor

from .access import (
    SnapshotEntry,
    _RegionBindingResolver,
    buffer_alias,
    eliminate_parameters,
    resolve_access,
)
from .errors import AnalysisError
from .loop_domain import induction_name, iteration_domain, refuse_runtime_bound
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


@dataclass
class DefinitionSnapshot:
    """Facts computed once for a lexical definition node."""

    owner: object
    parent: IterationScope | None = None
    children: tuple = ()
    domain: AccessRelation = field(default_factory=lambda: AccessRelation(isl.map("{ [] -> [] }")))
    mesh: Mesh | None = None
    topologies: tuple = ()
    values: dict = field(default_factory=dict)
    locations: dict = field(default_factory=dict)
    accesses: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)


@dataclass
class ResolvedSnapshot:
    """Derived results in one fixed lexical and calling environment."""

    values: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
    allocations: dict = field(default_factory=dict)
    scopes: dict = field(default_factory=dict)
    accesses: dict = field(default_factory=dict)


@dataclass
class ScopeSnapshot:
    definition: DefinitionSnapshot
    attachments: dict = field(default_factory=dict)
    resolved: ResolvedSnapshot = field(default_factory=ResolvedSnapshot)


@dataclass
class ValueProvenance:
    """One definition's value facts, interpreted through optional caller bindings."""

    scope: IterationScope

    @property
    def snapshot(self):
        return self.scope.snapshot.definition.values

    @property
    def arguments(self):
        return self.scope._binding._arguments

    @property
    def caller_provenance(self):
        caller = self.scope.caller_scope
        return None if caller is None else caller.provenance

    @property
    def resolved(self):
        return self.scope.snapshot.resolved.values

    def lookup(self, value: SSAValue) -> SnapshotEntry:
        if value in self.resolved:
            return self.resolved[value]
        entry = self.snapshot.get(value)
        if entry is None:
            if self.scope.parent is not None:
                return self.scope.parent.provenance.lookup(value)
            raise AnalysisError(f"no source recorded for {value.expr!r}")
        if self.caller_provenance is None:
            return entry
        self.resolved[value] = self.resolve(entry)
        return self.resolved[value]

    def resolve(self, entry):
        caller_provenance = self.caller_provenance
        argument = self.arguments.get(id(entry.root.expr))
        actual = (
            None
            if argument is None
            else caller_provenance.lookup(SSAValue(argument, entry.root.index))
        )
        changing = set(entry.varies)
        for formal, bound in self.arguments.items():
            if formal in changing:
                changing.remove(formal)
                changing.update(caller_provenance.varies(bound))
        relations = {}
        for view in ("narrow", "device", "unit"):
            relation = getattr(entry, view)
            if relation is not None:
                relation = self.scope.access.project(relation, view=view)
                if actual is not None:
                    source = getattr(actual, view)
                    relation = None if source is None else relation.compose(source)
            relations[view] = relation
        return replace(
            entry,
            root=entry.root if actual is None else actual.root,
            varies=frozenset(changing),
            **relations,
        )

    def actual(self, formal, index=0):
        if self.caller_provenance is None or id(formal) not in self.arguments:
            raise AnalysisError("parameter has no binding in this calling context")
        return self.caller_provenance.lookup(SSAValue(self.arguments[id(formal)], index))

    def result(self, index=0):
        return self.lookup(SSAValue(self.scope.root.owner.body, index))

    def varies(self, value):
        argument = self.arguments.get(id(value))
        if argument is not None and self.caller_provenance is not None:
            return self.caller_provenance.varies(argument)
        entry = self.snapshot.get(SSAValue(value))
        changing = set() if entry is None else set(entry.varies)
        if self.caller_provenance is not None:
            for formal, argument in self.arguments.items():
                if formal in changing:
                    changing.remove(formal)
                    changing.update(self.caller_provenance.varies(argument))
        return frozenset(changing)

    def dependencies(self, value):
        changing = self.varies(value)
        dependencies = []
        provenance = self
        while provenance is not None:
            scope = provenance.scope
            dependencies.extend(p for p in scope.root.owner.params if id(p) in changing)
            dependencies.extend(
                loop.owner.induction_var for loop in scope.loop_scopes if id(loop.owner) in changing
            )
            provenance = provenance.caller_provenance
        return tuple(dict.fromkeys(dependencies))


@dataclass
class AccessAnalysis:
    """Declared boundaries, resolved accesses, and projections in one environment."""

    scope: IterationScope

    @property
    def accesses(self):
        return self._bound_accesses("accesses")

    @property
    def outputs(self):
        return self._bound_accesses("outputs")

    @property
    def relations(self):
        return self.scope.snapshot.definition.relations

    @property
    def refused(self):
        definition = self.scope.snapshot.definition.refused
        if self.scope.caller_scope is None:
            return definition
        cache = self.scope.snapshot.resolved.accesses
        if "refused" not in cache:
            cache["refused"] = dict(definition)
        return cache["refused"]

    @property
    def projections(self):
        return self.scope.snapshot.resolved.relations

    @property
    def instance(self):
        return self.scope._binding._instance

    def _bound_accesses(self, name):
        definition = getattr(self.scope.snapshot.definition, name)
        if self.scope.caller_scope is None:
            return definition
        cache = self.scope.snapshot.resolved.accesses
        if name in cache:
            return cache[name]
        callee_scope = self.scope
        snapshot = callee_scope.caller_scope.snapshot.definition.values
        result = {}
        for view, records in definition.items():
            into = result.setdefault(view, {})
            for path, (primitive, items) in records.items():
                scope = callee_scope.at(path[:1])
                translated = []
                for item in items:
                    source = scope.provenance.lookup(item.target)
                    if source.root not in snapshot:
                        snapshot[source.root] = replace(
                            source,
                            forward=None,
                            **{
                                kind: None
                                if getattr(source, kind) is None
                                else identity_access(
                                    getattr(source, kind).relation.dim(isl.dim_type.OUT)
                                )
                                for kind in ("narrow", "device", "unit")
                            },
                        )
                    relation = scope.access.project(item.relation, view=view)
                    if id(item.target.expr) in callee_scope.provenance.arguments:
                        upstream = getattr(source, view)
                        if upstream is None:
                            refused = self.refused.get(view, frozenset())
                            self.refused[view] = refused | {primitive}
                            continue
                        relation = relation.compose(upstream)
                    if callee_scope.access.instance is not None and getattr(
                        item.target.expr, "is_const", False
                    ):
                        instance = callee_scope.access.instance
                        prefix = instance.relation.add_dims(
                            isl.dim_type.IN, relation.relation.dim(isl.dim_type.IN)
                        )
                        relation = relation._result(
                            prefix.flat_range_product(relation.relation),
                            instance.values,
                            may_access=instance.may_access,
                        )
                    translated.append(replace(item, target=source.root, relation=relation))
                into[path] = (primitive, tuple(translated))
        cache[name] = result
        return result

    def resolve_parameter(self, value, *, view):
        cache = self.scope.snapshot.resolved.parameters
        key = ("resolver", view)
        if key not in cache:
            cache[key] = _RegionBindingResolver(narrow=view == "narrow")
        return cache[key].visit(value, self.scope) if isinstance(value, Expr) else value

    def _affine(self, value, *, caller=False):
        cache = self.scope.snapshot.resolved.parameters
        key = ("affine", id(value), caller)
        if key not in cache:
            values = {}
            scope = self.scope.caller_scope if caller else self.scope
            term = AccessRelation.affine(value, scope, values=values)
            cache[key] = (value, term, values)
        return cache[key][1:]

    def projected_relations(self, call, ctx):
        key = (id(call), ctx.topology_level)
        if key not in self.projections:
            stated = self.relations.get(id(call))
            stated = relations_of(call, ctx) if stated is None else stated[1]
            self.projections[key] = tuple(r.normalized(ctx) for r in projected(stated, call, ctx))
        return self.projections[key]

    def project(self, relation, *, view="narrow"):
        """Bind scalar parameters before composing tensor origins at the call boundary."""
        key = ("bound", id(relation), view)
        if key in self.projections:
            return self.projections[key][1]
        result = self._project(relation, view=view)
        self.projections[key] = (relation, result)
        return result

    def _project(self, relation, *, view):
        provenance = self.scope.provenance
        caller_provenance = provenance.caller_provenance
        if caller_provenance is None:
            return relation.constrain(self.scope.domain)
        result = relation
        for name, value in tuple(relation.values.items()):
            if not isinstance(value, Expr):
                continue
            resolved = self.resolve_parameter(value, view=view)
            if resolved is value:
                continue
            term, values = self._affine(resolved, caller=True)
            named = isl.pw_aff(f"[{name}] -> {{ [({name})] }}")
            result = result.constrain(named.eq_set(term).params(), values).project((value,))
        caller = caller_provenance.scope
        result = result.constrain(caller.domain)
        return result if caller.caller_scope is None else caller.access.project(result, view=view)


class IterationScope:
    """A lexical node or a calling-environment entry over shared definition facts."""

    def __init__(self, owner, parent=None, children=(), domain=None, mesh=None, topologies=()):
        self.snapshot = ScopeSnapshot(
            DefinitionSnapshot(
                owner,
                parent,
                children,
                AccessRelation(isl.map("{ [] -> [] }")) if domain is None else domain,
                mesh,
                topologies,
            )
        )
        self._definition = self
        self._binding = self
        self.caller_scope = None
        self._arguments = {}
        self._instance = None
        self.provenance = ValueProvenance(self)
        self.access = AccessAnalysis(self)

    @classmethod
    def _entry(cls, definition, binding=None, *, caller=None, arguments=None, instance=None):
        entry = object.__new__(cls)
        entry.snapshot = ScopeSnapshot(definition.snapshot.definition)
        entry._definition = definition
        entry._binding = entry if binding is None else binding
        entry.caller_scope = caller if binding is None else binding.caller_scope
        entry._arguments = {} if arguments is None else dict(arguments)
        entry._instance = instance
        entry.provenance = ValueProvenance(entry)
        entry.access = AccessAnalysis(entry)
        return entry

    def attach(self, *, call_site, callee, arguments, instance=None):
        """Graft one call edge without visiting or copying the definition's children."""
        key = SSAValue(call_site)
        previous = self.snapshot.attachments.get(key)
        if (
            previous is None
            or previous._definition is not callee
            or previous._instance is not instance
            or previous._arguments.keys() != arguments.keys()
            or any(previous._arguments[name] is not value for name, value in arguments.items())
        ):
            self.snapshot.attachments[key] = self._entry(
                callee,
                caller=self,
                arguments=arguments,
                instance=instance,
            )
        return self.snapshot.attachments[key]

    def attachment(self, call_site):
        """Follow an existing edge in this calling environment."""
        call_site = call_site if isinstance(call_site, SSAValue) else SSAValue(call_site)
        edges = self.snapshot.attachments
        if call_site not in edges and self._definition is not self:
            template = self._definition.snapshot.attachments.get(call_site)
            if template is not None:
                instance = template.access.instance
                edges[call_site] = self._entry(
                    template._definition,
                    caller=self,
                    arguments=template.provenance.arguments,
                    instance=None if instance is None else self.access.project(instance),
                )
        return edges.get(call_site)

    def _at_definition(self, definition):
        if self.caller_scope is None:
            return definition
        root = self._binding
        if definition is root._definition:
            return root
        nodes = root.snapshot.resolved.scopes
        if definition not in nodes:
            nodes[definition] = self._entry(definition, root)
        return nodes[definition]

    def at(self, path):
        """Locate an occurrence using recorded locations and existing call edges."""
        scope = self
        for index, operation in enumerate(path):
            operation = operation if isinstance(operation, SSAValue) else SSAValue(operation)
            scope = scope._at_definition(scope.snapshot.definition.locations[id(operation.expr)])
            if index + 1 < len(path):
                scope = scope.attachment(operation)
                if scope is None:
                    raise AnalysisError("access path has no recorded call attachment")
        return scope

    def _access_environment(self, path):
        """Interpret the existing access index in its attached execution environment."""
        origin = self.at(path)
        loops = list(origin.loop_scopes)
        caller = origin.caller_scope
        while caller is not None:
            loops[:0] = caller.loop_scopes
            caller = caller.caller_scope
        return origin, tuple(loops)

    @property
    def owner(self):
        return self.snapshot.definition.owner

    @property
    def parent(self):
        parent = self.snapshot.definition.parent
        return None if parent is None else self._at_definition(parent)

    @property
    def children(self):
        return tuple(self._at_definition(child) for child in self.snapshot.definition.children)

    @children.setter
    def children(self, value):
        self.snapshot.definition.children = value

    @property
    def domain(self):
        original = self.snapshot.definition.domain
        return original if self.caller_scope is None else self.access.project(original)

    @property
    def mesh(self):
        return self.snapshot.definition.mesh

    @property
    def topologies(self):
        return self.snapshot.definition.topologies

    @property
    def root(self):
        return self if self.parent is None else self.parent.root

    def ancestors(self):
        scope = self
        while scope is not None:
            yield scope
            scope = scope.parent

    @property
    def loop_scopes(self):
        return tuple(reversed([s for s in self.ancestors() if isinstance(s.owner, LoopRegion)]))

    @property
    def captures(self):
        return self.owner.captures()

    def carries(self):
        return self.owner.carries()

    def capture_root(self, value):
        """Read capture identities without recursively resolving their relations."""
        while isinstance(value, Var):
            entry = self.provenance.snapshot.get(SSAValue(value))
            if entry is None or entry.forward is None:
                break
            value = entry.forward.expr
        return self.provenance.arguments.get(id(value), value)

    def first(self, loops=None):
        loops = self.loop_scopes if loops is None else tuple(loops)
        key = ("first", loops)
        cache = self.snapshot.resolved.relations
        if key not in cache:
            relation = iteration_domain(self.owner, self.parent, first=True, loops=loops)
            cache[key] = self.access.project(relation) if self.caller_scope else relation
        return cache[key]

    def is_variant(self, value):
        return id(self.owner) in self.provenance.varies(value)

    def mesh_positions(self, unit):
        return (
            1
            if self.mesh is None or unit is None
            else _positions_at(self.mesh, unit, self.topologies)
        )

    def active_positions(self, unit):
        """The most positions any mesh in this scope or below it runs at *unit*."""
        return max((self.mesh_positions(unit), *(c.active_positions(unit) for c in self.children)))

    def dependency_trips(self, formals):
        changing = set().union(*(self.provenance.varies(p) for p in formals))
        count, precision = 1, AnalysisPrecision.EXACT
        provenance = self.provenance
        while provenance is not None:
            for loop in provenance.scope.loop_scopes:
                if id(loop.owner) in changing:
                    changing.remove(id(loop.owner))
                    count *= loop.trips()
                    precision = precision.join(loop.trips_precision)
            provenance = provenance.caller_provenance
        return count, precision

    def repeats_of(self, call, unit, stop=None):
        varying = loops = 1
        precision = AnalysisPrecision.EXACT
        changing = self.provenance.varies(call)
        for scope in self.ancestors():
            if scope is stop:
                break
            if isinstance(scope.owner, LoopRegion):
                loops *= scope.trips()
                if id(scope.owner) in changing:
                    varying *= scope.trips()
                precision = precision.join(scope.trips_precision)
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
                refuse_runtime_bound(
                    owner, "extent" if static_dim_value(owner.extent) is None else "start"
                )
            widest = bounds[1] - 1
            if bounds[1] - bounds[0] > 1:
                precision = AnalysisPrecision.UPPER_BOUND
        if step <= 0 or widest <= 0:
            return 1, precision
        return max(1, -(-widest // step)), precision

    def trips(self) -> int:
        """Return this scope's iteration count relative to its parent."""
        cache = self.snapshot.resolved.parameters
        if "trips" not in cache:
            cache["trips"] = self._counted()
        return cache["trips"][0]

    @property
    def trips_precision(self) -> AnalysisPrecision:
        """How exactly :meth:`trips` counts this scope's iterations.

        ``EXACT`` where the span between the bounds is a number, ``UPPER_BOUND``
        where it is read at its widest because a bound names something no one
        here fixes -- a coordinate of the scope the loop is entered from.
        """
        self.trips()
        return self.snapshot.resolved.parameters["trips"][1]


class ScopeBuilder(ValuationVisitor[None]):
    """Calculate one definition's value and access facts in a single traversal."""

    def __init__(
        self, module, graph, *, views=("narrow", "device", "unit"), ctx=None, is_root=True
    ):
        super().__init__(root_function=graph)
        self.graph, self.views, self.is_root = graph, tuple(views), is_root
        self.context = ctx or AnalyzeContext(module=module, function=graph)
        self.snapshot = {}
        self.seeds = {}

    def _views(self, ctx):
        logical = ctx if ctx.topology_level is None else replace(ctx, topology_level=None)
        return {view: ctx if view == "unit" else logical for view in self.views}

    def _record(self, expr, ctx, *, forward=None, alias=None, relations=None, views=None):
        ctx.scope.snapshot.definition.locations[id(expr)] = ctx.scope
        views = self._views(ctx) if views is None else views
        scope = ctx.scope
        changing = set()
        for operand in expr_children(expr):
            changing.update(scope.provenance.varies(operand))
        if id(expr) in self.seeds:
            changing.add(id(self.seeds[id(expr)].owner))
        if isinstance(expr, Var) and any(expr is p for p in self.graph.params):
            changing.add(id(expr))
        for value in SSAValue.leaves(expr):
            source = None if forward is None else forward[value.index]
            origin = None if source is None else scope.provenance.lookup(source)
            if origin is not None:
                changing.update(origin.varies)
            view_relations = {view: None for view in ("narrow", "device", "unit")}
            root = value
            for view, local in views.items():
                if origin is not None:
                    relation, root = getattr(origin, view), origin.root
                elif alias is not None:
                    origin_alias = scope.provenance.lookup(alias.target)
                    relation = renaming_relation(expr, local, relations[view]).normalized(scope)
                    relation, _ = eliminate_parameters(
                        relation,
                        scope,
                        alias.target.leaf(),
                        narrow=view == "narrow",
                        formals=() if self.is_root else self.graph.params,
                        keep_possible=True,
                    )
                    upstream = getattr(origin_alias, view)
                    relation = None if upstream is None else relation.compose(upstream)
                    root = origin_alias.root
                else:
                    relation = identity_access(leaf_rank(value.leaf(local.local_type_of(expr))))
                view_relations[view] = (
                    None if relation is None else relation.constrain(scope.domain)
                )
            self.snapshot[value] = SnapshotEntry(
                root, **view_relations, varies=frozenset(changing), forward=source
            )

    def _record_accesses(self, call, ctx):
        scope, access = ctx.scope, ctx.scope.access
        if not SSAValue.leaves(call) or access_relation_registry.lookup(type(call.target)) is None:
            self._record(call, ctx)
            return
        views = self._views(ctx)
        logical = views["device"] if "device" in views else replace(ctx, topology_level=None)
        try:
            stated = relations_of(call, logical)
            access.relations[id(call)] = (call, stated)
            relations = {
                view: access.projected_relations(call, local) for view, local in views.items()
            }
            alias = (
                buffer_alias(call, logical, relations["narrow"])
                if aliased_operand(call) is not None
                else None
            )
            self._record(call, ctx, alias=alias, relations=relations, views=views)
            for view, local in views.items():
                for inputs, boundaries, into in (
                    (True, relations[view][: len(call.args)], access.accesses),
                    (False, relations[view][len(call.args) :], access.outputs),
                ):
                    built = tuple(
                        item
                        for index, boundary in enumerate(boundaries)
                        for item in resolve_access(
                            call.args[index] if inputs else call,
                            boundary,
                            scope,
                            local,
                            input_index=index if inputs else None,
                            output_index=None if inputs else index,
                            view=view,
                            formals=() if self.is_root else self.graph.params,
                        )
                    )
                    into.setdefault(view, {})[(SSAValue(call),)] = (call, built)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            self._record(call, ctx)
            for view in self.views:
                access.refused[view] = access.refused.get(view, frozenset()) | {call}
                access.accesses.setdefault(view, {})[(SSAValue(call),)] = (call, ())

    def on_region_start(self, region, ctx):
        inner = TypeInferContext.enter_region(ctx, region)
        child = IterationScope(
            region,
            parent=ctx.scope,
            domain=iteration_domain(region, ctx.scope),
            mesh=inner.current_mesh,
            topologies=ctx.topologies,
        )
        child.snapshot.definition.values = self.snapshot
        child.snapshot.definition.locations = ctx.scope.snapshot.definition.locations
        child.snapshot.definition.locations[id(region)] = ctx.scope
        ctx.scope.children += (child,)
        inner = replace(inner, scope=child)
        if isinstance(region, LoopRegion):
            for param in (region.induction_var, *(phi for phi, _, _ in region.carries())):
                self.seeds[id(param)] = child
                child.snapshot.definition.locations[id(param)] = child
        for param, argument in region.captures():
            self._record(param, inner, forward=SSAValue.leaves(argument))
            self._memo[id(param)] = (param, None)
        return inner

    def on_region_end(self, region, operands, ctx):
        parent = ctx.scope.parent.access
        for name in ("accesses", "outputs"):
            for view, records in getattr(ctx.scope.access, name).items():
                getattr(parent, name).setdefault(view, {}).update(records)
        for view, refused in ctx.scope.access.refused.items():
            parent.refused[view] = parent.refused.get(view, frozenset()) | refused
        parent.relations.update(ctx.scope.access.relations)
        return operands

    def visit_leaf_Tuple(self, expr, operands, ctx):
        self._record(
            expr,
            ctx,
            forward=tuple(v for element in expr.elements for v in SSAValue.leaves(element)),
        )

    def visit_leaf_MeshRegion(self, expr, operands, ctx):
        self._record(expr, ctx, forward=SSAValue.leaves(expr.body))

    def visit_leaf_LoopRegion(self, expr, operands, ctx):
        values = tuple(phi for phi, _, _ in expr.carries()) or (expr.body,)
        self._record(
            expr, ctx, forward=tuple(v for value in values for v in SSAValue.leaves(value))
        )

    def visit_TupleGetItem(self, call, ctx):
        index = static_dim_value(call.args[1])
        self._record_accesses(call, ctx)
        if index is not None:
            begin, count = leaf_span(call.args[0].type, index)
            self._record(
                call,
                ctx,
                forward=tuple(SSAValue(call.args[0], i) for i in range(begin, begin + count)),
            )

    def on_call_start(self, call, ctx):
        inner = super().on_call_start(call, ctx)
        ctx.scope.snapshot.definition.locations[id(call)] = ctx.scope
        return inner

    def visit_Call(self, call, ctx):
        self.visit_operands(call, ctx)
        callee_scope = ctx.callee_scope
        if callee_scope is None:
            handler = getattr(self, f"visit_{type(call.target).__name__}", self._record_accesses)
            handler(call, ctx)
            return
        for value in SSAValue.leaves(call):
            self.snapshot[value] = replace(
                callee_scope.provenance.result(value.index), forward=None
            )
        for name in ("accesses", "outputs"):
            for view, records in getattr(callee_scope.access, name).items():
                into = getattr(ctx.scope.access, name).setdefault(view, {})
                into.update({(SSAValue(call), *path): record for path, record in records.items()})
        for view, refused in callee_scope.access.refused.items():
            ctx.scope.access.refused[view] = (
                ctx.scope.access.refused.get(view, frozenset()) | refused
            )

    def default_visit_leaf(self, expr, operands, ctx):
        self._record(expr, ctx)

    def build(self):
        mesh = get_metadata(self.graph, ExecutionMeshMetadata)
        root = IterationScope(
            self.graph, mesh=None if mesh is None else mesh.mesh, topologies=self.context.topologies
        )
        root.snapshot.definition.values = self.snapshot
        self.context.scope = root
        ctx = replace(self.context, owner=self.graph, parent=None, current_mesh=root.mesh)
        for param in self.graph.params:
            self.visit(param, ctx)
        self.visit_function_body(self.graph, ctx)
        return root


def build_scopes(module, graph, *, views=("narrow", "device", "unit"), ctx=None):
    return ScopeBuilder(module, graph, views=views, ctx=ctx).build()
