"""Shared authored iteration scopes used by analysis families."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import isl

from tilefoundry.interfaces import access, typeinfer, valuation
from tilefoundry.ir import core, expr, types, utils

from .access import (
    RegionBindingResolver,
    SnapshotEntry,
    buffer_alias,
    eliminate_parameters,
    resolve_access,
)
from .errors import Error
from .loop_domain import induction_name, iteration_domain, refuse_runtime_bound, span_bounds
from .metadata import ExecutionMeshMetadata
from .precision import Precision
from .visitor import AnalyzeContext


def positions_at(mesh: types.Mesh, unit: str, topologies: tuple[types.Topology, ...]) -> int:
    """Count mesh positions at the selected unit and all coarser topology levels."""
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    selected = declared[unit]
    stated = mesh.layout.outer if isinstance(mesh.layout, types.ComposedLayout) else mesh.layout
    positions = 1
    for index, topology in enumerate(mesh.topologies):
        if declared[topology.name] > selected:
            continue
        count = types.size(types.get(stated, index))
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise Error(
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
    trips_precision: Precision


@dataclass
class DefinitionSnapshot:
    """Facts computed once for a lexical definition node."""

    owner: object
    parent: IterationScope | None = None
    children: tuple = ()
    domain: access.Relation = field(default_factory=lambda: access.Relation(isl.map("{ [] -> [] }")))
    mesh: types.Mesh | None = None
    topologies: tuple = ()
    values: dict = field(default_factory=dict)
    locations: dict = field(default_factory=dict)
    accesses: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)

    def add_child(self, child: IterationScope) -> None:
        """Record *child* as the last scope nested in this definition."""
        self.children += (child,)

    def locate(self, node, scope: IterationScope) -> None:
        """Record that *node* stands in *scope*."""
        self.locations[id(node)] = scope

    def share(self, values: dict, locations: dict | None = None) -> None:
        """Read value facts, and optionally locations, from the tables of an enclosing build."""
        self.values = values
        if locations is not None:
            self.locations = locations


@dataclass
class ResolvedSnapshot:
    """Derived results in one fixed lexical and calling environment."""

    values: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
    allocations: dict = field(default_factory=dict)
    scopes: dict = field(default_factory=dict)
    accesses: dict = field(default_factory=dict)

    def keep_allocation(self, node, summary) -> None:
        """Remember the allocation summary resolved at *node*."""
        self.allocations[id(node)] = summary


@dataclass
class ScopeSnapshot:
    definition: DefinitionSnapshot
    attachments: dict = field(default_factory=dict)
    resolved: ResolvedSnapshot = field(default_factory=ResolvedSnapshot)

    def attach(self, call_site, entry: IterationScope) -> None:
        """Graft *entry* as the calling environment of *call_site*."""
        self.attachments[call_site] = entry


@dataclass
class ValueProvenance:
    """One definition's value facts, interpreted through optional caller bindings."""

    scope: IterationScope

    @property
    def snapshot(self):
        return self.scope.snapshot.definition.values

    @property
    def arguments(self):
        return self.scope.binding.arguments

    @property
    def caller_provenance(self):
        caller = self.scope.caller_scope
        return None if caller is None else caller.provenance

    @property
    def resolved(self):
        return self.scope.snapshot.resolved.values

    def lookup(self, value: core.SSAValue) -> SnapshotEntry:
        if value in self.resolved:
            return self.resolved[value]
        entry = self.snapshot.get(value)
        if entry is None:
            if self.scope.parent is not None:
                return self.scope.parent.provenance.lookup(value)
            raise Error(f"no source recorded for {value.expr!r}")
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
            else caller_provenance.lookup(core.SSAValue(argument, entry.root.index))
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
            raise Error("parameter has no binding in this calling context")
        return self.caller_provenance.lookup(core.SSAValue(self.arguments[id(formal)], index))

    def result(self, index=0):
        return self.lookup(core.SSAValue(self.scope.root.owner.body, index))

    def varies(self, value):
        argument = self.arguments.get(id(value))
        if argument is not None and self.caller_provenance is not None:
            return self.caller_provenance.varies(argument)
        entry = self.snapshot.get(core.SSAValue(value))
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
class ScopeAccess:
    """Declared boundaries, resolved accesses, and projections in one environment."""

    scope: IterationScope

    @property
    def accesses(self):
        return self.bound_accesses("accesses")

    @property
    def outputs(self):
        return self.bound_accesses("outputs")

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

    def cache_projection(self, key, value) -> None:
        """Remember the projection stored under *key*."""
        self.projections[key] = value

    def record_relations(self, call, stated) -> None:
        """Remember the relations *call*'s operation states for its boundaries."""
        self.relations[id(call)] = (call, stated)

    def refuse(self, view, calls) -> None:
        """Mark *calls* as refused in *view*."""
        self.refused[view] = self.refused.get(view, frozenset()) | calls

    @property
    def instance(self):
        return self.scope.binding.instance

    def bound_accesses(self, name):
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
                                else access.identity(
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
                        relation = relation.aligned(
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
            cache[key] = RegionBindingResolver(narrow=view == "narrow")
        return cache[key].visit(value, self.scope) if isinstance(value, expr.Expr) else value

    def affine(self, value, *, caller=False):
        cache = self.scope.snapshot.resolved.parameters
        key = ("affine", id(value), caller)
        if key not in cache:
            values = {}
            scope = self.scope.caller_scope if caller else self.scope
            term = access.affine_of(value, scope, values=values)
            cache[key] = (value, term, values)
        return cache[key][1:]

    def projected_relations(self, call, ctx):
        key = (id(call), ctx.topology_level)
        if key not in self.projections:
            stated = self.relations.get(id(call))
            stated = access.relations_of(call, ctx) if stated is None else stated[1]
            self.projections[key] = tuple(r.normalized(ctx) for r in access.projected(stated, call, ctx))
        return self.projections[key]

    def project(self, relation, *, view="narrow"):
        """Bind scalar parameters before composing tensor origins at the call boundary."""
        key = ("bound", id(relation), view)
        if key in self.projections:
            return self.projections[key][1]
        result = self.project_relation(relation, view=view)
        self.projections[key] = (relation, result)
        return result

    def project_relation(self, relation, *, view):
        provenance = self.scope.provenance
        caller_provenance = provenance.caller_provenance
        if caller_provenance is None:
            return relation.constrain(self.scope.domain)
        result = relation
        for name, value in tuple(relation.values.items()):
            if not isinstance(value, expr.Expr):
                continue
            resolved = self.resolve_parameter(value, view=view)
            if resolved is value:
                continue
            term, values = self.affine(resolved, caller=True)
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
                access.Relation(isl.map("{ [] -> [] }")) if domain is None else domain,
                mesh,
                topologies,
            )
        )
        self.definition = self
        self.binding = self
        self.caller_scope = None
        self.arguments = {}
        self.instance = None
        self.provenance = ValueProvenance(self)
        self.access = ScopeAccess(self)

    @classmethod
    def from_definition(
        cls, definition, binding=None, *, caller=None, arguments=None, instance=None
    ):
        """A calling-environment entry over the shared facts of *definition*."""
        entry = object.__new__(cls)
        entry.bind(definition, binding, caller, arguments, instance)
        return entry

    def bind(self, definition, binding, caller, arguments, instance) -> None:
        """Place this scope in the calling environment given by *binding* or *caller*."""
        self.snapshot = ScopeSnapshot(definition.snapshot.definition)
        self.definition = definition
        self.binding = self if binding is None else binding
        self.caller_scope = caller if binding is None else binding.caller_scope
        self.arguments = {} if arguments is None else dict(arguments)
        self.instance = instance
        self.provenance = ValueProvenance(self)
        self.access = ScopeAccess(self)

    def attach(self, *, call_site, callee, arguments, instance=None):
        """Graft one call edge without visiting or copying the definition's children."""
        key = core.SSAValue(call_site)
        previous = self.snapshot.attachments.get(key)
        if (
            previous is None
            or previous.definition is not callee
            or previous.instance is not instance
            or previous.arguments.keys() != arguments.keys()
            or any(previous.arguments[name] is not value for name, value in arguments.items())
        ):
            self.snapshot.attach(
                key,
                self.from_definition(callee, caller=self, arguments=arguments, instance=instance),
            )
        return self.snapshot.attachments[key]

    def attachment(self, call_site):
        """Follow an existing edge in this calling environment."""
        call_site = call_site if isinstance(call_site, core.SSAValue) else core.SSAValue(call_site)
        edges = self.snapshot.attachments
        if call_site not in edges and self.definition is not self:
            template = self.definition.snapshot.attachments.get(call_site)
            if template is not None:
                instance = template.access.instance
                self.snapshot.attach(
                    call_site,
                    self.from_definition(
                        template.definition,
                        caller=self,
                        arguments=template.provenance.arguments,
                        instance=None if instance is None else self.access.project(instance),
                    ),
                )
        return edges.get(call_site)

    def at_definition(self, definition):
        if self.caller_scope is None:
            return definition
        root = self.binding
        if definition is root.definition:
            return root
        nodes = root.snapshot.resolved.scopes
        if definition not in nodes:
            nodes[definition] = self.from_definition(definition, root)
        return nodes[definition]

    def at(self, path):
        """Locate an occurrence using recorded locations and existing call edges."""
        scope = self
        for index, operation in enumerate(path):
            operation = (
                operation if isinstance(operation, core.SSAValue) else core.SSAValue(operation)
            )
            scope = scope.at_definition(scope.snapshot.definition.locations[id(operation.expr)])
            if index + 1 < len(path):
                scope = scope.attachment(operation)
                if scope is None:
                    raise Error("access path has no recorded call attachment")
        return scope

    def access_environment(self, path):
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
        return None if parent is None else self.at_definition(parent)

    @property
    def children(self):
        return tuple(self.at_definition(child) for child in self.snapshot.definition.children)

    def add_child(self, child):
        """Nest *child* last in this scope's definition."""
        self.snapshot.definition.add_child(child)

    def locate(self, node, scope=None):
        """Record that *node* stands in *scope*, or in this scope when none is given."""
        self.snapshot.definition.locate(node, self if scope is None else scope)

    def share_facts(self, values, locations=None):
        """Read value facts, and optionally locations, from the tables of an enclosing build."""
        self.snapshot.definition.share(values, locations)

    def keep_allocation(self, node, summary):
        """Remember the allocation summary resolved at *node*."""
        self.snapshot.resolved.keep_allocation(node, summary)

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
        return tuple(reversed([s for s in self.ancestors() if isinstance(s.owner, core.LoopRegion)]))

    @property
    def captures(self):
        return self.owner.captures()

    def carries(self):
        return self.owner.carries()

    def capture_root(self, value):
        """Read capture identities without recursively resolving their relations."""
        while isinstance(value, expr.Var):
            entry = self.provenance.snapshot.get(core.SSAValue(value))
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
            else positions_at(self.mesh, unit, self.topologies)
        )

    def active_positions(self, unit):
        """The most positions any mesh in this scope or below it runs at *unit*."""
        return max((self.mesh_positions(unit), *(c.active_positions(unit) for c in self.children)))

    def dependency_trips(self, formals):
        changing = set().union(*(self.provenance.varies(p) for p in formals))
        count, precision = 1, Precision.EXACT
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
        precision = Precision.EXACT
        changing = self.provenance.varies(call)
        for scope in self.ancestors():
            if scope is stop:
                break
            if isinstance(scope.owner, core.LoopRegion):
                loops *= scope.trips()
                if id(scope.owner) in changing:
                    varying *= scope.trips()
                precision = precision.join(scope.trips_precision)
        return Repeats(varying, loops, self.mesh_positions(unit), precision)

    def counted(self) -> tuple[int, Precision]:
        """This scope's iteration count relative to its parent, and how exact it is.

        A loop runs its own span, not the grid it is entered from, so the count
        is read off ``stop - start``: exactly where that span reaches one value
        -- however it is written, so bounds naming an enclosing coordinate that
        cancel are exact -- and at its widest where it reaches several. The
        widest span is an upper bound on every coordinate's count, which is
        what every consumer of a trip count already reads it as.
        """
        if isinstance(self.owner, core.MeshRegion) or self.parent is None:
            return 1, Precision.EXACT
        owner = self.owner
        step = expr.static_dim_value(owner.step)
        if step is None:
            raise Error(
                f"loop {induction_name(owner)!r} steps by {owner.step!r}, which is not "
                "a number; how often it runs is not stated"
            )
        bounds = span_bounds(owner)
        if bounds is None:
            refuse_runtime_bound(
                owner, "extent" if expr.static_dim_value(owner.extent) is None else "start"
            )
        least, widest = bounds
        precision = Precision.EXACT if widest <= least else Precision.UPPER_BOUND
        if step <= 0 or widest <= 0:
            return 1, precision
        return max(1, -(-widest // step)), precision

    def trips(self) -> int:
        """Return this scope's iteration count relative to its parent."""
        cache = self.snapshot.resolved.parameters
        if "trips" not in cache:
            cache["trips"] = self.counted()
        return cache["trips"][0]

    @property
    def trips_precision(self) -> Precision:
        """How exactly :meth:`trips` counts this scope's iterations.

        ``EXACT`` where the span between the bounds is a number, ``UPPER_BOUND``
        where it is read at its widest because a bound names something no one
        here fixes -- a coordinate of the scope the loop is entered from.
        """
        self.trips()
        return self.snapshot.resolved.parameters["trips"][1]


class ScopeBuilder(valuation.Visitor[None]):
    """Calculate one definition's value and access facts in a single traversal."""

    def __init__(
        self, module, graph, *, views=("narrow", "device", "unit"), ctx=None, is_root=True
    ):
        super().__init__(root_function=graph)
        self.graph, self.views, self.is_root = graph, tuple(views), is_root
        self.context = ctx or AnalyzeContext(module=module, function=graph)
        self.snapshot = {}
        self.seeds = {}

    def views_of(self, ctx):
        logical = ctx if ctx.topology_level is None else replace(ctx, topology_level=None)
        return {view: ctx if view == "unit" else logical for view in self.views}

    def record_node(self, node, ctx, *, forward=None, alias=None, relations=None, views=None):
        ctx.scope.locate(node)
        views = self.views_of(ctx) if views is None else views
        scope = ctx.scope
        changing = set()
        for operand in core.expr_children(node):
            changing.update(scope.provenance.varies(operand))
        if id(node) in self.seeds:
            changing.add(id(self.seeds[id(node)].owner))
        if isinstance(node, expr.Var) and any(node is p for p in self.graph.params):
            changing.add(id(node))
        for value in core.ssa_leaves(node):
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
                    relation = access.renaming_relation(node, local, relations[view]).normalized(scope)
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
                    relation = access.identity(utils.leaf_rank(value.leaf(local.local_type_of(node))))
                view_relations[view] = (
                    None if relation is None else relation.constrain(scope.domain)
                )
            self.snapshot[value] = SnapshotEntry(
                root, **view_relations, varies=frozenset(changing), forward=source
            )

    def record_accesses(self, call, ctx):
        scope, acc = ctx.scope, ctx.scope.access
        if not core.ssa_leaves(call) or not access.relation_registry.has(type(call.target)):
            self.record_node(call, ctx)
            return
        views = self.views_of(ctx)
        logical = views["device"] if "device" in views else replace(ctx, topology_level=None)
        try:
            stated = access.relations_of(call, logical)
            acc.record_relations(call, stated)
            relations = {
                view: acc.projected_relations(call, local) for view, local in views.items()
            }
            alias = (
                buffer_alias(call, logical, relations["narrow"])
                if access.aliased_operand(call) is not None
                else None
            )
            self.record_node(call, ctx, alias=alias, relations=relations, views=views)
            for view, local in views.items():
                for inputs, boundaries, into in (
                    (True, relations[view][: len(call.args)], acc.accesses),
                    (False, relations[view][len(call.args) :], acc.outputs),
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
                    into.setdefault(view, {})[(core.SSAValue(call),)] = (call, built)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            self.record_node(call, ctx)
            for view in self.views:
                acc.refuse(view, {call})
                acc.accesses.setdefault(view, {})[(core.SSAValue(call),)] = (call, ())

    def on_region_start(self, region, ctx):
        inner = typeinfer.TypeInferContext.enter_region(ctx, region)
        child = IterationScope(
            region,
            parent=ctx.scope,
            domain=iteration_domain(region, ctx.scope),
            mesh=inner.current_mesh,
            topologies=ctx.topologies,
        )
        child.share_facts(self.snapshot, ctx.scope.snapshot.definition.locations)
        ctx.scope.locate(region)
        ctx.scope.add_child(child)
        inner = replace(inner, scope=child)
        if isinstance(region, core.LoopRegion):
            for param in (region.induction_var, *(phi for phi, _, _ in region.carries())):
                self.seeds[id(param)] = child
                child.locate(param)
        for param, argument in region.captures():
            self.record_node(param, inner, forward=core.ssa_leaves(argument))
            self.memo[id(param)] = (param, None)
        return inner

    def on_region_end(self, region, operands, ctx):
        parent = ctx.scope.parent.access
        for name in ("accesses", "outputs"):
            for view, records in getattr(ctx.scope.access, name).items():
                getattr(parent, name).setdefault(view, {}).update(records)
        for view, refused in ctx.scope.access.refused.items():
            parent.refuse(view, refused)
        parent.relations.update(ctx.scope.access.relations)
        return operands

    def visit_leaf_Tuple(self, node, operands, ctx):
        self.record_node(
            node,
            ctx,
            forward=tuple(v for element in node.elements for v in core.ssa_leaves(element)),
        )

    def visit_leaf_MeshRegion(self, node, operands, ctx):
        self.record_node(node, ctx, forward=core.ssa_leaves(node.body))

    def visit_leaf_LoopRegion(self, node, operands, ctx):
        values = tuple(phi for phi, _, _ in node.carries()) or (node.body,)
        self.record_node(
            node, ctx, forward=tuple(v for value in values for v in core.ssa_leaves(value))
        )

    def visit_TupleGetItem(self, call, ctx):
        index = expr.static_dim_value(call.args[1])
        self.record_accesses(call, ctx)
        if index is not None:
            begin, count = utils.leaf_span(call.args[0].type, index)
            self.record_node(
                call,
                ctx,
                forward=tuple(core.SSAValue(call.args[0], i) for i in range(begin, begin + count)),
            )

    def on_call_start(self, call, ctx):
        inner = super().on_call_start(call, ctx)
        ctx.scope.locate(call)
        return inner

    def visit_Call(self, call, ctx):
        self.visit_operands(call, ctx)
        callee_scope = ctx.callee_scope
        if callee_scope is None:
            handler = getattr(self, f"visit_{type(call.target).__name__}", self.record_accesses)
            handler(call, ctx)
            return
        for value in core.ssa_leaves(call):
            self.snapshot[value] = replace(
                callee_scope.provenance.result(value.index), forward=None
            )
        for name in ("accesses", "outputs"):
            for view, records in getattr(callee_scope.access, name).items():
                into = getattr(ctx.scope.access, name).setdefault(view, {})
                into.update({(core.SSAValue(call), *path): record for path, record in records.items()})
        for view, refused in callee_scope.access.refused.items():
            ctx.scope.access.refuse(view, refused)

    def default_visit_leaf(self, node, operands, ctx):
        self.record_node(node, ctx)

    def build(self):
        mesh = expr.get_metadata(self.graph, ExecutionMeshMetadata)
        root = IterationScope(
            self.graph, mesh=None if mesh is None else mesh.mesh, topologies=self.context.topologies
        )
        root.share_facts(self.snapshot)
        self.context.bind_scope(root)
        ctx = replace(self.context, owner=self.graph, parent=None, current_mesh=root.mesh)
        for param in self.graph.params:
            self.visit(param, ctx)
        self.visit_function_body(self.graph, ctx)
        return root


def build_scopes(module, graph, *, views=("narrow", "device", "unit"), ctx=None):
    return ScopeBuilder(module, graph, views=views, ctx=ctx).build()


__all__ = [
    "IterationScope",
    "Repeats",
    "ScopeBuilder",
    "build_scopes",
]
