"""The shared lexical scopes and access relations used by analysis families."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

import isl

from tilefoundry.ir.core import Call, Expr, value_label, value_labels
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim_isl import range_expr
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.ir.visitor import expr_children
from tilefoundry.utils.isl_utils import (
    PARAM_POINT_LIMIT,
    ParameterBoxTooLarge,
    UnboundedParameterBox,
    count,
    has_unbounded_param,
    param_points,
)
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
    index_set,
    projected,
    relation_of,
    relations_of,
    renaming_relation,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

from .affine import loop_affine_term
from .errors import AnalysisError
from .footprint import _widest_allowed
from .metadata import BufferFootprint, LoopFootprintMetadata


class AccessPrecision(Enum):
    """How faithfully an access relation describes the authored access."""

    EXACT = auto()
    WIDENED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Access:
    """One relation from a lexical scope to the allocation it reaches."""

    relation: isl.map
    buffer: Expr
    precision: AccessPrecision = AccessPrecision.EXACT


@dataclass(eq=False)
class Scope:
    """One Function or authored loop, with all accesses below it."""

    owner: Function | LoopRegion
    parent: "Scope | None"
    children: tuple["Scope", ...]
    depth: int
    domain: isl.set
    accesses: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    outputs: dict[str, dict[int, tuple[Call, tuple[Access, ...]]]] = field(default_factory=dict)
    relations: dict[int, tuple[Call, AccessRelations]] = field(default_factory=dict)
    refused: dict[str, frozenset[Call]] = field(default_factory=dict)
    _variance: dict[int, frozenset[int]] = field(default_factory=dict, repr=False)

    def stated_relations(self, call: Call, ctx: TypeInferContext) -> AccessRelations:
        """Return the Op-declared relations recorded in this scope chain."""
        cursor: Scope | None = self
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

    def is_invariant(self, value: Expr) -> bool:
        """Whether *value* is independent of this loop's induction values."""
        return not self.is_variant(value)

    def trips(self) -> int:
        """Return this scope's iteration count relative to its parent."""
        cached = getattr(self, "_trips_cache", None)
        if cached is not None:
            return cached
        if self.parent is None:
            return 1
        if isinstance(self.owner, LoopRegion):
            start, extent, step = self.owner.start, self.owner.extent, self.owner.step
            if all(isinstance(value, int) for value in (start, extent, step)):
                result = 1 if step <= 0 or extent <= start else -(-(extent - start) // step)
                self._trips_cache = result
                return result
        domain = self.domain
        parent = self.parent.domain.align_params(domain.get_space())
        domain = domain.align_params(parent.get_space())
        try:
            points = param_points(domain.params().intersect(parent.params()))
        except UnboundedParameterBox as error:
            raise AnalysisError(
                f"loop {_induction_of(self.owner)!r} has unbounded parameter "
                f"{error.parameter!r}, so its trip count cannot be determined"
            ) from error
        except ParameterBoxTooLarge as error:
            raise AnalysisError(
                f"loop {_induction_of(self.owner)!r} has a parameter box exceeding "
                f"the {PARAM_POINT_LIMIT}-point analysis limit, so its trip count cannot be "
                "determined"
            ) from error
        ratios = []
        for point in points:
            amount = count(domain.intersect_params(point))
            parent_count = count(parent.intersect_params(point))
            if amount is None or not parent_count:
                continue
            ratios.append(max(1, amount // parent_count))
        result = max(ratios, default=1)
        self._trips_cache = result
        return result

    def one_pass(self, access: Access) -> int:
        """Count one pass of this scope's relation with loop axes held still."""
        cache = getattr(self, "_one_pass_cache", {})
        cached = cache.get(id(access))
        if cached is not None:
            return cached
        standing = self.domain.insert_dims(
            isl.dim_type.SET,
            self.depth,
            access.relation.dim(isl.dim_type.IN) - self.depth,
        )
        relation = access.relation.intersect_domain(standing)
        try:
            points = param_points(relation.params())
        except UnboundedParameterBox as error:
            label = value_label(access.buffer) or type(access.buffer).__name__
            raise AnalysisError(
                f"scope access to {label!r} still has unbound parameter "
                f"{error.parameter!r}"
            ) from error
        except ParameterBoxTooLarge as error:
            raise AnalysisError(
                "scope access parameter box exceeds the "
                f"{PARAM_POINT_LIMIT}-point analysis limit"
            ) from error
        amounts = []
        for point in points:
            fixed = relation.intersect_params(point)
            fixed_standing = standing.intersect_params(point)
            for axis in range(self.depth):
                low = fixed_standing.dim_min_val(axis)
                if not low.is_int():
                    raise AnalysisError("scope access has no finite one-pass extent")
                fixed_standing = fixed_standing.fix_si(
                    isl.dim_type.SET,
                    axis,
                    low.get_num_si(),
                )
            amount = count(fixed.intersect_domain(fixed_standing).range())
            if amount is None:
                raise AnalysisError("scope access has no finite one-pass extent")
            amounts.append(amount)
        result = max(amounts, default=0)
        cache[id(access)] = result
        self._one_pass_cache = cache
        return result

    def over(self, access: Access) -> isl.set:
        """Return the source elements reached while this scope varies."""
        cache = getattr(self, "_over_cache", {})
        cached = cache.get(id(access))
        if cached is not None:
            return cached
        domain = self.domain.insert_dims(
            isl.dim_type.SET,
            self.depth,
            access.relation.dim(isl.dim_type.IN) - self.depth,
        )
        for axis in range(self.depth):
            domain = domain.fix_si(
                isl.dim_type.SET,
                axis,
                domain.dim_min_val(axis).get_num_si(),
            )
        result = access.relation.intersect_domain(domain).range()
        cache[id(access)] = result
        self._over_cache = cache
        return result

    def reaching(self, view: str) -> Iterator[Access]:
        """Yield accesses owned by this scope and all descendant scopes."""
        for _call, values in self.accesses.get(view, {}).values():
            yield from values
        for child in self.children:
            yield from child.reaching(view)

    def known(self, view: str) -> bool:
        """Whether this scope and every descendant answered every access."""
        if self.refused.get(view):
            return False
        return all(child.known(view) for child in self.children)

    def footprint(self) -> LoopFootprintMetadata:
        """Summarize device and per-unit access bytes for this scope.

        Two structurally equal buffers are distinct allocations, so identity
        groups the rows. It does not order or name them: an address is whatever
        the allocator handed out this run, and a report exists to be compared
        against another run.
        """
        rows: dict[tuple[int, str], tuple[Expr, int, int, int]] = {}
        for view, scale in (("narrow", "bytes"), ("device", "device_bytes")):
            for access in self.reaching(view):
                try:
                    amount = self.one_pass(access)
                except AnalysisError:
                    continue
                size = static_bytes(access.buffer.type)
                if size is None:
                    continue
                device_amount = amount * max(1, self.trips())
                key = (id(access.buffer), str(getattr(access.buffer.type, "storage", "unknown")))
                current = rows.get(key, (access.buffer, len(rows), 0, 0))
                rows[key] = (
                    current[0],
                    current[1],
                    current[2] + (amount * size if scale == "bytes" else 0),
                    current[3] + (device_amount * size if scale == "device_bytes" else 0),
                )
        entries = list(rows.items())
        labels = value_labels(buffer for _, (buffer, _, _, _) in entries)
        ordered = sorted(
            (label, memory_level, local, device)
            for label, ((_, memory_level), (_, _, local, device)) in zip(labels, entries)
        )
        footprints = tuple(
            BufferFootprint(
                buffer=label,
                memory_level=memory_level,
                bytes=local,
                device_bytes=device,
                repeated_bytes=local * self.trips(),
            )
            for label, memory_level, local, device in ordered
        )
        return LoopFootprintMetadata(
            footprints=footprints,
            known=self.known("narrow") and self.known("device"),
        )


def _induction_of(loop: LoopRegion) -> str:
    """How a diagnostic names one loop: by the variable the author bound it to."""
    return getattr(loop.induction_var, "name", None) or "<unnamed>"


def _reject_unbounded_bound(loop: LoopRegion, which: str) -> None:
    value = getattr(loop, which)
    if which == "extent":
        raise AnalysisError(
            f"loop {_induction_of(loop)!r} has a trip count the program computes "
            f"at run time from {value_label(value) or 'a value'!r}, so no "
            f"per-occurrence total can be scaled by it; bind the extent to a "
            f"literal, or state it as an open dimension"
        )
    raise AnalysisError(
        f"loop {_induction_of(loop)!r} takes its {which} from "
        f"{value_label(value) or 'a value'!r}, which the program computes at run time; "
        f"analysis needs a literal {which} or a stated value range"
    )


def _render_bound(
    loop: LoopRegion,
    which: str,
    params: dict[str, tuple[int, int] | None],
    param_map: dict[str, object],
    identities: dict[int, str],
) -> str:
    """Render one start/extent from bounded leaves, or reject an unknown value."""
    value = getattr(loop, which)
    number = static_dim_value(value)
    if number is not None:
        return str(number)
    try:
        rendered = range_expr(
            value,
            params,
            param_map=param_map,
            identities=identities,
        )
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        _reject_unbounded_bound(loop, which)
    if any(bound is None for bound in params.values()):
        _reject_unbounded_bound(loop, which)
    return rendered


def _domain_for(owner: Function | LoopRegion, parent: Scope | None) -> isl.set:
    if isinstance(owner, Function):
        return isl.set("{ [] }")
    loops: list[LoopRegion] = []
    cursor = parent
    while cursor is not None:
        if isinstance(cursor.owner, LoopRegion):
            loops.append(cursor.owner)
        cursor = cursor.parent
    loops.reverse()
    params: dict[str, tuple[int, int] | None] = {}
    param_map: dict[str, object] = {}
    identities: dict[int, str] = {}
    bounds: list[str] = []
    for index, loop in enumerate(loops + [owner]):
        start = _render_bound(loop, "start", params, param_map, identities)
        stop = _render_bound(loop, "extent", params, param_map, identities)
        step = static_dim_value(loop.step)
        if step is None:
            raise AnalysisError(
                f"loop {_induction_of(loop)!r} takes its step from "
                f"{value_label(loop.step) or 'a value'!r}; analysis needs a literal "
                "step, because a parametric stride has no isl representation"
            )
        bounds.append(f"{start} <= p{index} < {stop}")
        if step != 1:
            bounds.append(f"(p{index} - {start}) mod {step} = 0")
    for name, bound in params.items():
        if bound is None:
            raise AnalysisError(
                f"loop domain parameter {name!r} has no stated value range"
            )
        bounds.append(f"{bound[0]} <= {name} < {bound[1]}")
    names = ", ".join(f"p{index}" for index in range(len(loops) + 1))
    prefix = f"[{', '.join(params)}] -> " if params else ""
    return isl.set(f"{prefix}{{ [{names}] : {' and '.join(bounds)} }}")


def _bind_parameters(
    relation: isl.map,
    parameters: Mapping[str, object] | Sequence[tuple[str, object]],
    loops: tuple[LoopRegion, ...],
    held: object,
    *,
    narrow: bool,
) -> tuple[isl.map, AccessPrecision]:
    """Bind one relation's stated parameters to literals or loop terms."""
    precision = AccessPrecision.EXACT
    for name, value in dict(parameters).items():
        param_index = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        if param_index < 0:
            raise AnalysisError(
                f"access pattern parameter {name!r} is missing from its relation"
            )
        number = static_dim_value(value)
        if number is None:
            term = None
            try:
                term = loop_affine_term(value, loops, narrow=narrow)
            except (TypeError, ValueError, NotImplementedError):
                term = None
            if term is None:
                term = _widest_allowed(relation, name, held)
                precision = AccessPrecision.WIDENED
            if term is None:
                relation = relation.project_out(isl.dim_type.PARAM, param_index, 1)
                continue
        else:
            term = type(
                "Term", (), {"loop_axis": None, "stride": 0, "low": number, "high": number}
            )()
        local = isl.local_space.from_space(relation.get_space())

        def placed(kind: str, sign: int, constant: int) -> isl.constraint:
            constraint = getattr(isl.constraint, f"alloc_{kind}")(local)
            constraint = constraint.set_coefficient_si(
                isl.dim_type.PARAM, param_index, sign
            )
            if term.loop_axis is not None:
                constraint = constraint.set_coefficient_si(
                    isl.dim_type.IN, term.loop_axis, -sign * term.stride
                )
            return constraint.set_constant_si(constant)

        if term.low == term.high:
            relation = relation.add_constraint(placed("equality", 1, -term.low))
        else:
            relation = relation.add_constraint(placed("inequality", 1, -term.low))
            relation = relation.add_constraint(placed("inequality", -1, term.high))
        relation = relation.project_out(isl.dim_type.PARAM, param_index, 1)
    return relation, precision


def _bind_access(
    call: Call,
    operand: Expr,
    boundary,
    scope: Scope,
    ctx: TypeInferContext,
    *,
    narrow: bool,
) -> Access | None:
    relation = relation_of(boundary.pattern)
    precision = AccessPrecision.EXACT
    loops = []
    cursor = scope
    while cursor is not None:
        if isinstance(cursor.owner, LoopRegion):
            loops.append(cursor.owner)
        cursor = cursor.parent
    loops.reverse()
    relation = relation.insert_dims(isl.dim_type.IN, 0, len(loops))
    scope_domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
    )
    relation = relation.intersect_domain(scope_domain)
    relation, precision = _bind_parameters(
        relation,
        getattr(boundary.pattern, "parameters", ()) or (),
        tuple(loops),
        operand.type,
        narrow=narrow,
    )
    try:
        held = local_type_of(operand.type) if narrow else operand.type
    except (TypeError, ValueError, NotImplementedError):
        return None
    box = index_set(tuple(held.shape)) if isinstance(held, TensorType) else None
    if box is not None:
        relation = relation.intersect_range(box)
    while isinstance(operand, Call) and isinstance(operand.target, (Slice, Reshape)):
        folded = renaming_relation(operand, ctx, stated=scope.stated_relations(operand, ctx))
        relation = relation.apply_range(relation_of(folded))
        operand = operand.args[0]
        relation, folded_precision = _bind_parameters(
            relation,
            folded.parameters,
            tuple(loops),
            operand.type,
            narrow=narrow,
        )
        if folded_precision is AccessPrecision.WIDENED:
            precision = AccessPrecision.WIDENED
    if precision is AccessPrecision.EXACT and has_unbounded_param(relation):
        precision = AccessPrecision.UNKNOWN
    return Access(relation, operand, precision)


def build_scopes(
    module: Module,
    graph: Function,
    *,
    views: Sequence[str] = ("narrow", "device"),
) -> Scope:
    """Build the scope tree and both access views in one normalized walk."""

    def empty_accesses() -> dict[str, dict[int, tuple[Call, tuple[Access, ...]]]]:
        return {view: {} for view in views}

    type_ctx = TypeInferContext(scope=FunctionScope(module, graph))
    seeds: dict[int, Scope] = {}
    variance: dict[int, frozenset[int]] = {}
    seen: set[int] = set()

    def record_accesses(expr: Call, scope: Scope) -> None:
        if (
            isinstance(expr.target, Function)
            or access_relation_registry.lookup(type(expr.target)) is None
        ):
            return
        try:
            stated = relations_of(expr, type_ctx)
            scope.relations[id(expr)] = (expr, stated)
            local_relations = projected(stated, expr, type_ctx)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            for view in views:
                scope.refused[view] = scope.refused.get(view, frozenset()) | {expr}
            return
        for view in views:
            narrow = view == "narrow"
            built: list[Access] = []
            for index, boundary in enumerate(local_relations.inputs):
                if index >= len(expr.args):
                    continue
                access = _bind_access(
                    expr, expr.args[index], boundary, scope, type_ctx, narrow=narrow
                )
                if access is not None:
                    built.append(access)
            scope.accesses.setdefault(view, {})[id(expr)] = (expr, tuple(built))
            written: list[Access] = []
            for boundary in local_relations.outputs:
                access = _bind_access(expr, expr, boundary, scope, type_ctx, narrow=narrow)
                if access is not None:
                    written.append(access)
            scope.outputs.setdefault(view, {})[id(expr)] = (expr, tuple(written))

    def record_variance(expr: Expr, operands: tuple[Expr, ...]) -> None:
        changing: set[int] = set()
        for operand in operands:
            changing.update(variance.get(id(operand), frozenset()))
        if (loop := seeds.get(id(expr))) is not None:
            changing.add(id(loop))
        variance[id(expr)] = frozenset(changing)

    def visit(expr: Expr, scope: Scope) -> None:
        if id(expr) in seen:
            return
        seen.add(id(expr))
        if isinstance(expr, LoopRegion):
            for operand in expr.init_args:
                visit(operand, scope)
            child = Scope(
                expr, scope, (), scope.depth + 1, _domain_for(expr, scope), empty_accesses()
            )
            scope.children = (*scope.children, child)
            seeds[id(expr.induction_var)] = child
            for carried in expr.carried_args:
                seeds[id(carried)] = child
            visit(expr.body, child)
            for operand in expr.yield_values:
                visit(operand, child)
            record_variance(expr, expr_children(expr))
            return
        operands = expr_children(expr)
        for operand in operands:
            visit(operand, scope)
        if isinstance(expr, Call):
            record_accesses(expr, scope)
        record_variance(expr, operands)

    root = Scope(graph, None, (), 0, _domain_for(graph, None), empty_accesses())
    for param in graph.params:
        visit(param, root)
    if graph.body is not None:
        visit(graph.body, root)
    root._variance = variance
    return root


class ScopeBuilder:
    """Build one lexical Scope tree and its access views for a derived Function."""

    def __init__(self, module: Module, graph: Function) -> None:
        self.module = module
        self.graph = graph

    def build(self) -> Scope:
        return build_scopes(self.module, self.graph)


def walk_scopes(root: Scope) -> Iterator[Scope]:
    """Yield a scope and its descendants in lexical order."""
    yield root
    for child in root.children:
        yield from walk_scopes(child)


__all__ = [
    "Access",
    "AccessPrecision",
    "Scope",
    "ScopeBuilder",
    "build_scopes",
    "walk_scopes",
]
