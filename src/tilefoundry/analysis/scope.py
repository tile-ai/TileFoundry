"""The shared lexical scopes and access relations used by analysis families."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import isl

from tilefoundry.ir.core import Call, Expr, binding_name
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.ir.visitor import expr_children
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    index_set,
    relation_of,
    relations_of,
    renaming_relation,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

from .errors import AnalysisError
from .footprint import _widest_allowed
from .metadata import BufferFootprint, LoopFootprintMetadata
from .poly.affine import loop_affine_term


@dataclass(frozen=True)
class Access:
    """One relation from a lexical scope to the allocation it reaches."""

    relation: isl.map
    buffer: Expr


@dataclass(eq=False)
class Scope:
    """One Function or authored loop, with all accesses below it."""

    owner: Function | GridRegionExpr
    parent: "Scope | None"
    children: tuple["Scope", ...]
    depth: int
    domain: isl.set
    accesses: dict[str, dict[Call, tuple[Access, ...]]] = field(default_factory=dict)
    refused: dict[str, frozenset[Call]] = field(default_factory=dict)

    def is_variant(self, value: Expr) -> bool:
        """Whether *value* depends on this loop's induction or carry values."""
        if not isinstance(self.owner, GridRegionExpr):
            return False
        seeds = (self.owner.induction_var, *self.owner.carried_args)
        pending = [value]
        seen: set[int] = set()
        while pending:
            expr = pending.pop()
            if any(expr is seed for seed in seeds):
                return True
            if id(expr) in seen:
                continue
            seen.add(id(expr))
            pending.extend(expr_children(expr))
        return False

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
        if isinstance(self.owner, GridRegionExpr):
            start, extent, step = self.owner.start, self.owner.extent, self.owner.step
            if all(isinstance(value, int) for value in (start, extent, step)):
                result = 1 if step <= 0 or extent <= start else -(-(extent - start) // step)
                self._trips_cache = result
                return result
        count = self.domain.count_val()
        parent_count = self.parent.domain.count_val()
        if not count.is_int() or not parent_count.is_int() or not parent_count.get_num_si():
            return 1
        result = max(1, count.get_num_si() // parent_count.get_num_si())
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
        for axis in range(self.depth):
            standing = standing.fix_si(
                isl.dim_type.SET,
                axis,
                standing.dim_min_val(axis).get_num_si(),
            )
        reached = access.relation.intersect_domain(standing).range()
        if reached.dim(isl.dim_type.PARAM):
            raise AnalysisError("scope access still has an unbound parameter")
        amount = reached.coalesce().count_val()
        if not amount.is_int():
            raise AnalysisError("scope access has no finite one-pass extent")
        result = amount.get_num_si()
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
        for values in self.accesses.get(view, {}).values():
            yield from values
        for child in self.children:
            yield from child.reaching(view)

    def known(self, view: str) -> bool:
        """Whether this scope and every descendant answered every access."""
        if self.refused.get(view):
            return False
        return all(child.known(view) for child in self.children)

    def footprint(self) -> LoopFootprintMetadata:
        """Summarize device and per-unit access bytes for this scope."""
        rows: dict[tuple[int, str], tuple[Expr, int, int]] = {}
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
                current = rows.get(key, (access.buffer, 0, 0))
                rows[key] = (
                    current[0],
                    current[1] + (amount * size if scale == "bytes" else 0),
                    current[2] + (device_amount * size if scale == "device_bytes" else 0),
                )
        footprints = tuple(
            BufferFootprint(
                buffer=binding_name(values[0]) or f"<buffer {buffer_id}>",
                level=level,
                bytes=values[1],
                device_bytes=values[2],
                repeated_bytes=values[1] * self.trips(),
            )
            for (buffer_id, level), values in sorted(rows.items())
        )
        return LoopFootprintMetadata(
            footprints=footprints,
            known=self.known("narrow") and self.known("device"),
        )


def _domain_for(owner: Function | GridRegionExpr, parent: Scope | None) -> isl.set:
    if isinstance(owner, Function):
        return isl.set("{ [] }")
    loops: list[GridRegionExpr] = []
    cursor = parent
    while cursor is not None:
        if isinstance(cursor.owner, GridRegionExpr):
            loops.append(cursor.owner)
        cursor = cursor.parent
    loops.reverse()
    params: dict[str, DimVar] = {}
    bounds: list[str] = []
    for index, loop in enumerate(loops + [owner]):
        start = static_dim_value(loop.start)
        step = static_dim_value(loop.step)
        if start is None or step is None:
            raise AnalysisError(f"scope {loop!r}: loop bounds must be static")
        extent = loop.extent
        if isinstance(extent, DimVar):
            params[extent.name] = extent
            stop = extent.name
        else:
            value = static_dim_value(extent)
            if value is None:
                raise AnalysisError(f"scope {loop!r}: loop extent is not bounded")
            stop = str(value)
        bounds.append(f"{start} <= p{index} < {stop}")
        if step != 1:
            bounds.append(f"(p{index} - {start}) mod {step} = 0")
    for name, dim in params.items():
        bounds.append(f"{dim.lo} <= {name} < {dim.hi}")
    names = ", ".join(f"p{index}" for index in range(len(loops) + 1))
    prefix = f"[{', '.join(params)}] -> " if params else ""
    return isl.set(f"{prefix}{{ [{names}] : {' and '.join(bounds)} }}")


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
    loops = []
    cursor = scope
    while cursor is not None:
        if isinstance(cursor.owner, GridRegionExpr):
            loops.append(cursor.owner)
        cursor = cursor.parent
    loops.reverse()
    relation = relation.insert_dims(isl.dim_type.IN, 0, len(loops))
    scope_domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
    )
    relation = relation.intersect_domain(scope_domain)
    params = dict(getattr(boundary.pattern, "parameters", ()) or ())
    for name in list(params):
        value = params[name]
        number = static_dim_value(value)
        if number is None:
            term = None
            try:
                term = loop_affine_term(value, tuple(loops), narrow=narrow)
            except (TypeError, ValueError, NotImplementedError):
                term = None
            if term is None:
                term = _widest_allowed(relation, name, operand.type)
            if term is None:
                relation = relation.project_out(isl.dim_type.PARAM, 0, 1)
                continue
        else:
            term = type("Term", (), {"loop_axis": None, "stride": 0, "low": number, "high": number})()
        local = isl.local_space.from_space(relation.get_space())

        def placed(kind: str, sign: int, constant: int) -> isl.constraint:
            constraint = getattr(isl.constraint, f"alloc_{kind}")(local)
            constraint = constraint.set_coefficient_si(isl.dim_type.PARAM, 0, sign)
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
        relation = relation.project_out(isl.dim_type.PARAM, 0, 1)
    try:
        held = local_type_of(operand.type) if narrow else operand.type
    except (TypeError, ValueError, NotImplementedError):
        return None
    box = index_set(tuple(held.shape)) if isinstance(held, TensorType) else None
    if box is not None:
        relation = relation.intersect_range(box)
    while isinstance(operand, Call) and isinstance(operand.target, (Slice, Reshape)):
        folded = renaming_relation(operand, ctx)
        relation = relation.apply_range(relation_of(folded))
        operand = operand.args[0]
    return Access(relation, operand)


def build_scopes(
    module: Module,
    graph: Function,
    *,
    views: Sequence[str] = ("narrow", "device"),
) -> Scope:
    """Build the scope tree and both access views in one normalized walk."""
    class IdentityMap:
        """Small identity-keyed mapping for recursive IR expressions."""

        def __init__(self) -> None:
            self._entries: list[tuple[Call, tuple[Access, ...]]] = []

        def __setitem__(self, key: Call, value: tuple[Access, ...]) -> None:
            for index, (existing, _value) in enumerate(self._entries):
                if existing is key:
                    self._entries[index] = (key, value)
                    return
            self._entries.append((key, value))

        def __iter__(self):
            return (key for key, _value in self._entries)

        def __len__(self) -> int:
            return len(self._entries)

        def get(self, key: Call, default=None):
            for existing, value in self._entries:
                if existing is key:
                    return value
            return default

        def values(self):
            return (value for _key, value in self._entries)

        def items(self):
            return tuple(self._entries)

    def empty_accesses() -> dict[str, IdentityMap]:
        return {view: IdentityMap() for view in views}

    by_owner: dict[int, Scope] = {}
    calls: list[tuple[Call, Scope]] = []
    seen: set[int] = set()

    def visit(expr: Expr, scope: Scope) -> None:
        if id(expr) in seen:
            return
        seen.add(id(expr))
        if isinstance(expr, GridRegionExpr):
            for operand in expr.init_args:
                visit(operand, scope)
            child = Scope(expr, scope, (), scope.depth + 1, _domain_for(expr, scope), empty_accesses())
            by_owner[id(expr)] = child
            scope.children = (*scope.children, child)
            visit(expr.body, child)
            for operand in expr.yield_values:
                visit(operand, child)
            return
        if isinstance(expr, Call):
            calls.append((expr, scope))
        for operand in expr_children(expr):
            visit(operand, scope)

    root = Scope(graph, None, (), 0, _domain_for(graph, None), empty_accesses())
    by_owner[id(graph)] = root
    for param in graph.params:
        visit(param, root)
    if graph.body is not None:
        visit(graph.body, root)
    type_ctx = TypeInferContext(scope=FunctionScope(module, graph))
    for expr, owner in calls:
        if isinstance(expr.target, Function) or access_relation_registry.lookup(type(expr.target)) is None:
            continue
        try:
            relations = relations_of(expr, type_ctx)
        except (NotImplementedError, TypeError, ValueError, isl.Error):
            for view in views:
                owner.refused[view] = owner.refused.get(view, frozenset()) | {expr}
            continue
        for view in views:
            narrow = view == "narrow"
            built: list[Access] = []
            for index, boundary in enumerate(relations.inputs):
                if index >= len(expr.args):
                    continue
                access = _bind_access(expr, expr.args[index], boundary, owner, type_ctx, narrow=narrow)
                if access is not None:
                    built.append(access)
            owner.accesses.setdefault(view, IdentityMap())[expr] = tuple(built)
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


__all__ = ["Access", "Scope", "ScopeBuilder", "build_scopes", "walk_scopes"]
