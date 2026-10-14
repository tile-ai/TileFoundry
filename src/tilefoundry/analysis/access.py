"""Access relations resolved against authored iteration scopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import isl

from tilefoundry.ir.core import Call, Constant, Expr, Var
from tilefoundry.ir.hir.schedule import ScheduleOp, instruction_schema
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.isl_interop import IslParamValues, dim_to_isl_pw_aff, shape_to_isl_set
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.utils import is_literal_shape, local_type_of, static_dim_value
from tilefoundry.ir.visitor import ExprCloner
from tilefoundry.utils.isl_utils import cardinality, has_unbounded_param
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    renaming_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .errors import AnalysisError
from .precision import AnalysisPrecision


@dataclass(frozen=True)
class Access:
    """One relation from an iteration scope to the allocation it reaches."""

    input_index: int | None
    relation: isl.map
    buffer: Expr
    precision: AnalysisPrecision = AnalysisPrecision.EXACT
    output_index: int | None = None
    index: int = 0
    may_access: bool = False
    occurrence: object | None = None


@dataclass(frozen=True)
class BufferAlias:
    """The result is buffer's bytes, at the coordinates relation names."""

    buffer: Expr
    relation: AccessRelation
    index: int = 0
    occurrence: object | None = None


def aligned_source(
    source: BufferAlias, rank: int, depth: int,
    parameters: Mapping[str, object] | None = None
) -> isl.map:
    """Align a cached source to the caller's enclosing-loop prefix."""
    relation = source.relation.relation
    canonical = {id(value): name for name, value in (parameters or {}).items()}
    for name, value in source.relation.values.items():
        target = canonical.get(id(value), name)
        if target == name:
            continue
        position = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        existing = relation.find_dim_by_name(isl.dim_type.PARAM, target)
        if existing < 0:
            relation = relation.set_dim_name(isl.dim_type.PARAM, position, target)
        else:
            relation = relation.equate(isl.dim_type.PARAM, position, isl.dim_type.PARAM, existing)
            relation = relation.project_out(isl.dim_type.PARAM, position, 1)
    source_depth = relation.dim(isl.dim_type.IN) - rank
    if source_depth > depth:
        return relation.project_out(isl.dim_type.IN, depth, source_depth - depth)
    if source_depth < depth:
        return relation.insert_dims(isl.dim_type.IN, source_depth, depth - source_depth)
    return relation


def apply_source(
    relation: isl.map, source: BufferAlias, rank: int, depth: int,
    parameters: Mapping[str, object] | None = None,
) -> isl.map:
    """Compose once, carrying the call's loop coordinates into the source domain."""
    relation = relation.insert_dims(isl.dim_type.OUT, 0, depth)
    for axis in range(depth):
        relation = relation.equate(isl.dim_type.IN, axis, isl.dim_type.OUT, axis)
    return relation.apply_range(aligned_source(source, rank, depth, parameters))


def buffer_alias(
    call: Call, ctx: TypeInferContext, local_relations: tuple[AccessRelation, ...]
) -> BufferAlias | None:
    """A registered view's bytes and coordinates, proving the declaration on every ask."""
    position = aliased_operand(call)
    if position is None:
        return None
    operand = call.args[position]
    relation = renaming_relation(call, ctx, local_relations)
    shape = tuple(operand.type.shape)
    box = shape_to_isl_set(shape, {}) if is_literal_shape(shape) else None
    if (
        box is None
        or not relation.relation.is_single_valued()
        or not relation.relation.is_injective()
        or not relation.relation.range().is_subset(box)
    ):
        params = (
            instruction_schema(call.target.op, len(call.args))[0]
            if isinstance(call.target, ScheduleOp)
            else tuple(
                param for param in type(call.target)._op_schema.signature if param.kind == "input"
            )
        )
        param = params[position]
        raise AnalysisError(
            f"{type(call.target).__name__} declares its result is {param.name}'s bytes, "
            f"but its access relation {relation.relation} is not single-valued, injective, "
            "and within the operand"
        )
    return BufferAlias(operand, relation)


def widest_allowed(access: isl.map, name: str, held: object) -> int | None:
    """The value a parameter may take that reaches the most of its operand.

    A footprint is an upper bound, so a parameter nobody here can place takes
    whichever end of its legal range touches more: where a window sits does not
    change how much of it there is, but how long it is does. Both ends are the
    Op's own contract, read off the relation rather than guessed.
    """
    names = [
        access.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(access.dim(isl.dim_type.PARAM))
    ]
    space = f"[{', '.join(names)}] -> "
    probe = isl.set(f"{space}{{ [x] : x = {name} }}").intersect_params(access.params())
    ends = (probe.dim_min_val(0), probe.dim_max_val(0))
    if not all(end.is_int() for end in ends):
        return None
    box = (
        shape_to_isl_set(tuple(held.shape), {})
        if isinstance(held, TensorType) and is_literal_shape(held.shape)
        else None
    )
    if box is None or box.dim(isl.dim_type.SET) != access.dim(isl.dim_type.OUT):
        return ends[0].get_num_si()
    best: tuple[int, int] | None = None
    for value in sorted({end.get_num_si() for end in ends}):
        reach = (
            access.intersect_params(isl.set(f"{space}{{ : {name} = {value} }}"))
            .range()
            .intersect(box)
        )
        amount = cardinality(reach)
        if amount is None:
            return None
        if best is None or amount > best[0]:
            best = (amount, value)
    return None if best is None else best[1]


class _RegionBindingResolver(ExprCloner):
    """Resolve region captures and local coordinates before affine rendering."""

    def __init__(self, *, narrow: bool) -> None:
        super().__init__()
        self.narrow = narrow
        self._scope_memos: dict[IterationScope, dict[int, tuple[Expr, Expr]]] = {}

    def dispatch_visit(self, value: Expr, scope: "IterationScope") -> Expr:
        memo = self._memo
        self._memo = self._scope_memos.setdefault(scope, {})
        try:
            return super().dispatch_visit(value, scope)
        finally:
            self._memo = memo

    def visit_Call(self, value: Call, scope: "IterationScope") -> Expr:
        if isinstance(value.target, Local):
            return (
                Constant(type=value.type, value=0)
                if self.narrow
                else self.visit(value.args[0], scope)
            )
        return self.default_visit(value, scope)

    def visit_Var(self, value: Var, scope: "IterationScope") -> Expr:
        cursor = scope
        while cursor is not None:
            for param, argument in cursor.captures:
                if value is param:
                    return self.visit(argument, cursor.parent)
            cursor = cursor.parent
        return value


def _parameter_term(
    value: object,
    relation: isl.map,
    name: str,
    scope: "IterationScope",
    held: object,
    *,
    narrow: bool,
    retained: dict[str, object] | None = None,
) -> tuple[isl.pw_aff | None, AnalysisPrecision]:
    """What one parameter equals, over the enclosing induction variables.

    The scope's own parameters keep their names and the ranges its domain gives
    them, and a capture of one of them is that same parameter rather than a
    second one. A value nothing here can bound is widened to the end of its
    legal range that reaches the most.
    """
    number = static_dim_value(value)
    if number is not None:
        return isl.pw_aff(f"{{ [{number}] }}"), AnalysisPrecision.EXACT
    try:
        resolved = _RegionBindingResolver(narrow=narrow).visit(value, scope)
        coords = {
            id(loop.induction_var): f"__tf_in_{i}" for i, loop in enumerate(scope.enclosing_loops())
        }
        values: IslParamValues = {
            param: scope.capture_root(parameter) for param, parameter in scope.domain_params.items()
        }
        term = dim_to_isl_pw_aff(resolved, values, coords=coords).intersect_params(
            scope.domain.params()
        )
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        pass
    else:
        if not has_unbounded_param(term):
            if retained is not None:
                retained.update(values)
            return term, AnalysisPrecision.EXACT
    number = widest_allowed(relation, name, held)
    term = None if number is None else isl.pw_aff(f"{{ [{number}] }}")
    return term, AnalysisPrecision.UPPER_BOUND


def eliminate_parameters(
    relation: isl.map,
    parameters: Mapping[str, object],
    scope: "IterationScope",
    held: object,
    *,
    narrow: bool,
    retained: dict[str, object] | None = None,
) -> tuple[isl.map, AnalysisPrecision]:
    """Eliminate parameters using scoped affine expressions or widening."""
    precision = AnalysisPrecision.EXACT
    rank = relation.dim(isl.dim_type.IN)
    for name, value in parameters.items():
        if relation.find_dim_by_name(isl.dim_type.PARAM, name) < 0:
            raise AnalysisError(f"access pattern parameter {name!r} is missing from its relation")
        term, resolved_precision = _parameter_term(
            value, relation, name, scope, held, narrow=narrow, retained=retained
        )
        if term is not None:
            term = term.add_dims(isl.dim_type.IN, rank - term.dim(isl.dim_type.IN))
            inputs = ", ".join(f"i{index}" for index in range(rank))
            named = isl.pw_aff(f"[{name}] -> {{ [{inputs}] -> [{name}] }}")
            relation = relation.intersect_domain(named.eq_set(term))
        precision = precision.join(resolved_precision)
        param_index = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        relation = relation.project_out(isl.dim_type.PARAM, param_index, 1)
    return relation, precision


def resolve_access(
    operand: Expr,
    boundary: AccessRelation,
    scope: "IterationScope",
    ctx: TypeInferContext,
    *,
    input_index: int | None,
    output_index: int | None = None,
    narrow: bool,
    sources: Mapping[int, BufferAlias | tuple],
    precisions: Mapping[int, AnalysisPrecision | tuple],
    source_scope: "IterationScope",
) -> Access | None:
    """Resolve one declared boundary into an access from its iteration scope.

    A whole-tuple input boundary names leaf slots rather than tensor elements,
    so it uses the operand's own identity instead of an individual leaf source.
    """
    relation = boundary.relation
    loops = scope.enclosing_loops()
    relation = relation.insert_dims(isl.dim_type.IN, 0, len(loops))
    scope_domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
    )
    relation = relation.intersect_domain(scope_domain)
    try:
        held = (
            ctx.local_type_of(operand) if getattr(ctx, "topology_level", None) is not None
            else local_type_of(operand.type) if narrow else operand.type
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    if output_index is not None and isinstance(held, TupleType):
        held = held.fields[output_index]
    relation, precision = eliminate_parameters(
        relation,
        boundary.values,
        scope,
        held if getattr(ctx, "topology_level", None) is not None else operand.type,
        narrow=narrow,
    )
    if isinstance(held, TensorType) and is_literal_shape(held.shape):
        relation = relation.intersect_range(shape_to_isl_set(tuple(held.shape), {}))
    entry = sources[id(operand)]
    source_precision = precisions[id(operand)]
    source_type = ctx.local_type_of(operand)
    if isinstance(entry, tuple):
        if output_index is not None:
            entry = entry[output_index]
            source_precision = source_precision[output_index]
            source_type = source_type.fields[output_index]
        else:
            identity = identity_access(1).relation
            identity = identity.insert_dims(isl.dim_type.IN, 0, source_scope.depth)
            identity = identity.intersect_domain(
                source_scope.domain.insert_dims(isl.dim_type.SET, source_scope.depth, 1)
            )
            values = {
                name: value for name, value in source_scope.domain_params.items()
                if identity.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
            }
            entry = BufferAlias(operand, AccessRelation(identity, values))
            source_precision = AnalysisPrecision.EXACT
    rank = len(source_type.shape) if isinstance(source_type, TensorType) else 1
    relation = apply_source(relation, entry, rank, scope.depth, scope.domain_params)
    precision = precision.join(source_precision)
    operand = entry.buffer
    if has_unbounded_param(relation):
        precision = AnalysisPrecision.UNKNOWN
    return Access(
        input_index=input_index,
        relation=relation,
        buffer=operand,
        precision=precision,
        output_index=output_index,
        index=entry.index,
        may_access=boundary.may_access,
        occurrence=entry.occurrence,
    )


__all__ = [
    "Access",
    "BufferAlias",
    "aligned_source",
    "apply_source",
    "buffer_alias",
    "eliminate_parameters",
    "resolve_access",
    "widest_allowed",
]
