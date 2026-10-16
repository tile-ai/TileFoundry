"""Access relations resolved against authored iteration scopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import isl

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var, binding_name
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, instruction_schema
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import IslParamValues, dim_to_isl_pw_aff, shape_to_isl_set
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.utils import is_literal_shape, local_type_of, static_dim_value
from tilefoundry.ir.visitor import ExprCloner, collect_exprs
from tilefoundry.utils.isl_utils import cardinality, has_unbounded_param
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    leaf_span,
    leaves_of,
    renaming_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .errors import AnalysisError
from .precision import AnalysisPrecision


@dataclass(frozen=True, eq=False)
class SSAValue:
    """One expression's flattened result, compared by SSA identity."""

    expr: Expr
    index: int = 0

    def __eq__(self, other):
        return isinstance(other, SSAValue) and self.expr is other.expr and self.index == other.index

    def __hash__(self):
        return hash((id(self.expr), self.index))


def forwarded_leaves(expr, ctx):
    """Select direct operand leaves for structural forwarding, before source lookup."""
    if isinstance(expr, Tuple):
        values = expr.elements
    elif isinstance(expr, LoopRegion):
        values = expr.params[: len(expr.yield_values)] or (expr.body,)
    elif isinstance(expr, MeshRegion):
        values = (expr.body,)
    elif isinstance(expr, Var) and (argument := ctx.capture_root(expr)) is not expr:
        values = (argument,)
    elif (
        isinstance(expr, Call)
        and isinstance(expr.target, TupleGetItem)
        and isinstance(expr.args[1], Constant)
        and type(expr.args[1].value) is int
    ):
        begin, count = leaf_span(expr.args[0].type, expr.args[1].value)
        return tuple(SSAValue(expr.args[0], i) for i in range(begin, begin + count))
    else:
        return None
    return tuple(SSAValue(item, i) for item in values for i in range(len(leaves_of(item.type))))


class AliasKind(IntEnum):
    """Structural sharing first, then the deterministic tie order."""

    VIEW = 0
    FORWARD = 1
    CARRY_INIT = 2
    CARRY_YIELD = 3
    INSERT_SLICE = 4
    WINDOW = 5
    RETURN_REUSE = 6
    IN_PLACE = 7


@dataclass(frozen=True)
class Access:
    """One relation from an iteration scope to an SSA result."""

    input_index: int | None
    relation: isl.map
    target: SSAValue
    precision: AnalysisPrecision = AnalysisPrecision.EXACT
    output_index: int | None = None
    may_access: bool = False


@dataclass(frozen=True)
class BufferAlias:
    """A result's coordinates in one SSA result's storage."""

    target: SSAValue
    relation: AccessRelation
    kind: AliasKind


def referenced_values(relation, values):
    """Keep parameter evidence in its supplied order."""
    return {
        name: value
        for name, value in values.items()
        if relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
    }


def aligned_source(
    source: BufferAlias, rank: int, depth: int, parameters: Mapping[str, object] | None = None
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
    relation: isl.map,
    source: BufferAlias,
    rank: int,
    depth: int,
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
    return BufferAlias(SSAValue(operand), relation, AliasKind.VIEW)


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

    def visit_Call(self, value: Call, scope: "IterationScope") -> Expr:
        if isinstance(value.target, Local):
            return (
                Constant(type=value.type, value=0)
                if self.narrow
                else self.visit(value.args[0], scope)
            )
        return self.default_visit(value, scope)

    def visit_Var(self, value: Var, scope: "IterationScope") -> Expr:
        root = scope.context.capture_root(value)
        return value if root is value else self.visit(root, scope)


def _parameter_term(
    value: object,
    relation: isl.map,
    name: str,
    scope: "IterationScope",
    held: object,
    *,
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
        coords = {
            id(loop.induction_var): f"__tf_in_{i}" for i, loop in enumerate(scope.context.enclosing_loops())
        }
        values: IslParamValues = {
            param: scope.context.capture_root(parameter) for param, parameter in scope.domain_params.items()
        }
        term = dim_to_isl_pw_aff(value, values, coords=coords).intersect_params(
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
    formals: tuple[Var, ...] = (),
    keep_possible: bool = False,
) -> tuple[isl.map, AnalysisPrecision]:
    """Eliminate scoped parameters, retaining formal-dependent coordinates for callers."""
    precision = AnalysisPrecision.EXACT
    rank = relation.dim(isl.dim_type.IN)
    for name, value in parameters.items():
        if relation.find_dim_by_name(isl.dim_type.PARAM, name) < 0:
            raise AnalysisError(f"access pattern parameter {name!r} is missing from its relation")
        resolved = (
            _RegionBindingResolver(narrow=narrow).visit(value, scope)
            if isinstance(value, Expr)
            else value
        )
        if isinstance(resolved, Expr) and any(
            expr is formal for expr in collect_exprs(resolved) for formal in formals
        ):
            if isinstance(resolved, Var):
                label = f"formal_{next(index for index, formal in enumerate(formals) if resolved is formal)}"
            else:
                label = f"value_{binding_name(value)}"
            target = f"__tf_binding_{label}"
            present = relation.find_dim_by_name(isl.dim_type.PARAM, target)
            position = relation.find_dim_by_name(isl.dim_type.PARAM, name)
            if present >= 0 and present != position:
                relation = relation.equate(
                    isl.dim_type.PARAM, position, isl.dim_type.PARAM, present
                )
                relation = relation.project_out(isl.dim_type.PARAM, position, 1)
            else:
                relation = relation.set_dim_name(isl.dim_type.PARAM, position, target)
            if retained is not None:
                retained[target] = resolved
            continue
        term, resolved_precision = _parameter_term(
            resolved, relation, name, scope, held, retained=retained
        )
        if keep_possible and resolved_precision is not AnalysisPrecision.EXACT:
            term = None
        if term is not None:
            term = term.add_dims(isl.dim_type.IN, rank - term.dim(isl.dim_type.IN))
            inputs = ", ".join(f"i{index}" for index in range(rank))
            named = isl.pw_aff(f"[{name}] -> {{ [{inputs}] -> [{name}] }}")
            relation = relation.intersect_domain(named.eq_set(term))
        precision = precision.join(resolved_precision)
        param_index = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        relation = relation.project_out(isl.dim_type.PARAM, param_index, 1)
    return relation, precision


def loop_parameter(value: Var) -> str:
    """One coordinate name per induction identity, across call boundaries."""
    return f"__tf_loop_{id(value)}"


def name_loop_parameters(relation: isl.map, inductions) -> tuple[isl.map, dict]:
    values = {}
    for axis, induction in reversed(tuple(enumerate(inductions))):
        name = loop_parameter(induction)
        parameter = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        if parameter < 0:
            relation = relation.insert_dims(isl.dim_type.PARAM, 0, 1)
            relation = relation.set_dim_name(isl.dim_type.PARAM, 0, name)
            parameter = 0
        relation = relation.equate(isl.dim_type.IN, axis, isl.dim_type.PARAM, parameter)
        values[name] = induction
    return relation, values


def restore_loop_dimensions(relation: isl.map, inductions) -> isl.map:
    """Return named loop parameters to their local input coordinates."""
    for axis, induction in enumerate(inductions):
        parameter = relation.find_dim_by_name(isl.dim_type.PARAM, loop_parameter(induction))
        if parameter >= 0:
            relation = relation.equate(isl.dim_type.IN, axis, isl.dim_type.PARAM, parameter)
            relation = relation.project_out(isl.dim_type.PARAM, parameter, 1)
    return relation


def bind_parameters(
    relation: isl.map,
    values: Mapping[str, object],
    binding: BindingValuation,
    formals: tuple[Var, ...],
    scope: IterationScope,
    held: object,
    *,
    narrow: bool,
    retained_formals: tuple[Var, ...] = (),
) -> tuple[isl.map, AnalysisPrecision]:
    """Substitute formal-dependent coordinate parameters through one call."""

    class BindParameters(ExprCloner):
        def visit_Var(self, node, ctx=None):
            return (
                binding.actual(node).target.expr
                if any(node is formal for formal in formals)
                else node
            )

    bound = {}
    for name, value in values.items():
        if name.startswith("__tf_loop_"):
            continue
        position = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        if position < 0:
            continue
        temporary = f"__tf_callee_{len(bound)}"
        while relation.find_dim_by_name(isl.dim_type.PARAM, temporary) >= 0:
            temporary += "_"
        relation = relation.set_dim_name(isl.dim_type.PARAM, position, temporary)
        bound[temporary] = BindParameters().visit(value) if isinstance(value, Expr) else value
    return eliminate_parameters(
        relation,
        bound,
        scope,
        held,
        narrow=narrow,
        retained=scope.domain_params,
        formals=retained_formals,
    )


def resolve_access(
    operand: Expr,
    boundary: AccessRelation,
    scope: "IterationScope",
    ctx: TypeInferContext,
    *,
    input_index: int | None,
    output_index: int | None = None,
    narrow: bool,
    sources: Mapping[SSAValue, BufferAlias],
    precisions: Mapping[SSAValue, AnalysisPrecision],
    formals: tuple[Var, ...] = (),
) -> tuple[Access, ...]:
    """Resolve a boundary with one cached-source lookup per tensor leaf.

    Whole-tuple boundaries select leaf slots. Each possible slot becomes a
    separate access to that tuple value's leaf, in tensor-element coordinates,
    over the selected iteration domain and the leaf's legal element box.
    """
    relation = boundary.relation.insert_dims(isl.dim_type.IN, 0, scope.depth)
    relation = relation.intersect_domain(
        scope.domain.insert_dims(
            isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
        )
    )
    try:
        held = (
            ctx.local_type_of(operand)
            if getattr(ctx, "topology_level", None) is not None
            else local_type_of(operand.type)
            if narrow
            else operand.type
        )
    except (TypeError, ValueError, NotImplementedError):
        return ()
    types = leaves_of(held)
    if not types:
        return ()
    whole_tuple = isinstance(held, TupleType) and output_index is None
    if whole_tuple:
        indexes = range(len(types))
    else:
        begin = leaf_span(held, output_index)[0] if isinstance(held, TupleType) else 0
        indexes = (begin,)
    built = []
    for index in indexes:
        leaf = types[index]
        access, precision = eliminate_parameters(
            relation,
            boundary.values,
            scope,
            leaf
            if getattr(ctx, "topology_level", None) is not None
            else leaves_of(operand.type)[index],
            narrow=narrow,
            retained=scope.domain_params,
            formals=formals,
            keep_possible=whole_tuple,
        )
        if whole_tuple:
            selected = access.fix_val(isl.dim_type.OUT, 0, isl.val(index))
            if selected.is_empty():
                continue
            box = shape_to_isl_set(tuple(leaf.shape), {})
            access = isl.map.from_domain_and_range(selected.domain(), box)
            target = SSAValue(operand, index)
        else:
            if is_literal_shape(leaf.shape):
                access = access.intersect_range(shape_to_isl_set(tuple(leaf.shape), {}))
            value = SSAValue(operand, index)
            entry = sources[value]
            access = apply_source(access, entry, len(leaf.shape), scope.depth, scope.domain_params)
            precision = precision.join(precisions[value])
            target = entry.target
        if has_unbounded_param(access) and any(
            access.get_dim_name(isl.dim_type.PARAM, axis) not in scope.domain_params
            for axis in range(access.dim(isl.dim_type.PARAM))
        ):
            precision = AnalysisPrecision.UNKNOWN
        built.append(
            Access(
                input_index=input_index,
                relation=access,
                target=target,
                precision=precision,
                output_index=output_index,
                may_access=boundary.may_access,
            )
        )
    return tuple(built)


__all__ = [
    "Access",
    "SSAValue",
    "AliasKind",
    "BufferAlias",
    "aligned_source",
    "apply_source",
    "buffer_alias",
    "eliminate_parameters",
    "resolve_access",
    "widest_allowed",
]
