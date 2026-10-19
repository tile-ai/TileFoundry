"""Access relations resolved against authored iteration scopes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import isl

from tilefoundry.ir.core import Call, Constant, Expr, SSAValue, Tuple, Var
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, instruction_schema
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.utils import (
    is_literal_shape,
    leaf_span,
    leaves_of,
    local_type_of,
)
from tilefoundry.ir.visitor import ExprCloner, collect_exprs
from tilefoundry.utils.isl_utils import cardinality, has_unbounded_param
from tilefoundry.visitor_registry.access_relation import AccessRelation, renaming_relation
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .errors import AnalysisError
from .precision import AnalysisPrecision


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
    IN_PLACE = 6


@dataclass(frozen=True)
class SnapshotEntry:
    """Data identity and coordinates, shared by every use of a definition value."""

    root: SSAValue
    narrow: AccessRelation
    device: AccessRelation
    unit: AccessRelation | None
    varies: frozenset[int] = frozenset()
    forward: SSAValue | None = None


@dataclass(frozen=True)
class Access:
    input_index: int | None
    relation: AccessRelation
    target: SSAValue
    output_index: int | None = None
    may_access: bool = False


@dataclass(frozen=True)
class BufferAlias:
    """A result's storage coordinates; ``writes`` marks mutations in the composed callee chain."""

    target: SSAValue
    relation: AccessRelation
    kind: AliasKind
    writes: bool = False


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
        root = scope.capture_root(value)
        return value if root is value else self.visit(root, scope)


def eliminate_parameters(
    relation: AccessRelation,
    scope,
    held,
    *,
    narrow,
    formals=(),
    keep_possible=False,
) -> tuple[AccessRelation, AnalysisPrecision]:
    """Bind affine values, retain definition inputs, and widen unresolved runtime values."""
    result = relation.normalized(scope)
    precision = AnalysisPrecision.EXACT
    retained = {id(formal) for formal in formals}
    induction = {id(loop.owner.induction_var) for loop in scope.loop_scopes}
    for name, value in tuple(result.values.items()):
        resolved = scope.access.resolve_parameter(value, view="narrow" if narrow else "device")
        if resolved is value and id(value) in induction:
            continue
        if (
            retained
            and isinstance(resolved, Expr)
            and any(id(expr) in retained for expr in collect_exprs(resolved))
        ):
            continue
        try:
            term, values = scope.access._affine(resolved)
            term = term.intersect_params(scope.domain.relation.params())
        except (TypeError, ValueError, NotImplementedError, isl.Error):
            term = None
        if term is not None and not has_unbounded_param(term):
            named = isl.pw_aff(f"[{name}] -> {{ [{name}] }}")
            result = result.constrain(
                named.eq_set(term).params(), {**scope.domain.values, **values}
            )
            if name not in values:
                result = result.project((value,))
            continue
        precision = precision.join(AnalysisPrecision.UPPER_BOUND)
        if keep_possible:
            continue
        number = widest_allowed(result.relation, name, held)
        if number is not None:
            result = result.constrain(isl.set(f"[{name}] -> {{ : {name} = {number} }}"))
            result = result.project((value,))
        else:
            precision = precision.join(AnalysisPrecision.UNKNOWN)
    return result.constrain(scope.domain), precision


def resolve_access(
    operand,
    boundary,
    scope,
    ctx,
    *,
    input_index,
    output_index=None,
    view="narrow",
    formals=(),
):
    """Resolve registered boundaries through the shared value snapshots."""
    held = (
        ctx.local_type_of(operand)
        if view == "unit"
        else (local_type_of(operand.type) if view == "narrow" else operand.type)
    )
    types = leaves_of(held)
    whole_tuple = isinstance(held, TupleType) and output_index is None
    indexes = (
        range(len(types))
        if whole_tuple
        else (leaf_span(held, output_index)[0] if isinstance(held, TupleType) else 0,)
    )
    built = []
    for index in indexes:
        leaf = types[index]
        relation, _ = eliminate_parameters(
            boundary,
            scope,
            leaf,
            narrow=view == "narrow",
            formals=formals,
            keep_possible=True,
        )
        value = SSAValue(operand, index)
        if whole_tuple:
            selected = relation.relation.fix_val(isl.dim_type.OUT, 0, isl.val(index))
            if selected.is_empty():
                continue
            mapped = isl.map.from_domain_and_range(
                selected.domain(), shape_to_isl_set(tuple(leaf.shape), {})
            )
            relation = relation._result(mapped)
            target = value
        else:
            entry = scope.provenance.lookup(value)
            source = getattr(entry, view)
            if source is None:
                continue
            if is_literal_shape(leaf.shape):
                relation = relation._result(
                    relation.relation.intersect_range(shape_to_isl_set(tuple(leaf.shape), {}))
                )
            relation = relation.compose(source)
            target = entry.root
        built.append(Access(input_index, relation, target, output_index, boundary.may_access))
    return tuple(built)


STRUCTURAL_KINDS = frozenset({AliasKind.VIEW, AliasKind.FORWARD})
SEMANTIC_KINDS = frozenset({AliasKind.INSERT_SLICE, AliasKind.FORWARD})
