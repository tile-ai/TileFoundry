"""Access relations resolved against authored iteration scopes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import core, dim, expr, hir, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.hir import sharding, tensor

from .errors import Error
from .precision import Precision


def forwarded_leaves(node, ctx):
    """Select direct operand leaves for structural forwarding, before source lookup."""
    if isinstance(node, expr.Tuple):
        values = node.elements
    elif isinstance(node, core.LoopRegion):
        values = node.params[: len(node.yield_values)] or (node.body,)
    elif isinstance(node, core.MeshRegion):
        values = (node.body,)
    elif isinstance(node, expr.Var) and (argument := ctx.capture_root(node)) is not node:
        values = (argument,)
    elif (
        isinstance(node, expr.Call)
        and isinstance(node.target, tensor.TupleGetItem)
        and isinstance(node.args[1], expr.Constant)
        and type(node.args[1].value) is int
    ):
        begin, count = ir_utils.leaf_span(node.args[0].type, node.args[1].value)
        return tuple(core.SSAValue(node.args[0], i) for i in range(begin, begin + count))
    else:
        return None
    return tuple(core.SSAValue(item, i) for item in values for i in range(len(ir_utils.leaves_of(item.type))))


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

    root: core.SSAValue
    narrow: access.Relation
    device: access.Relation
    unit: access.Relation | None
    varies: frozenset[int] = frozenset()
    forward: core.SSAValue | None = None


@dataclass(frozen=True)
class Access:
    input_index: int | None
    relation: access.Relation
    target: core.SSAValue
    output_index: int | None = None
    may_access: bool = False


@dataclass(frozen=True)
class BufferAlias:
    """A result's storage coordinates; ``writes`` marks mutations in the composed callee chain."""

    target: core.SSAValue
    relation: access.Relation
    kind: AliasKind
    writes: bool = False


def buffer_alias(
    call: expr.Call, ctx: typeinfer.TypeInferContext, local_relations: tuple[access.Relation, ...]
) -> BufferAlias | None:
    """A registered view's bytes and coordinates, proving the declaration on every ask."""
    position = access.aliased_operand(call)
    if position is None:
        return None
    operand = call.args[position]
    relation = access.renaming_relation(call, ctx, local_relations)
    shape = tuple(operand.type.shape)
    box = dim.shape_to_isl_set(shape, {}) if expr.is_literal_shape(shape) else None
    if (
        box is None
        or not relation.relation.is_single_valued()
        or not relation.relation.is_injective()
        or not relation.relation.range().is_subset(box)
    ):
        params = (
            hir.schedule.instruction_schema(call.target.op, len(call.args))[0]
            if isinstance(call.target, hir.schedule.ScheduleOp)
            else tuple(
                param for param in expr.schema_of(type(call.target)).signature if param.kind == "input"
            )
        )
        param = params[position]
        raise Error(
            f"{type(call.target).__name__} declares its result is {param.name}'s bytes, "
            f"but its access relation {relation.relation} is not single-valued, injective, "
            "and within the operand"
        )
    return BufferAlias(core.SSAValue(operand), relation, AliasKind.VIEW)


def widest_allowed(acc: isl.map, name: str, held: object) -> int | None:
    """The value a parameter may take that reaches the most of its operand.

    A footprint is an upper bound, so a parameter nobody here can place takes
    whichever end of its legal range touches more: where a window sits does not
    change how much of it there is, but how long it is does. Both ends are the
    Op's own contract, read off the relation rather than guessed.
    """
    names = [
        acc.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(acc.dim(isl.dim_type.PARAM))
    ]
    space = f"[{', '.join(names)}] -> "
    probe = isl.set(f"{space}{{ [x] : x = {name} }}").intersect_params(acc.params())
    ends = (probe.dim_min_val(0), probe.dim_max_val(0))
    if not all(end.is_int() for end in ends):
        return None
    box = (
        dim.shape_to_isl_set(tuple(held.shape), {})
        if isinstance(held, types.TensorType) and expr.is_literal_shape(held.shape)
        else None
    )
    if box is None or box.dim(isl.dim_type.SET) != acc.dim(isl.dim_type.OUT):
        return ends[0].get_num_si()
    best: tuple[int, int] | None = None
    for value in sorted({end.get_num_si() for end in ends}):
        reach = (
            acc.intersect_params(isl.set(f"{space}{{ : {name} = {value} }}"))
            .range()
            .intersect(box)
        )
        amount = utils.cardinality(reach)
        if amount is None:
            return None
        if best is None or amount > best[0]:
            best = (amount, value)
    return None if best is None else best[1]


class RegionBindingResolver(core.ExprCloner):
    """Resolve region captures and local coordinates before affine rendering."""

    def __init__(self, *, narrow: bool) -> None:
        super().__init__()
        self.narrow = narrow

    def visit_Call(self, value: expr.Call, scope: "IterationScope") -> expr.Expr:
        if isinstance(value.target, sharding.Local):
            return (
                expr.Constant(type=value.type, value=0)
                if self.narrow
                else self.visit(value.args[0], scope)
            )
        return self.default_visit(value, scope)

    def visit_Var(self, value: expr.Var, scope: "IterationScope") -> expr.Expr:
        root = scope.capture_root(value)
        return value if root is value else self.visit(root, scope)


def eliminate_parameters(
    relation: access.Relation,
    scope,
    held,
    *,
    narrow,
    formals=(),
    keep_possible=False,
) -> tuple[access.Relation, Precision]:
    """Bind affine values, retain definition inputs, and widen unresolved runtime values."""
    result = relation.normalized(scope)
    precision = Precision.EXACT
    retained = {id(formal) for formal in formals}
    induction = {id(loop.owner.induction_var) for loop in scope.loop_scopes}
    for name, value in tuple(result.values.items()):
        resolved = scope.access.resolve_parameter(value, view="narrow" if narrow else "device")
        if resolved is value and id(value) in induction:
            continue
        if (
            retained
            and isinstance(resolved, expr.Expr)
            and any(id(node) in retained for node in core.collect_exprs(resolved))
        ):
            continue
        try:
            term, values = scope.access.affine(resolved)
            term = term.intersect_params(scope.domain.relation.params())
        except (TypeError, ValueError, NotImplementedError, isl.Error):
            term = None
        if term is not None and not utils.has_unbounded_param(term):
            named = isl.pw_aff(f"[{name}] -> {{ [{name}] }}")
            result = result.constrain(
                named.eq_set(term).params(), {**scope.domain.values, **values}
            )
            if name not in values:
                result = result.project((value,))
            continue
        precision = precision.join(Precision.UPPER_BOUND)
        if keep_possible:
            continue
        number = widest_allowed(result.relation, name, held)
        if number is not None:
            result = result.constrain(isl.set(f"[{name}] -> {{ : {name} = {number} }}"))
            result = result.project((value,))
        else:
            precision = precision.join(Precision.UNKNOWN)
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
        else (ir_utils.local_type_of(operand.type) if view == "narrow" else operand.type)
    )
    type_list = ir_utils.leaves_of(held)
    whole_tuple = isinstance(held, types.TupleType) and output_index is None
    indexes = (
        range(len(type_list))
        if whole_tuple
        else (ir_utils.leaf_span(held, output_index)[0] if isinstance(held, types.TupleType) else 0,)
    )
    built = []
    for index in indexes:
        leaf = type_list[index]
        relation, _ = eliminate_parameters(
            boundary,
            scope,
            leaf,
            narrow=view == "narrow",
            formals=formals,
            keep_possible=True,
        )
        value = core.SSAValue(operand, index)
        if whole_tuple:
            selected = relation.relation.fix_val(isl.dim_type.OUT, 0, isl.val(index))
            if selected.is_empty():
                continue
            mapped = isl.map.from_domain_and_range(
                selected.domain(), dim.shape_to_isl_set(tuple(leaf.shape), {})
            )
            relation = relation.aligned(mapped)
            target = value
        else:
            entry = scope.provenance.lookup(value)
            source = getattr(entry, view)
            if source is None:
                continue
            if expr.is_literal_shape(leaf.shape):
                relation = relation.aligned(
                    relation.relation.intersect_range(dim.shape_to_isl_set(tuple(leaf.shape), {}))
                )
            relation = relation.compose(source)
            target = entry.root
        built.append(Access(input_index, relation, target, output_index, boundary.may_access))
    return tuple(built)


STRUCTURAL_KINDS = frozenset({AliasKind.VIEW, AliasKind.FORWARD})
SEMANTIC_KINDS = frozenset({AliasKind.INSERT_SLICE, AliasKind.FORWARD})


__all__ = [
    "Access",
    "AliasKind",
    "BufferAlias",
    "RegionBindingResolver",
    "SEMANTIC_KINDS",
    "STRUCTURAL_KINDS",
    "SnapshotEntry",
    "buffer_alias",
    "eliminate_parameters",
    "forwarded_leaves",
    "resolve_access",
    "widest_allowed",
]
