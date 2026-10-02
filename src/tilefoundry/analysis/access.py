"""Access relations resolved against authored iteration scopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import isl

from tilefoundry.ir.core import Call, Constant, Expr, Var
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.isl_interop import dim_range, dim_to_isl_expr, index_set
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.utils import local_type_of, static_dim_value
from tilefoundry.ir.visitor import ExprCloner
from tilefoundry.utils.isl_utils import cardinality, has_unbounded_param
from tilefoundry.visitor_registry.access_relation import (
    BoundaryRelation,
    relation_of,
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
    box = index_set(tuple(held.shape)) if isinstance(held, TensorType) else None
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
) -> tuple[str | None, dict[str, tuple[int, int] | None], AnalysisPrecision]:
    number = static_dim_value(value)
    if number is not None:
        return str(number), {}, AnalysisPrecision.EXACT
    try:
        resolved = _RegionBindingResolver(narrow=narrow).visit(value, scope)
        identities = {
            id(loop.induction_var): f"__tf_in_{i}" for i, loop in enumerate(scope.enclosing_loops())
        }
        param_map = dict(scope.domain_params)
        identities.update(
            (id(scope.capture_root(parameter)), param) for param, parameter in param_map.items()
        )
        params = {param: dim_range(parameter) for param, parameter in param_map.items()}
        expression = dim_to_isl_expr(resolved, params, param_map=param_map, identities=identities)
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        pass
    else:
        if all(bound is not None for bound in params.values()):
            return expression, params, AnalysisPrecision.EXACT
    number = widest_allowed(relation, name, held)
    return None if number is None else str(number), {}, AnalysisPrecision.UPPER_BOUND


def eliminate_parameters(
    relation: isl.map,
    parameters: Mapping[str, object] | Sequence[tuple[str, object]],
    scope: "IterationScope",
    held: object,
    *,
    narrow: bool,
) -> tuple[isl.map, AnalysisPrecision]:
    """Eliminate parameters using scoped affine expressions or widening."""
    precision = AnalysisPrecision.EXACT
    inputs = ", ".join(f"__tf_in_{i}" for i in range(relation.dim(isl.dim_type.IN)))
    outputs = ", ".join(f"__tf_out_{i}" for i in range(relation.dim(isl.dim_type.OUT)))
    names = dict.fromkeys(
        relation.get_dim_name(isl.dim_type.PARAM, i)
        for i in range(relation.dim(isl.dim_type.PARAM))
    )
    for name, value in dict(parameters).items():
        if name not in names:
            raise AnalysisError(f"access pattern parameter {name!r} is missing from its relation")
        expression, params, resolved_precision = _parameter_term(
            value, relation, name, scope, held, narrow=narrow
        )
        if expression is not None:
            names.update(dict.fromkeys(params))
            conditions = " and ".join(
                (
                    f"{name} = {expression}",
                    *(f"{lo} <= {param} < {hi}" for param, (lo, hi) in params.items()),
                )
            )
            relation = relation.intersect(
                isl.map(f"[{', '.join(names)}] -> {{ [{inputs}] -> [{outputs}] : {conditions} }}")
            )
        precision = precision.join(resolved_precision)
        param_index = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        relation = relation.project_out(isl.dim_type.PARAM, param_index, 1)
        del names[name]
    return relation, precision


def resolve_access(
    operand: Expr,
    boundary: BoundaryRelation,
    scope: "IterationScope",
    ctx: TypeInferContext,
    *,
    input_index: int | None,
    output_index: int | None = None,
    narrow: bool,
) -> Access | None:
    """Resolve one declared boundary into an access from its iteration scope."""
    relation = relation_of(boundary.pattern)
    loops = scope.enclosing_loops()
    relation = relation.insert_dims(isl.dim_type.IN, 0, len(loops))
    scope_domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
    )
    relation = relation.intersect_domain(scope_domain)
    relation, precision = eliminate_parameters(
        relation,
        getattr(boundary.pattern, "parameters", ()) or (),
        scope,
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
    operand = scope.capture_root(operand)
    while isinstance(operand, Call) and (position := aliased_operand(operand)) is not None:
        folded = renaming_relation(operand, ctx, scope.projected_relations(operand, ctx))
        relation = relation.apply_range(relation_of(folded))
        operand = scope.capture_root(operand.args[position])
        relation, folded_precision = eliminate_parameters(
            relation,
            folded.parameters,
            scope,
            operand.type,
            narrow=narrow,
        )
        precision = precision.join(folded_precision)
    if has_unbounded_param(relation):
        precision = AnalysisPrecision.UNKNOWN
    return Access(
        input_index=input_index,
        relation=relation,
        buffer=operand,
        precision=precision,
        output_index=output_index,
    )


__all__ = [
    "Access",
    "eliminate_parameters",
    "resolve_access",
    "widest_allowed",
]
