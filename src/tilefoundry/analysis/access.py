"""Access relations resolved against authored iteration scopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import isl

from tilefoundry.ir.core import Call, Expr
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.utils.isl_utils import has_unbounded_param
from tilefoundry.visitor_registry.access_relation import (
    BoundaryRelation,
    index_set,
    relation_of,
    renaming_relation,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .affine import LoopAffineTerm, loop_affine_term
from .errors import AnalysisError
from .footprint import _widest_allowed


class AccessPrecision(Enum):
    """How faithfully an access relation describes the authored access."""

    EXACT = auto()
    WIDENED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Access:
    """One relation from an iteration scope to the allocation it reaches."""

    relation: isl.map
    buffer: Expr
    precision: AccessPrecision = AccessPrecision.EXACT


def _parameter_term(
    value: object,
    relation: isl.map,
    name: str,
    loops: tuple[LoopRegion, ...],
    held: object,
    *,
    narrow: bool,
) -> tuple[LoopAffineTerm | None, AccessPrecision]:
    number = static_dim_value(value)
    if number is not None:
        return LoopAffineTerm(None, 0, number, number), AccessPrecision.EXACT
    try:
        term = loop_affine_term(value, loops, narrow=narrow)
    except (TypeError, ValueError, NotImplementedError):
        term = None
    if term is not None:
        return term, AccessPrecision.EXACT
    return _widest_allowed(relation, name, held), AccessPrecision.WIDENED


def _constrain_parameter(
    relation: isl.map,
    param_index: int,
    term: LoopAffineTerm,
) -> isl.map:
    local = isl.local_space.from_space(relation.get_space())

    def placed(kind: str, sign: int, constant: int) -> isl.constraint:
        constraint = getattr(isl.constraint, f"alloc_{kind}")(local)
        constraint = constraint.set_coefficient_si(isl.dim_type.PARAM, param_index, sign)
        if term.loop_axis is not None:
            constraint = constraint.set_coefficient_si(
                isl.dim_type.IN, term.loop_axis, -sign * term.stride
            )
        return constraint.set_constant_si(constant)

    if term.low == term.high:
        return relation.add_constraint(placed("equality", 1, -term.low))
    relation = relation.add_constraint(placed("inequality", 1, -term.low))
    return relation.add_constraint(placed("inequality", -1, term.high))


def eliminate_parameters(
    relation: isl.map,
    parameters: Mapping[str, object] | Sequence[tuple[str, object]],
    loops: tuple[LoopRegion, ...],
    held: object,
    *,
    narrow: bool,
) -> tuple[isl.map, AccessPrecision]:
    """Eliminate declared parameters using literals, loop terms, or widening."""
    precision = AccessPrecision.EXACT
    for name, value in dict(parameters).items():
        param_index = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        if param_index < 0:
            raise AnalysisError(f"access pattern parameter {name!r} is missing from its relation")
        term, resolved_precision = _parameter_term(
            value, relation, name, loops, held, narrow=narrow
        )
        if resolved_precision is AccessPrecision.WIDENED:
            precision = AccessPrecision.WIDENED
        if term is not None:
            relation = _constrain_parameter(relation, param_index, term)
        relation = relation.project_out(isl.dim_type.PARAM, param_index, 1)
    return relation, precision


def _enclosing_loops(scope: "IterationScope") -> tuple[LoopRegion, ...]:
    loops = []
    cursor = scope
    while cursor is not None:
        if isinstance(cursor.owner, LoopRegion):
            loops.append(cursor.owner)
        cursor = cursor.parent
    loops.reverse()
    return tuple(loops)


def resolve_access(
    operand: Expr,
    boundary: BoundaryRelation,
    scope: "IterationScope",
    ctx: TypeInferContext,
    *,
    narrow: bool,
) -> Access | None:
    """Resolve one declared boundary into an access from its iteration scope."""
    relation = relation_of(boundary.pattern)
    precision = AccessPrecision.EXACT
    loops = _enclosing_loops(scope)
    relation = relation.insert_dims(isl.dim_type.IN, 0, len(loops))
    scope_domain = scope.domain.insert_dims(
        isl.dim_type.SET, scope.depth, relation.dim(isl.dim_type.IN) - scope.depth
    )
    relation = relation.intersect_domain(scope_domain)
    relation, precision = eliminate_parameters(
        relation,
        getattr(boundary.pattern, "parameters", ()) or (),
        loops,
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
        relation, folded_precision = eliminate_parameters(
            relation,
            folded.parameters,
            loops,
            operand.type,
            narrow=narrow,
        )
        if folded_precision is AccessPrecision.WIDENED:
            precision = AccessPrecision.WIDENED
    if precision is AccessPrecision.EXACT and has_unbounded_param(relation):
        precision = AccessPrecision.UNKNOWN
    return Access(relation, operand, precision)


__all__ = [
    "Access",
    "AccessPrecision",
    "eliminate_parameters",
    "resolve_access",
]
