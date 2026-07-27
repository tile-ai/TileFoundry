"""The gate every analysis runs behind.

An analysis reads inferred types and assumes the authored program holds
together. Both conditions are established once per public call rather than
per algorithm, so no family can be the one that forgot.
"""

from __future__ import annotations

from collections.abc import Iterable

from tilefoundry.ir.constraints import ScheduleConstraintMetadata
from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import Type, callable_type_for
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

from .errors import AnalysisError
from .walk import describe, postorder, tensor_types


def infer_authored_types(
    functions: Iterable[Function], module: Module | None
) -> None:
    """Re-derive every authored value type in place.

    Callees are inferred before their callers, so a call site reads a return
    type that has already been recomputed rather than the one it was authored
    with.
    """
    for fn in reversed(tuple(functions)):
        ctx = TypeInferContext(module=module)
        for expr in postorder(fn.body):
            computed = TypeInferVisitor(ctx).visit(expr)
            if computed != expr.type:
                object.__setattr__(expr, "type", computed)
            ctx.cache[id(expr)] = computed
        if fn.body is not None and fn.return_type != fn.body.type:
            object.__setattr__(fn, "return_type", fn.body.type)
            object.__setattr__(fn, "type", callable_type_for(fn.params, fn.body.type))


def _unresolved_local_layout(type_: Type) -> bool:
    """Whether *type_* places data in local storage without saying how.

    A shaped value in registers or shared memory is distributed across the
    threads of its level. Without a layout there is no such distribution, so
    every per-thread number measured from it would be invented.
    """
    return any(
        tensor.storage in {StorageKind.RMEM, StorageKind.SMEM}
        and tensor.shape
        and tensor.layout is None
        for tensor in tensor_types(type_)
    )


def _reject_schedule_constraint(expr: Expr) -> None:
    if get_metadata(expr, ScheduleConstraintMetadata) is None:
        return
    raise AnalysisError(
        f"{describe(expr)}: authored analysis does not accept where(...); "
        "write a concrete layout/storage with Tensor annotations or reshard"
    )


def validate_authored(functions: Iterable[Function]) -> None:
    """Reject an authored program no analysis can measure.

    A schedule constraint means the author deferred a decision to the schedule
    stage, so there is no single program to measure yet; an unresolved local
    layout means distribution inference stopped short of one.
    """
    for fn in functions:
        for expr in (*fn.params, *postorder(fn.body)):
            _reject_schedule_constraint(expr)
            if isinstance(expr, Call) and _unresolved_local_layout(expr.type):
                raise AnalysisError(
                    f"{describe(expr)}: distribution inference stopped with an "
                    "unresolved layout"
                )
        if fn.body is not None and _unresolved_local_layout(fn.body.type):
            raise AnalysisError(
                f"function {fn.name!r} result: distribution inference stopped "
                "with an unresolved layout"
            )


__all__ = ["infer_authored_types", "validate_authored"]
