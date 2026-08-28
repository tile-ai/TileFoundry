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
from tilefoundry.ir.hir.tensor.reshape import is_induction_var_singleton_reshape
from tilefoundry.ir.types import Type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

from .errors import AnalysisError
from .walk import called_functions, collect_exprs, describe, owning_module, tensor_types


def infer_authored_types(
    functions: Iterable[Function], module: Module | None
) -> None:
    """Re-derive every authored value type in place.

    Callees are inferred before their callers, so a call site reads a return
    type that has already been recomputed rather than the one it was authored
    with.
    """
    for fn in reversed(tuple(functions)):
        ctx = TypeInferContext(scope=FunctionScope(module, fn))
        if fn.body is not None:
            TypeInferVisitor(owns_body=True).visit(fn.body, ctx)


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
        for expr in (*fn.params, *collect_exprs(fn.body)):
            _reject_schedule_constraint(expr)
            if (
                isinstance(expr, Call)
                and _unresolved_local_layout(expr.type)
                and not is_induction_var_singleton_reshape(expr)
            ):
                raise AnalysisError(
                    f"{describe(expr)}: distribution inference stopped with an "
                    "unresolved layout"
                )
        if fn.body is not None and _unresolved_local_layout(fn.body.type):
            raise AnalysisError(
                f"function {fn.name!r} result: distribution inference stopped "
                "with an unresolved layout"
            )


def validate_call_context(module: Module, functions: Iterable[Function]) -> None:
    """Reject a reached call whose two ends do not share one execution context.

    Checked over what the selected query reaches, not at construction: an
    attached child no call reaches has no edge to validate. Inheritance is the
    canonical spelling, so a child declaring the caller's hierarchy explicitly
    passes and any other value is a different context, not a nested launch.
    """
    for caller in functions:
        caller_owner = owning_module(module, caller)
        for callee in called_functions(caller):
            callee_owner = owning_module(module, callee)
            if callee_owner is caller_owner:
                continue
            if not any(child is callee_owner for child in caller_owner.modules):
                raise AnalysisError(
                    f"{caller.name!r} calls {callee.name!r} of module "
                    f"{callee_owner.name!r}, which is not a child of "
                    f"{caller_owner.name!r}; a call reaches a child of its own module"
                )
            if callee_owner.effective_topologies() != caller_owner.effective_topologies():
                raise AnalysisError(
                    f"{caller_owner.name!r} calls {callee_owner.name!r}, which "
                    f"resolves a different topology hierarchy "
                    f"{callee_owner.effective_topologies()} against "
                    f"{caller_owner.effective_topologies()}; one kernel invocation "
                    f"runs one hierarchy -- declare none on the child and inherit"
                )


__all__ = ["infer_authored_types", "validate_authored", "validate_call_context"]
