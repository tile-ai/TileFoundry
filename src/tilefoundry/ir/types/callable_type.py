from __future__ import annotations

from dataclasses import dataclass

from .tensor_type import Type, UnitType


@dataclass(frozen=True)
class CallableType:
    """IR-level type of a callable Expr (e.g. ``hir.Function``).

    Carries the parameter types and the return type. Parameter names
    are not part of the type — at the IR level they live on
    ``Function.params`` (each ``Var.name``); at the host-ABI level
    they live on ``tilefoundry.runtime.module.CallableType`` /
    ``ParamABI``, which is a separate layer.
    """

    return_type: Type
    parameters: tuple[Type, ...]


def callable_type_for(params, return_type: Type) -> CallableType:
    """Project a callee's ``params`` + ``return_type`` into a ``CallableType``.

    Parameter names are not part of the type; only each param's ``.type`` is
    projected, in order.
    """
    return CallableType(
        return_type=return_type,
        parameters=tuple(p.type for p in params),
    )


def callable_type_for_prim_function(fn) -> CallableType:
    """``CallableType`` of a TIR ``PrimFunction`` callee.

    ``CallableType`` of a TIR ``PrimFunction`` callee: its parameter types
    and a ``UnitType`` return — a prim_function returns no value, its outputs
    are trailing params.
    """
    return callable_type_for(fn.params, UnitType())


__all__ = [
    "CallableType",
    "callable_type_for",
    "callable_type_for_prim_function",
]
