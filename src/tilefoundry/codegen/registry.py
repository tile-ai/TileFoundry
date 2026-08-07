"""Code-generation services selected by concrete Target values."""

from __future__ import annotations

from tilefoundry.ir.core.module import Module
from tilefoundry.target.base import Target, _target_summary
from tilefoundry.target.services import CodeGenerator


def group_functions_by_target(
    module: Module,
) -> dict[Target, tuple["PrimFunction", ...]]:
    """Group functions by their equal Target value, preserving source order."""
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415

    groups: dict[Target, list[PrimFunction]] = {}
    for function in module.functions:
        if not isinstance(function, PrimFunction):
            raise TypeError(
                f"tilefoundry: codegen expects PrimFunction values, got "
                f"{type(function).__name__} {function.name!r}"
            )
        if function.target is None:
            raise ValueError(
                f"tilefoundry: function {function.name!r} has no resolved Target "
                "at codegen grouping"
            )
        groups.setdefault(function.target, []).append(function)

    from tilefoundry.target import CudaTarget  # noqa: PLC0415

    device_groups = [
        (target, functions)
        for target, functions in groups.items()
        if isinstance(target, CudaTarget)
    ]
    if len(device_groups) > 1:
        first_target, first_functions = device_groups[0]
        second_target, second_functions = device_groups[1]
        raise ValueError(
            f"tilefoundry: module {module.name!r} mixes unequal device Targets: "
            f"{_target_summary(first_target)} (function {first_functions[0].name!r}) "
            f"vs {_target_summary(second_target)} (function {second_functions[0].name!r}); "
            "multiple device translation units are not supported"
        )
    return {target: tuple(functions) for target, functions in groups.items()}


__all__ = ["CodeGenerator", "group_functions_by_target"]
