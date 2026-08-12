"""The shared authored-program gate for analysis and scheduling."""

from __future__ import annotations

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function

from .errors import AnalysisError
from .preflight import infer_authored_types, validate_call_context
from .walk import reachable_functions


def check_program(
    module: Module,
    function: Function,
    *,
    level: str | None = None,
) -> None:
    """Prove one authored program's shared invariants before an algorithm runs."""
    target = module.resolve_target()
    for topology in module.effective_topologies():
        try:
            target.validate_program_topology(topology)
        except ValueError as error:
            raise AnalysisError(
                f"program topology level {topology.name!r} with extent "
                f"{topology.size!r} is invalid: {error}"
            ) from None
    if level is not None:
        try:
            module.resolve_topology(level)
        except ValueError as error:
            raise AnalysisError(f"program topology level {level!r} is invalid: {error}") from None

    functions = reachable_functions(function)
    infer_authored_types(functions, module)
    validate_call_context(module, functions)


__all__ = ["check_program"]
