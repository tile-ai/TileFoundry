"""The topology questions every emitter asks of a module and its target.

Host module, device module and the entry signature describe three segments of
one call, so they read the answer here rather than each deciding for
themselves.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from tilefoundry.ir.core.module import Module, module_functions, subtree
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.types.shape_helpers import static_dim_value

Geometry = tuple[tuple[int | None, int, int], tuple[int, int, int]]
"""The grid and block one launch runs at; a launch-provided CTA extent is ``None``."""


def coarsest_topology(module) -> str:
    """The coarsest topology level this module's program names.

    ``effective_topologies()`` is ordered coarsest first, so the first entry
    says where this program's run of levels begins; a module that names none
    is one CTA.
    """
    declared = module.effective_topologies()
    return declared[0].name if declared else "cta"


def topology_domains(root: Module) -> Iterator[tuple[Module, tuple]]:
    """Each module that states topologies of its own, and the functions under it.

    A program states its instance counts once, so a module that states its own
    is its own translation unit; one that states none runs in its owner's
    program and its functions are emitted into the owner's unit.
    """
    yield root, _domain_functions(root)
    for node in subtree(root):
        for child in node.modules:
            if child.topologies is not None:
                yield child, _domain_functions(child)


def _domain_functions(node: Module) -> tuple:
    """*node*'s own functions and those of every child that states no topologies."""
    inherited = tuple(
        function
        for child in node.modules
        if child.topologies is None
        for function in _domain_functions(child)
    )
    return (*node.functions, *inherited)


def launch_geometry(root: Module) -> Mapping[int, Geometry]:
    """The geometry each launched function runs at, as its ``Launch`` states it.

    The host entry settled it when it was written, so the device side reads it
    rather than walking a body to derive the same answer a second time.
    """
    geometry: dict[int, Geometry] = {}
    for function in module_functions(root):
        body = function.body
        if not isinstance(body, Sequential):
            continue
        for stmt in body.body:
            if not (isinstance(stmt, Evaluate) and isinstance(stmt.callable, Launch)):
                continue
            callee = root.lookup(stmt.args[0].name)
            geometry[id(callee)] = (
                _grid(stmt.args[1:4], callee.name),
                _block(stmt.args[4:7], callee.name),
            )
    return geometry


def _grid(args, name: str) -> tuple:
    """The three grid extents, whose first may be the one a launch provides.

    A launch-provided CTA count is the one extent a translation unit cannot
    state; the runtime states that one when it is asked.
    """
    return (
        static_dim_value(args[0]),
        *(_static(arg, name, "grid", axis + 1) for axis, arg in enumerate(args[1:])),
    )


def _block(args, name: str) -> tuple:
    """The three block extents, which multiply into the thread count a unit states."""
    return tuple(_static(arg, name, "block", axis) for axis, arg in enumerate(args))


def _static(arg, name: str, what: str, axis: int) -> int:
    """One extent as the compile-time count the program dimensions need."""
    value = static_dim_value(arg)
    if value is None:
        raise ValueError(
            f"launch of {name!r}: {what}[{axis}] is decided at run time, and "
            f"only a launch-provided CTA count can be"
        )
    return value


__all__ = [
    "Geometry",
    "coarsest_topology",
    "launch_geometry",
    "topology_domains",
]
