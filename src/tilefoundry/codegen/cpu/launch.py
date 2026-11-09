"""How a host entry launches a function another target defines."""

from __future__ import annotations

from tilefoundry.codegen import registry
from tilefoundry.ir import tir
from tilefoundry.target import cpu


@registry.register(cpu.Target, registry.Role.EMIT, tir.launch.Launch)
def emit_launch(entry, stmt, ctx):
    """A launch is written by the callee's target, which states how its function is called."""
    callee = ctx.callee_of(stmt)
    callee_target = ctx.codegen_context.targets[id(callee)]
    return ctx.lookup((type(callee_target), registry.Role.LAUNCH, tir.launch.Launch))(entry, stmt, ctx)


__all__: list[str] = []
