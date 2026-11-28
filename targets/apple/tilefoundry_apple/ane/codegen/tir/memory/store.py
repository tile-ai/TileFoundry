"""Codegen for ``T.ane.store``: no task of its own; the task producing its source writes it out."""

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir


@codegen.register(target.Target, codegen.Role.EMIT, tir.Store)
def emit(call: expr.Call, ctx: context.Context) -> None:
    """Emit nothing: ``Context.plan`` hands the destination to the task that produces the source."""


__all__: list[str] = []
