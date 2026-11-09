"""Emit effect-form matrix multiply-accumulate operations.

The SM80 and SM89 declarations each emit one ``tilefoundry::ops::mma`` call,
which picks the instruction from the operands' element type. Other
declarations need their own runtime mapping; in particular, WGMMA has no
emitter in this stage
([runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops)).
"""
from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.ir.tir import nn


@codegen.register(target.Target, codegen.Role.EMIT, nn.TiledMma)
def emit(call, ctx: context.Context) -> None:
    acc, lhs, rhs = call.args[0], call.args[1], call.args[2]
    if not isinstance(lhs, expr.Var) or not isinstance(rhs, expr.Var) or not isinstance(acc, expr.Var):
        raise RuntimeError(
            "tir.cuda.nn.TiledMma: codegen path expects Var operands on acc/lhs/rhs"
        )
    a = ctx.name_for(acc)
    l = ctx.name_for(lhs)
    r = ctx.name_for(rhs)
    atom = call.target.atom
    if not isinstance(atom, (nn.sm80_mma.Mma, nn.sm89_mma.Mma)):
        raise RuntimeError(
            f"tir.cuda.nn.TiledMma: no codegen handler for atom {expr.get_registered_name(atom)}"
        )

    ctx.emit(f"tilefoundry::ops::mma({l}, {r}, {a});")


__all__: list[str] = []
