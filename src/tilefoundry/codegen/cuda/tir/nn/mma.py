"""Emit effect-form matrix multiply-accumulate operations.

One call, ``tilefoundry::ops::mma``, whichever tier the operand layouts pick:
a lane's gathered fragments take the single PTX instruction and a rank-2 tile
loops the atom over it. The table below is what an atom name is allowed to
emit, not a tier -- other architecture, dtype and shape combinations need their
own runtime mapping ([runtime §2.6](docs/spec/runtime.md#26-cudaops)).
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.cuda.nn.mma import Mma
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, Mma)
def _emit(call, ctx: CudaCodegenContext) -> None:
    acc, lhs, rhs = call.args[0], call.args[1], call.args[2]
    if not isinstance(lhs, Var) or not isinstance(rhs, Var) or not isinstance(acc, Var):
        raise RuntimeError(
            "tir.cuda.nn.Mma: codegen path expects Var operands on acc/lhs/rhs"
        )
    a = ctx.name_for(acc)
    l = ctx.name_for(lhs)
    r = ctx.name_for(rhs)


    atom = call.target.atom
    if atom is not None and atom.op.name != "SM80_16x8x16_F32BF16BF16F32_TN":
        raise RuntimeError(
            f"tir.cuda.nn.Mma: no codegen handler for MMA op {atom.op.name!r}"
        )

    ctx.emit(f"tilefoundry::ops::mma({l}, {r}, {a});")
