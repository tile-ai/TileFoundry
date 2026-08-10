"""Emit effect-form matrix multiply-accumulate operations.

The current runtime entry supports the SM80 16x8x16 BF16 atom and packs each
lane's coalesced fragments for the corresponding PTX MMA instruction. Other
architecture, dtype, and shape combinations require an explicit runtime mapping.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.cuda.nn.mma import Mma

_MMA_RUNTIME = {
    "SM80_16x8x16_F32BF16BF16F32_TN": "tilefoundry::ops::mma_sm80_16x8x16_bf16",
}


@register_codegen_cuda(Mma)
def _emit(call, ctx: CodegenContext) -> None:
    acc, lhs, rhs = call.args[0], call.args[1], call.args[2]
    if not isinstance(lhs, Var) or not isinstance(rhs, Var) or not isinstance(acc, Var):
        raise RuntimeError(
            "tir.cuda.nn.Mma: codegen path expects Var operands on acc/lhs/rhs"
        )
    a = ctx.name_for(acc)
    l = ctx.name_for(lhs)
    r = ctx.name_for(rhs)


    atom = call.target.atom
    if atom is None:
        runtime = _MMA_RUNTIME["SM80_16x8x16_F32BF16BF16F32_TN"]
    else:
        runtime = _MMA_RUNTIME.get(atom.op.name)
        if runtime is None:
            raise RuntimeError(
                f"tir.cuda.nn.Mma: no codegen handler for MMA op {atom.op.name!r}; "
                f"add an entry to _MMA_RUNTIME"
            )

    ctx.emit(f"{runtime}({l}, {r}, {a});")
