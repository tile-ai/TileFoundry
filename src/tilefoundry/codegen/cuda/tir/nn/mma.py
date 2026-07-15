"""Emitter for ``tir.cuda.nn.Mma`` (effect-form matrix-multiply-accumulate).

The handler emits a single call to ``tilefoundry::ops::mma_sm80_16x8x16_bf16``
defined in ``include/tilefoundry/runtime/cuda/runtime.cuh``. That runtime entry packs
each lane's coalesced cute fragment registers in the order required by
the PTX ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``
instruction (cf. ``cute/atom/mma_traits_sm80.hpp`` ALayout / BLayout /
CLayout).

A more general dispatch over ``Mma`` arch / dtype / shape combinations
is a follow-up; the SM80 BF16 atom is the currently supported form.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.cuda.nn.mma import Mma

# MmaOpSpec.name → runtime entry. One handler today; a new instruction adds a
# row here, no change to the Mma op / T.mma surface.
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
    # ``atom`` is implicit (None) on the hir_to_tir lowered path → the SM80 BF16
    # atom (current sole arch). When present, dispatch on its op name.
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
    # Runtime entry takes (lhs, rhs, acc) and accumulates lhs@rhs into acc.
    ctx.emit(f"{runtime}({l}, {r}, {a});")
