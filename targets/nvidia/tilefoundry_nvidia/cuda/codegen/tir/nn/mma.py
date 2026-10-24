"""Emit effect-form matrix multiply-accumulate operations.

The SM80 and SM89 declarations each emit one ``tilefoundry::ops::mma`` call,
which picks the instruction from the operands' element type. Other
declarations need their own runtime mapping; in particular, WGMMA has no
emitter in this stage
([runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops)).
"""
from __future__ import annotations

from tilefoundry.ir.core import Var
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma
from tilefoundry_nvidia.cuda.ir.tir.nn.sm80_mma import Mma as Sm80Mma
from tilefoundry_nvidia.cuda.ir.tir.nn.sm89_mma import Mma as Sm89Mma


@register_codegen(CudaTarget, Role.EMIT, TiledMma)
def _emit(call, ctx: CudaCodegenContext) -> None:
    acc, lhs, rhs = call.args[0], call.args[1], call.args[2]
    if not isinstance(lhs, Var) or not isinstance(rhs, Var) or not isinstance(acc, Var):
        raise RuntimeError(
            "tir.cuda.nn.TiledMma: codegen path expects Var operands on acc/lhs/rhs"
        )
    a = ctx.name_for(acc)
    l = ctx.name_for(lhs)
    r = ctx.name_for(rhs)
    atom = call.target.atom
    if not isinstance(atom, (Sm80Mma, Sm89Mma)):
        raise RuntimeError(
            f"tir.cuda.nn.TiledMma: no codegen handler for atom {atom.reference_name}"
        )

    ctx.emit(f"tilefoundry::ops::mma({l}, {r}, {a});")
