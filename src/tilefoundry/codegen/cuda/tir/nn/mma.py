"""Emit effect-form matrix multiply-accumulate operations.

The SM80 declaration emits one ``tilefoundry::ops::mma`` call. Other
declarations need their own runtime mapping; in particular, WGMMA has no
emitter in this stage ([runtime §2.6](docs/spec/runtime.md#26-cudaops)).
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma as Sm80Mma
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


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
    if not isinstance(atom, Sm80Mma):
        raise RuntimeError(
            f"tir.cuda.nn.TiledMma: no codegen handler for atom {atom.reference_name}"
        )

    ctx.emit(f"tilefoundry::ops::mma({l}, {r}, {a});")
