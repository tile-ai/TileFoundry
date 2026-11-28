"""Codegen for ``T.ane.load``: one identity planar task that reads its source by DMA.

The DMA lands in the destination's L2 block and the task keeps its result there.
"""

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic


@codegen.register(target.Target, codegen.Role.EMIT, tir.Load)
def emit(call: expr.Call, ctx: context.Context) -> None:
    source, destination = call.args
    src = context.tensor_desc_of(source.type, "src", strided=True, view=destination.type.shape)
    dst = context.tensor_desc_of(destination.type, "dst")
    base, _ = ctx.result(destination)
    ctx.emit(
        intrinsic.load(src, address=ctx.address_of(source), fresh=ctx.fresh, l2_base=base),
        intrinsic.compute_pe(isa.TaskType.PE_EW_UNARY, dst, patch=context.raster_patch(dst)),
        intrinsic.store(dst, result_base=base, to_dram=False),
    )


__all__: list[str] = []
