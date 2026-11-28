"""Codegen for ``T.ane.unary``: one planar engine affine pass with its clamp."""

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic


@codegen.register(target.Target, codegen.Role.EMIT, tir.Unary)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.abs and op.square and op.clamp is not None:
        raise ValueError("T.ane.unary: abs with square and clamp is not decoded yet")
    if op.square and op.scale != 1.0:
        raise ValueError("T.ane.unary: square requires scale=1; hardware square mode ignores scale")
    if op.width_code is not None:
        raise ValueError(f"T.ane.unary: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.unary: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.unary: omit_ocg={op.omit_ocg!r} is not decoded yet")
    source, destination = call.args
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if (src.channels, src.height, src.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.unary: dst shape must equal src shape")
    base, producer = ctx.source(source)
    result_base, stored = ctx.result(destination)
    ctx.emit(
        intrinsic.load_chained(src, producer=producer, address=base),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_UNARY,
            dst,
            pre_scale=op.pre_scale,
            scale=op.scale,
            bias=op.bias,
            absolute=op.abs,
            square=op.square,
            clamp=op.clamp,
            patch=context.raster_patch(dst, op.patch),
        ),
        intrinsic.store(
            context.written(dst, stored), result_base=result_base, address=ctx.address_of(stored),
            fresh=ctx.fresh, producer=producer, to_dram=stored is not None,
        ),
    )


__all__: list[str] = []
