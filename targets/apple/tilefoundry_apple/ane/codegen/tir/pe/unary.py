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
    if op.width_code is not None:
        raise ValueError(f"T.ane.unary: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.unary: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.unary: omit_ocg={op.omit_ocg!r} is not decoded yet")
    if type(op.chained_source) is not bool:
        raise ValueError("T.ane.unary: chained_source must be a boolean")
    if op.chained_source:
        if ctx.fresh:
            raise ValueError("T.ane.unary: chained_source requires a preceding pass")
        raise ValueError("T.ane.unary: chained_source continuation mapping is not decoded yet")
    destination, source = call.args
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if (src.channels, src.height, src.width) != (dst.channels, dst.height, dst.width):
        raise ValueError("T.ane.unary: dst shape must equal src shape")
    rows = context.tile_rows(op, dst)
    ctx.emit(
        intrinsic.load(src, address=ctx.address_of(source), fresh=ctx.fresh, width=op.tile_width),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_UNARY,
            dst,
            pre_scale=op.pre_scale,
            scale=op.scale,
            bias=op.bias,
            absolute=op.abs,
            square=op.square,
            clamp=op.clamp,
            tile_height=rows,
            patch=context.raster_patch(op, rows, dst),
        ),
        intrinsic.store(dst, rows=rows, address=ctx.address_of(destination), fresh=ctx.fresh, width=op.tile_width),
    )


__all__: list[str] = []
