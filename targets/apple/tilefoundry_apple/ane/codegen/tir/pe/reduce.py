"""Codegen for ``T.ane.reduce``: one planar engine reduction over the spatial axes.

``MODES`` gives each reduction kind its L2 compare code and its pooling mode. A
sum's ``scale`` is the engine's fp32 final scale, so a mean is a sum scaled by the
reciprocal of its count.
"""

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic

MODES = MappingProxyType(
    {
        expr.ReduceKind.SUM: (None, isa.PoolMode.NONE_OR_SUM),
        expr.ReduceKind.MAX: (1, isa.PoolMode.MAX),
        expr.ReduceKind.MIN: (3, isa.PoolMode.MIN),
    }
)
SPATIAL = (2, 3)
RESULT_SLOTS = 1


@codegen.register(target.Target, codegen.Role.EMIT, tir.Reduce)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.width_code is not None:
        raise ValueError(f"T.ane.reduce: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.reduce: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.reduce: omit_ocg={op.omit_ocg!r} is not decoded yet")
    destination, source = call.args
    if any(arg.type.storage is expr.StorageKind.SMEM for arg in call.args):
        raise ValueError("L2 operands for this operation is not decoded yet")
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if tuple(op.axes) != SPATIAL or not op.keepdim:
        raise ValueError(f"T.ane.reduce: axes {tuple(op.axes)} is not decoded yet")
    if (dst.channels, dst.height, dst.width) != (src.channels, 1, 1):
        raise ValueError(
            f"T.ane.reduce: expected a {src.channels}x1x1 result, got "
            f"{dst.channels}x{dst.height}x{dst.width}"
        )
    if op.square or op.post is not tir.Post.NONE:
        raise ValueError("T.ane.reduce: square and post is not decoded yet")
    if op.kind not in MODES:
        raise ValueError(f"T.ane.reduce: {op.kind} has no planar engine encoding")
    if op.scale != 1.0 and op.kind is not expr.ReduceKind.SUM:
        raise ValueError(f"T.ane.reduce: scale with {op.kind} is not decoded yet")
    compare, mode = MODES[op.kind]
    rows = context.tile_rows(op, src)
    ctx.emit(
        intrinsic.load(
            src, address=ctx.address_of(source), fresh=ctx.fresh,
            width=op.tile_width, producer=isa.Engine.PLANAR
        ),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_REDUCE_UNARY,
            dst,
            declared_in=src,
            declared_out=src,
            tile_height=rows,
            pool_mode=mode,
            compare=compare,
            final_scale=op.scale,
            patch=context.raster_patch(op, context.tile_rows(op, dst), dst),
        ),
        intrinsic.store(
            dst,
            staging=src,
            width=op.tile_width,
            rows=rows,
            staged=RESULT_SLOTS,
            address=ctx.address_of(destination),
            fresh=ctx.fresh,
            declare=False,
        ),
    )


__all__: list[str] = []
