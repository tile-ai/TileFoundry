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


@codegen.register(target.Target, codegen.Role.EMIT, tir.Reduce)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.width_code is not None:
        raise ValueError(f"T.ane.reduce: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.reduce: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.reduce: omit_ocg={op.omit_ocg!r} is not decoded yet")
    source, destination = call.args
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if tuple(op.axes) != SPATIAL or not op.keepdim:
        raise ValueError(f"T.ane.reduce: axes {tuple(op.axes)} is not decoded yet")
    if (dst.channels, dst.height, dst.width) != (src.channels, 1, 1):
        raise ValueError(
            f"T.ane.reduce: expected a {src.channels}x1x1 result, got "
            f"{dst.channels}x{dst.height}x{dst.width}"
        )
    if op.post is not tir.Post.NONE:
        raise ValueError("T.ane.reduce: post is not decoded yet")
    if op.kind not in MODES:
        raise ValueError(f"T.ane.reduce: {op.kind} has no planar engine encoding")
    if op.square and op.kind is not expr.ReduceKind.SUM:
        raise ValueError(f"T.ane.reduce: square with {op.kind} is not decoded yet")
    if op.scale != 1.0 and op.kind is not expr.ReduceKind.SUM:
        raise ValueError(f"T.ane.reduce: scale with {op.kind} is not decoded yet")
    compare, mode = MODES[op.kind]
    base, producer = ctx.source(source)
    result_base, stored = ctx.result(destination)
    ctx.emit(
        intrinsic.load_chained(src, producer=producer, address=base),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_REDUCE_UNARY,
            dst,
            declared_in=src,
            declared_out=src,
            tile_height=src.height,
            pool_mode=mode,
            square=op.square,
            compare=compare,
            final_scale=op.scale,
            patch=context.raster_patch(dst, op.patch),
        ),
        intrinsic.store(
            context.written(dst, stored), result_base=result_base, address=ctx.address_of(stored),
            fresh=ctx.fresh, producer=producer, to_dram=stored is not None, declare=False,
        ),
    )


__all__: list[str] = []
