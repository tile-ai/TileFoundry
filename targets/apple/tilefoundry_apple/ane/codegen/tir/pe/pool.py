"""Codegen for ``T.ane.pool``: one planar engine pooling pass.

A max pads with negative infinity. An average multiplies the window sum by the
reciprocal of the window size, or, with padding that ``count_include_pad`` leaves
out, by that of the cells the window holds inside the map.
"""

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic, rules

PATCH_EDGE = 4
KERNEL = (1, 3)
STRIDE = (1, 2)
PADDED_KERNEL = 3
NEGATIVE_INFINITY_PAD = 1


def patch_for(rows: int) -> tuple[int, int]:
    """The patch pair a pooling pass declares for a tile of *rows* result rows."""
    return PATCH_EDGE, rules.ceil_log2(rows)


def result_slots(rows: int, desc: isa.TensorDesc) -> int:
    """How many staged images the pooled result starts after.

    A pass that stages a whole plane writes right behind its source; a tile of
    fewer rows keeps the source and the result in separate staged images.
    """
    return 1 if rows == desc.height else 2


@codegen.register(target.Target, codegen.Role.EMIT, tir.Pool)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.width_code is not None:
        raise ValueError(f"T.ane.pool: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.pool: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.pool: omit_ocg={op.omit_ocg!r} is not decoded yet")
    destination, source = call.args
    if any(arg.type.storage is expr.StorageKind.SMEM for arg in call.args):
        raise ValueError("L2 operands for this operation is not decoded yet")
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if op.ceil_mode:
        raise ValueError("T.ane.pool: ceil_mode is not decoded yet")
    kernel, stride = tuple(op.kernel_size), tuple(op.stride)
    if len(kernel) != 2 or len(stride) != 2:
        raise ValueError(f"T.ane.pool: kernel {kernel} and stride {stride} must be 2D")
    for axis, values, (low, high) in (("kernel", kernel, KERNEL), ("stride", stride, STRIDE)):
        if any(not low <= value <= high for value in values):
            raise ValueError(f"T.ane.pool: {axis} {values} is outside [{low}, {high}]")
    padding = tuple(op.padding) if op.padding else (0, 0)
    if any(pad not in (0, 1) or pad and size != PADDED_KERNEL for pad, size in zip(padding, kernel)):
        raise ValueError(f"T.ane.pool: padding {padding} for kernel {kernel} is not decoded yet")
    if op.kind is expr.PoolKind.MAX:
        mode = isa.PoolMode.MAX
    elif op.kind is not expr.PoolKind.AVG:
        raise ValueError(f"T.ane.pool: {op.kind} has no planar engine encoding")
    elif any(padding) and not op.count_include_pad:
        mode = isa.PoolMode.NONE_OR_SUM
    else:
        mode = isa.PoolMode.AVG
    rows = context.tile_rows(op, dst)
    if dst.width > 1 << PATCH_EDGE:
        raise ValueError(f"T.ane.pool: a result {dst.width} wide is not decoded yet")
    patch = patch_for(rows) if op.patch is None else tuple(op.patch)
    if len(patch) != 2 or any(type(n) is not int or not 0 <= n <= 15 for n in patch):
        raise ValueError(f"patch {patch!r} must contain two integers within [0, 15]")
    if patch != patch_for(rows):
        raise ValueError(f"T.ane.pool: patch {patch} for this tile is not decoded yet")
    ctx.emit(
        intrinsic.load(src, address=ctx.address_of(source), fresh=ctx.fresh, width=op.tile_width),
        intrinsic.compute_pe(
            isa.TaskType.PE_POOL_UNARY,
            dst,
            pool_mode=mode,
            compare=NEGATIVE_INFINITY_PAD if mode is isa.PoolMode.MAX else None,
            declared_in=src,
            declared_out=dst,
            window=(kernel[1], kernel[0], stride[1], stride[0]),
            padding=(padding[1], padding[0]),
            tile_height=rows,
            patch=patch,
        ),
        intrinsic.store(
            dst,
            staging=src,
            width=op.tile_width,
            rows=context.tile_rows(op, src),
            staged=result_slots(rows, dst),
            address=ctx.address_of(destination),
            separated=False,
            fresh=ctx.fresh,
            declare=False,
        ),
    )


__all__: list[str] = []
