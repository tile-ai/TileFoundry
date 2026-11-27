"""Codegen for ``T.ane.pool``: one planar engine pooling pass."""

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic

MODES = MappingProxyType(
    {expr.PoolKind.MAX: isa.PoolMode.MAX, expr.PoolKind.AVG: isa.PoolMode.AVG}
)
PATCH_EDGE = 4
PATCH_SWEEP = 5
WINDOW = (1, 2)


def patch_for(kernel: tuple[int, int], stride: tuple[int, int]) -> tuple[int, int]:
    """The patch pair a pooling pass declares for its window."""
    return PATCH_EDGE + (1 if kernel[1] > 1 else 0), PATCH_SWEEP - stride[0]


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
    for axis, value in zip(("kernel", "stride"), (*kernel, *stride)):
        if not WINDOW[0] <= value <= WINDOW[1]:
            raise ValueError(f"T.ane.pool: {axis} {value} is outside [1, 2]")
    padding = tuple(op.padding) if op.padding else (0, 0)
    if any(padding):
        raise ValueError("T.ane.pool: padding is not decoded yet")
    if op.kind not in MODES:
        raise ValueError(f"T.ane.pool: {op.kind} has no planar engine encoding")
    mode = MODES[op.kind]
    rows = context.tile_rows(op, dst)
    patch = patch_for(kernel, stride) if op.patch is None else tuple(op.patch)
    if len(patch) != 2 or any(type(n) is not int or not 0 <= n <= 15 for n in patch):
        raise ValueError(f"patch {patch!r} must contain two integers within [0, 15]")
    if patch != patch_for(kernel, stride):
        raise ValueError(f"T.ane.pool: patch {patch} for this window is not decoded yet")
    ctx.emit(
        intrinsic.load(src, address=ctx.address_of(source), fresh=ctx.fresh, width=op.tile_width),
        intrinsic.compute_pe(
            isa.TaskType.PE_POOL_UNARY,
            dst,
            pool_mode=mode,
            compare=1 if mode is isa.PoolMode.MAX else None,
            declared_in=src,
            declared_out=dst,
            window=(kernel[1], kernel[0], stride[1], stride[0]),
            padding=padding,
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
