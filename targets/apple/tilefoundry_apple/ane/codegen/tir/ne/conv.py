"""Codegen for ``T.ane.conv``: one neural engine convolution over a coefficient stream."""

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context, hwx
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic

PLAIN = ((1, 1), 1)


@codegen.register(target.Target, codegen.Role.EMIT, tir.Conv)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.patch is not None:
        raise ValueError(f"T.ane.conv: patch={op.patch!r} is not decoded yet")
    if op.width_code is not None:
        raise ValueError(f"T.ane.conv: width_code={op.width_code!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.conv: omit_ocg={op.omit_ocg!r} is not decoded yet")
    if len(call.args) == 4:
        raise ValueError("T.ane.conv: bias operand encoding is not decoded yet")
    destination, source, weight = call.args
    if source.type.storage is expr.StorageKind.SMEM:
        raise ValueError("T.ane.conv: full-height L2 source small_source_mode mapping is not decoded yet")
    if destination.type.storage is expr.StorageKind.SMEM:
        raise ValueError("T.ane.conv: full-height L2 result pitch mapping is not decoded yet")
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if (tuple(op.dilation), op.groups) != PLAIN:
        raise ValueError("T.ane.conv: dilated or grouped convolution encoding is not decoded yet")
    rows = context.tile_rows(op, dst)
    if rows != dst.height or src.height != dst.height:
        raise ValueError("T.ane.conv: partial or unequal-height tile plan is not decoded yet")
    ocg_size = isa.rules.channel_groups(dst.channels)
    if op.ocg_size is not None:
        if type(op.ocg_size) is not int or op.ocg_size != ocg_size:
            raise ValueError(
                f"T.ane.conv: ocg_size {op.ocg_size!r} contradicts channels; expected {ocg_size}"
            )
        if dst.channels > 64:
            raise ValueError("T.ane.conv: explicit ocg_size above 64 channels is not decoded yet")
    layout = ctx.coefficient_use(weight)
    isa.dma.check_coefficient_lanes(
        layout.lanes, min(src.channels, len(isa.KernelDMA.kernel_dma_src_coeff_dma_en))
    )
    shape = layout.shape
    source_base, result_base = ctx.staging_address(1), ctx.staging_address(0)
    if source_base is None or result_base is None:
        raise ValueError("T.ane.conv: source and result require an allocated staging plan")
    ctx.emit(
        intrinsic.load(
            src,
            isa.Engine.NEURAL,
            address=ctx.address_of(source),
            fresh=ctx.fresh,
            rows=rows,
            width=op.tile_width,
            l2_base=source_base,
        ),
        intrinsic.convolution(
            src,
            dst,
            index=len(ctx.passes),
            coeff_base=layout.base,
            width=op.tile_width,
            ocg_size=ocg_size,
            tile_height=rows,
            read_config=False,
            coeff_stride=layout.stride,
            kernel_width=shape[3],
            kernel_height=shape[2],
            stride=op.stride,
            padding=op.padding,
            result_base=result_base,
            activate=op.nonlinear is not tir.Nonlinear.NONE,
            staged_source=False,
            to_dram=True,
            address=ctx.address_of(destination),
        ),
    )


def weight_writer(symbol: str, layout: hwx.CoefficientLayout) -> str:
    """Render one OIHW writer with validated CPU fp16 source geometry."""
    shape = ", ".join(map(str, layout.shape))
    return (
        f"static int {symbol}(uint8_t *coefficients, const DLTensor *tensor) {{\n"
        f"  static const int64_t shape[] = {{{shape}}};\n"
        "  if (!ane_check_tensor(tensor, shape, 16, true)) return ANE_BAD_ARGUMENT;\n"
        "  const auto *source = static_cast<const uint8_t *>(tensor->data) + tensor->byte_offset;\n"
        f"{layout.writer_body()}\n  return ANE_OK;\n}}\n"
    )


__all__ = ["weight_writer"]
