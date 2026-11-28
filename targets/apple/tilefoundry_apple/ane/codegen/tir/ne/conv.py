"""Codegen for ``T.ane.conv``: one neural engine convolution over a coefficient stream."""

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context, hwx
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic

DEPTHWISE_CHANNELS = 32
DEPTHWISE_KERNEL = (3, 3)
NONLINEAR = MappingProxyType(
    {
        tir.Nonlinear.NONE: isa.NENonlinear.NONE,
        tir.Nonlinear.RELU: isa.NENonlinear.RELU,
        tir.Nonlinear.SIGMOID: isa.NENonlinear.LUT,
        tir.Nonlinear.SILU: isa.NENonlinear.LUT,
    }
)


@codegen.register(target.Target, codegen.Role.EMIT, tir.Conv)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.patch is not None:
        raise ValueError(f"T.ane.conv: patch={op.patch!r} is not decoded yet")
    if op.width_code is not None:
        raise ValueError(f"T.ane.conv: width_code={op.width_code!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.conv: omit_ocg={op.omit_ocg!r} is not decoded yet")
    source, weight, destination, *held = call.args
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst")
    if tuple(op.dilation) != (1, 1):
        raise ValueError("T.ane.conv: dilated convolution encoding is not decoded yet")
    if tuple(op.stride) != (1, 1):
        raise ValueError("T.ane.conv: strided convolution encoding is not decoded yet")
    depthwise = op.groups > 1
    if depthwise and (
        op.groups != src.channels
        or src.channels != dst.channels
        or src.channels != DEPTHWISE_CHANNELS
        or tuple(weight.type.shape[2:]) != DEPTHWISE_KERNEL
    ):
        raise ValueError(
            f"T.ane.conv: grouped convolution other than a depthwise {DEPTHWISE_KERNEL} kernel "
            f"over {DEPTHWISE_CHANNELS} channels is not decoded yet"
        )
    if src.height != dst.height:
        raise ValueError("T.ane.conv: unequal-height convolution is not decoded yet")
    ocg_size = 0 if depthwise else isa.rules.channel_groups(dst.channels)
    if op.ocg_size is not None:
        if type(op.ocg_size) is not int or op.ocg_size != ocg_size:
            raise ValueError(
                f"T.ane.conv: ocg_size {op.ocg_size!r} contradicts channels; expected {ocg_size}"
            )
        if dst.channels > 64:
            raise ValueError("T.ane.conv: explicit ocg_size above 64 channels is not decoded yet")
    layout = ctx.coefficient_use(
        weight, tir.ne.nonlinear.TABLES.get(op.nonlinear, b""), held[0] if held else None, depthwise
    )
    isa.dma.check_coefficient_lanes(
        layout.lanes, min(src.channels, len(isa.KernelDMA.kernel_dma_src_coeff_dma_en))
    )
    shape = layout.shape
    base, producer = ctx.source(source)
    result_base, stored = ctx.result(destination)
    ctx.emit(
        intrinsic.load_chained(src, isa.Engine.NEURAL, producer=producer, address=base),
        intrinsic.convolution(
            src,
            context.written(dst, stored),
            index=len(ctx.passes),
            coeff_base=layout.base,
            ocg_size=ocg_size,
            read_config=False,
            coeff_stride=layout.stride,
            kernel_width=shape[3],
            kernel_height=shape[2],
            stride=op.stride,
            padding=op.padding,
            result_base=result_base,
            nonlinear=NONLINEAR[op.nonlinear],
            bias=layout.bias,
            depthwise=depthwise,
            staged_source=False,
            to_dram=stored is not None,
            address=ctx.address_of(stored),
        ),
    )


def weight_writer(symbol: str, layout: hwx.CoefficientLayout, bias: bool = False) -> str:
    """Render the OIHW weight writer, or its bias writer, with validated CPU fp16 geometry."""
    dims = (layout.shape[0],) if bias else layout.shape
    shape = ", ".join(map(str, dims))
    return (
        f"static int {symbol}(uint8_t *coefficients, const DLTensor *tensor) {{\n"
        f"  static const int64_t shape[] = {{{shape}}};\n"
        f"  if (!ane_check_tensor(tensor, shape, {len(dims)}, 16, true)) return ANE_BAD_ARGUMENT;\n"
        "  const auto *source = static_cast<const uint8_t *>(tensor->data) + tensor->byte_offset;\n"
        f"{layout.writer_body(bias)}\n  return ANE_OK;\n}}\n"
    )


__all__ = ["weight_writer"]
