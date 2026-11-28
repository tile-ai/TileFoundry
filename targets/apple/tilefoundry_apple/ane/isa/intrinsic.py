"""Encode feature-map transfers and engine operations as configuration register maps."""

import struct
from collections.abc import Mapping

from . import fields
from .arith import align_up
from .tensor_desc import NE_CHANNEL_GROUP, SURFACE_ROW_ALIGN, Engine, TensorDesc

FP16_BYTES = 2
L2_UNIT = 16
STREAM_TAG = 1
DEST_TAG = 1
DATA_SET_ID = 14
PLANAR_MIN_ROW_BYTES = 32
NEURAL_MIN_ROW_BYTES = 16

NE_WIDTH_UNIT = 256
NE_BINARY_POINT = 28
NE_OUTPUT_SELECT = 1

SOURCE_HINT = 0x21
SECOND_SOURCE_HINT = 0x20
DESCRIPTOR_MARKER = 0xDEAD
SOURCE_CACHE_HINT = 0xCE
DEST_HINT = 0x3
F16_TRUNC = 3
INTERLEAVE = 1
FIFO_MODE = 1
KERNEL_ARMED = 1
COMMON_FORMAT_F16 = 2


L2 = fields.L2()
PE = fields.PE()
NE = fields.NE()
COMMON = fields.Common()
SRC_DMA = fields.TileDMASrc()
DST_DMA = fields.TileDMADst()
KERNEL_DMA = fields.KernelDMA()

def fp16(value: float, name: str) -> int:
    """Encode one immediate as an fp16 bit pattern."""
    try:
        return int.from_bytes(struct.pack("<e", value), "little")
    except OverflowError as error:
        raise ValueError(f"{name} {value} does not fit fp16") from error


def fp32(value: float, name: str) -> int:
    """Encode one immediate as an fp32 bit pattern."""
    try:
        return int.from_bytes(struct.pack("<f", value), "little")
    except OverflowError as error:
        raise ValueError(f"{name} {value} does not fit fp32") from error


def splice(*parts: Mapping[int, int]) -> fields.Writes:
    """Combine primitive fields, rejecting different values for overlapping writes."""
    image = fields.Writes()
    for part in parts:
        image.merge(part)
    return image


def require_row_width(desc: TensorDesc, minimum: int) -> None:
    """Reject a feature map whose dense row selects an undecoded format entry."""
    desc.validate_feature()
    dense = desc.width * FP16_BYTES
    if dense < minimum:
        raise ValueError(
            f"feature map {desc.channels}x{desc.height}x{desc.width} has a {dense} "
            f"byte dense row; the format entry for rows below {minimum} bytes is not decoded yet"
        )


def row_stride(name: str, byte_stride: int) -> dict[str, int]:
    """The row-stride field, omitted for a feature map whose extent the hardware ignores."""
    return {} if byte_stride == 0 else {name: dram_units(byte_stride)}


def dram_units(byte_stride: int) -> int:
    """Express a DRAM stride in the unit its field holds."""
    if byte_stride % SURFACE_ROW_ALIGN:
        raise ValueError(f"stride {byte_stride} is not a multiple of {SURFACE_ROW_ALIGN}")
    return byte_stride // SURFACE_ROW_ALIGN


def l2_units(byte_stride: int) -> int:
    """Express an L2 stride in the unit its field holds."""
    if byte_stride % L2_UNIT:
        raise ValueError(f"stride {byte_stride} is not a multiple of {L2_UNIT}")
    return byte_stride // L2_UNIT


def split_address(address: int) -> tuple[int, int]:
    """Split a byte address into the low and high words the DMA programs."""
    if not 0 <= address < (1 << 64):
        raise ValueError(f"address {address:#x} does not fit 64 bits")
    return address & 0xFFFFFFFF, address >> 32


def staged_flags(fresh: bool, producer: Engine | None = None) -> dict[str, int]:
    """Return the buffer-reuse bits a descriptor programs for its staged slots.

    A descriptor that stages into fresh L2 buffers marks each slot as first use;
    a descriptor that continues a stream reuses the buffers its predecessor left
    behind, so it clears those bits and tags its DMA words with the stream index.
    A source read from the result of ``producer`` keeps that engine's result bit set.
    """
    alias = 1 if fresh else 0
    return {
        "l2_src_alias_conv_src": alias,
        "l2_src_alias_conv_rslt": 1 if producer is Engine.NEURAL else alias,
        "l2_src_alias_planar_src": alias,
        "l2_src_alias_planar_rslt": 1 if producer is Engine.PLANAR else alias,
        "l2_result_alias_src": alias,
        "l2_result_alias_rslt": alias,
        "l2_result_alias_planar_src": alias,
        "l2_result_alias_planar_rslt": alias,
    }


def src_tags(fresh: bool) -> dict[str, int]:
    """Return the stream tag a descriptor stamps on its first source transfer."""
    return {"tile_dma_src_user_tag": 0 if fresh else STREAM_TAG}


def dst_tags(fresh: bool) -> dict[str, int]:
    """Return the stream tag a descriptor stamps on its result transfer."""
    return {"tile_dma_dst_user_tag": 0 if fresh else STREAM_TAG}


def load(source: TensorDesc, *, address: int = 0, fresh: bool = True, l2_base: int) -> dict[int, int]:
    """Program reading one feature map by DMA into the planar engine's L2 buffer at *l2_base*."""
    require_row_width(source, PLANAR_MIN_ROW_BYTES)
    low, high = split_address(address)
    return splice(
        COMMON.pack(
            cin=source.channels, win=source.width, hin=source.height, din=1,
            common_in_fmt=COMMON_FORMAT_F16, common_src2_in_fmt=COMMON_FORMAT_F16,
        ),
        L2.pack(
            l2_src_type=fields.L2Access.DMA_READ,
            l2_src_base_address=l2_units(l2_base),
            **staged_flags(fresh),
            l2_src_dma_format=fields.TensorFormatMode.COMPRESSED,
            l2_src_interleave=INTERLEAVE,
            l2_src_channel_stride=l2_units(source.l2_channel),
            l2_src_row_stride=l2_units(source.l2_row),
            l2_src_depth_stride=l2_units(source.l2_row),
            l2_src_group_stride=l2_units(source.l2_row),
        ),
        SRC_DMA.pack(
            **src_tags(fresh),
            tile_dma_src_hint=SOURCE_HINT,
            tile_dma_src_base_addr_lo=low,
            tile_dma_src_base_addr_hi=high,
            tile_dma_src_data_set_id=DATA_SET_ID,
            tile_dma_src_cache_hint=SOURCE_CACHE_HINT,
            **row_stride("tile_dma_src_row_stride", source.dram_row_stride),
            tile_dma_src_channel_stride=dram_units(source.channel),
            tile_dma_src_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_src_trunc=F16_TRUNC,
            tile_dma_src_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_src_interleave=INTERLEAVE,
        ),
    )


def load_chained(
    source: TensorDesc, engine: Engine = Engine.PLANAR, *, producer: Engine, address: int = 0
) -> dict[int, int]:
    """Read a source the previous descriptor left staged instead of reloading it.

    A chained pass reads the L2 buffer its predecessor filled, so it programs no
    source DMA at all and leaves the access type at L2 read.  The reference
    streams for an add-then-pool and an abs-then-add chain both show a chained
    descriptor carrying no source DMA block, which is what this encodes.
    ``producer`` is the engine whose pass filled the buffer.
    """
    require_row_width(source, PLANAR_MIN_ROW_BYTES)


    window = dict(win=source.width, hin=source.height, din=1)
    if engine is Engine.PLANAR:
        window.update(common_in_fmt=COMMON_FORMAT_F16, common_src2_in_fmt=COMMON_FORMAT_F16)
    return splice(
        COMMON.pack(cin=source.channels, **window),
        L2.pack(
            l2_src_type=fields.L2Access.L2_READ,
            l2_src_base_address=l2_units(address),
            **staged_flags(False, producer),
            l2_src_interleave=INTERLEAVE,
            l2_src_channel_stride=l2_units(source.l2_channel),
            l2_src_row_stride=l2_units(source.l2_row),
        ),
    )


def load_second_chained(source: TensorDesc, *, address: int) -> dict[int, int]:
    """Read the second source from the L2 buffer an earlier descriptor filled.

    Like ``load_chained`` it programs no DMA; the second stream stays off and only
    the buffer base and its channel and row pitches are set.
    """
    require_row_width(source, PLANAR_MIN_ROW_BYTES)
    return L2.pack(
        l2_src2_base_address=l2_units(address),
        l2_src2_channel_stride=l2_units(source.l2_channel),
        l2_src2_row_stride=l2_units(source.l2_row),
    )


def store(
    destination: TensorDesc,
    *,
    result_base: int,
    address: int = 0,
    fresh: bool = True,
    declare: bool = True,
    declared: TensorDesc | None = None,
    producer: Engine | None = None,
    to_dram: bool = True,
) -> dict[int, int]:
    """Program writing one feature map: the L2 result buffer at *result_base* and the result DMA."""
    destination.validate_feature()
    if declared is not None:
        declared.validate_feature()
    low, high = split_address(address)
    return splice(
        COMMON.pack(
            **(
                {}
                if not declare
                else {
                    "wout": (destination if declared is None else declared).width,
                    "hout": (destination if declared is None else declared).height,
                    "dout": 1,
                    "cout": (destination if declared is None else declared).channels,
                }
            ),
            common_out_fmt=COMMON_FORMAT_F16,
        ),
        L2.pack(
            l2_result_type=fields.L2Access.DMA_READ if to_dram else fields.L2Access.L2_READ,
            **({"l2_result_buffer_mode": 1} if to_dram else {}),
            **staged_flags(fresh, producer),
            l2_result_dma_fmt=fields.TensorFormatMode.COMPRESSED,
            l2_result_interleave=INTERLEAVE,
            **(
                {} if to_dram else {
                    "l2_result_channel_stride": l2_units(destination.l2_channel),
                    "l2_result_row_stride": l2_units(destination.l2_row),
                }
            ),
            l2_result_address=l2_units(result_base),
        ),
        {} if not to_dram else DST_DMA.pack(
            **dst_tags(fresh),
            tile_dma_dst_enable=1,
            tile_dma_dst_base_addr_lo=low,
            tile_dma_dst_base_addr_hi=high,
            tile_dma_dst_hint=DEST_HINT,
            tile_dma_dst_data_set_id=DATA_SET_ID,
            tile_dma_dst_fifo_mode=FIFO_MODE,


            **row_stride("tile_dma_dst_row_stride", destination.dram_row_stride),
            tile_dma_dst_channel_stride=dram_units(destination.channel),
            tile_dma_dst_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_dst_trunc=F16_TRUNC,
            tile_dma_dst_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_dst_zero_pad_first=1,
            tile_dma_dst_zero_pad_last=1,
            tile_dma_dst_interleave=INTERLEAVE,
        ),
    )


KERNEL_DMA_COEFF_SLOTS = 16
KERNEL_DMA_MIN_BLOCK = 64


def coefficient_dma(
    outputs: int,
    inputs: int,
    *,
    index: int = 0,
    armed: bool = True,
    address: int = 0,
    stride: int | None = None,
) -> dict[int, int]:
    """Program the coefficient stream of one convolution.

    Every input column owns a coefficient block of ``outputs`` fp16 values, so the
    column offsets step by that block and the per-column length is the same value.
    A lookup-table activation leads every lane with its table, so its ``stride``
    covers the table and the weights.
    """
    if not 1 <= inputs:
        raise ValueError(f"coefficient columns {inputs} must be positive")
    columns = max(1, -(-inputs // KERNEL_DMA_COEFF_SLOTS))
    payload = columns * outputs * FP16_BYTES
    block = max(
        KERNEL_DMA_MIN_BLOCK,
        -(-payload // KERNEL_DMA_MIN_BLOCK) * KERNEL_DMA_MIN_BLOCK,
    )
    if stride is not None:

        block = stride
    image = KERNEL_DMA.pack(
        kernel_dma_src_enable=1,
        kernel_dma_armed=int(armed),
        kernel_dma_coeff_present=1,
        kernel_dma_coeff_tag=index + 1,
    )
    for slot in range(KERNEL_DMA_COEFF_SLOTS):
        active = slot < inputs
        for group, value in (
            (KERNEL_DMA.kernel_dma_src_coeff_dma_en, int(active)),
            (KERNEL_DMA.kernel_dma_src_coeff_loaded, int(active)),
            (KERNEL_DMA.kernel_dma_src_coeff_tag, index if active else 0),
            (KERNEL_DMA.kernel_dma_src_coeff_base_offset, dram_units(slot * block)),
            (KERNEL_DMA.kernel_dma_src_coeff_mem_buffer_size, dram_units(block)),
        ):
            field = group[slot]
            if value == 0 and field.lo != 16:
                continue
            if (
                group is KERNEL_DMA.kernel_dma_src_coeff_mem_buffer_size
                and block == KERNEL_DMA_MIN_BLOCK
            ):
                continue
            image.put(field, value)
    return image


def convolution(
    source: TensorDesc,
    destination: TensorDesc,
    *,
    index: int,
    ocg_size: int,
    coeff_stride: int | None = None,
    kernel_width: int = 1,
    kernel_height: int = 1,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    active_nodes: int = 4,
    multi_tile: bool = False,
    coeff_base: int = 0,
    nonlinear: fields.NENonlinear = fields.NENonlinear.NONE,
    bias: bool = False,
    armed: bool | None = None,
    staged_source: bool = False,
    to_dram: bool = True,
    source_base: int = 0,
    result_base: int | None = None,
    read_config: bool = True,
    address: int = 0,
) -> dict[int, int]:
    """Program one convolution pass over a staged source with a coefficient stream.

    The pass reads the source the previous descriptor staged, so no source DMA
    and no source relocation are programmed.  Its result lands one channel slab
    after the staged source. With ``bias`` every coefficient lane leads its weights
    with one fp16 bias word per lane output.
    """
    source.validate_feature()
    destination.validate_feature()
    low, high = split_address(address)
    if to_dram and result_base is None:
        raise ValueError("convolution result base must be stated; it may not overlap the staged source")
    return splice(
        {fields.KernelDMA.kernel_dma_src_base_lo.addr: coeff_base & 0xFFFFFFFF,
         fields.KernelDMA.kernel_dma_src_base_hi.addr: coeff_base >> 32},


        COMMON.pack(
            cin=source.channels,
            cout=destination.channels,
            **(
                {"win": 1, "hin": 1, "din": 1, "wout": 1, "hout": 1, "dout": 1, "tile_height": 1}
                if staged_source
                else {
                    "wout": destination.width,
                    "hout": destination.height,
                    "dout": 1,
                    "tile_height": destination.height,
                    **(
                        {"shift_preferred": 3, "shift_minimum": 3, "shift_maximum": 3}
                        if multi_tile
                        else {"shift_maximum": 3}
                    ),
                }
            ),
            **({} if destination.channels > NE_CHANNEL_GROUP * 4 else {"ne_ocg_size": ocg_size}),
            source_routing=1,
            task_type=fields.TaskType.NE,
            active_nodes=active_nodes,
            node_id=1,
            **({"output_transpose": 1, "small_source_mode": 1} if staged_source else {}),
        ),
        {}
        if staged_source
        else COMMON.pack(
            kw=kernel_width,
            kh=kernel_height,
            sx=stride[1],
            sy=stride[0],
            pad_left=padding[1],
            pad_top=padding[0],
            out_offset_x=1,
            out_offset_y=1,
        ),
        L2.pack(
            **(
                {}
                if not read_config
                else {
                    "l2_src_type": fields.L2Access.L2_READ,
                    **(
                        {"l2_src_dependent": 2, "l2_src_interleave": INTERLEAVE}
                        if staged_source
                        else {}
                    ),
                }
            ),
            **({} if not staged_source else {"l2_src_base_address": l2_units(source_base)}),
            l2_src_channel_stride=l2_units(source.l2_channel),
            l2_result_type=(fields.L2Access.DMA_READ if to_dram else fields.L2Access.L2_READ),
            **({} if not to_dram else {"l2_result_buffer_mode": 1}),
            l2_result_dma_fmt=fields.TensorFormatMode.COMPRESSED,
            l2_result_interleave=INTERLEAVE,
            **(
                {} if to_dram else {
                    "l2_result_channel_stride": l2_units(destination.l2_channel),
                    "l2_result_row_stride": l2_units(destination.l2_row),
                }
            ),
            **(
                {}
                if result_base is None and not to_dram
                else {
                    "l2_result_address": l2_units(
                        result_base
                    )
                }
            ),
        ),
        NE.pack(
            op=fields.NEOp.CONV,
            ne_bias=int(bias),
            ne_binary_point=NE_BINARY_POINT,
            arg_output_select=NE_OUTPUT_SELECT,
            post_scale=fp16(1.0, "post_scale"),
            nonlinear_mode=nonlinear,
        ),
        coefficient_dma(
            destination.channels,
            source.channels,
            index=index,
            armed=armed if armed is not None else to_dram,
            stride=coeff_stride,
        ),
        {}
        if not to_dram
        else DST_DMA.pack(
            tile_dma_dst_enable=1,
            tile_dma_dst_base_addr_lo=low,
            tile_dma_dst_base_addr_hi=high,
            tile_dma_dst_hint=DEST_HINT,
            tile_dma_dst_data_set_id=DATA_SET_ID,
            tile_dma_dst_user_tag=index,
            tile_dma_dst_fifo_mode=1,
            tile_dma_dst_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_dst_trunc=F16_TRUNC,
            tile_dma_dst_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_dst_zero_pad_first=1,
            tile_dma_dst_zero_pad_last=1,
            tile_dma_dst_interleave=INTERLEAVE,
            **(
                {}
                if staged_source
                else {
                    **row_stride("tile_dma_dst_row_stride", destination.dram_row_stride),
                    "tile_dma_dst_channel_stride": dram_units(destination.channel),
                }
            ),
        ),
    )


NE_LUT_BYTES = 128


def activation(
    source: TensorDesc,
    *,
    lut_bytes: int = NE_LUT_BYTES,
    address: int = 0,
    dest_address: int = 0,
) -> dict[int, int]:
    """Program a neural engine activation that reads a lookup table.

    The reference streams for silu and sigmoid are register-identical: the
    activation lives in the table the descriptor points at, so the words carry
    only its size and enable.  Every stride here is the dense row of the staged
    row, and the result lands one row along.
    """
    source.validate_feature()
    low, high = split_address(address)
    dest_low, dest_high = split_address(dest_address)
    row = source.width * FP16_BYTES
    column = l2_units(row)
    return splice(
        COMMON.pack(
            win=source.width,
            hin=source.height,
            din=1,
            cin=source.channels,
            cout=source.channels,
            wout=source.width,
            hout=source.height,
            dout=1,
            ne_ocg_size=0,
            num_groups=1,
            unicast_en=1,
            unicast_cin=1,
            tile_height=1,
            node_id=1,
        ),
        L2.pack(
            l2_src_type=fields.L2Access.DMA_READ,
            l2_src_alias_conv_src=1,
            l2_src_alias_conv_rslt=1,
            l2_src_dma_format=fields.TensorFormatMode.COMPRESSED,
            l2_src_interleave=INTERLEAVE,
            l2_src_alias_planar_src=1,
            l2_src_alias_planar_rslt=1,
            l2_src_channel_stride=column,
            l2_src_row_stride=column,
            l2_src_depth_stride=column,
            l2_src_group_stride=column,
            l2_result_type=fields.L2Access.DMA_READ,
            l2_result_buffer_mode=1,
            l2_result_alias_src=1,
            l2_result_alias_rslt=1,
            l2_result_dma_fmt=fields.TensorFormatMode.COMPRESSED,
            l2_result_interleave=INTERLEAVE,
            l2_result_alias_planar_src=1,
            l2_result_alias_planar_rslt=1,
            l2_result_address=column,
        ),
        NE.pack(
            kernel_format=fields.KernelFormat.FP16,
            ne_lut_select=1,
            ne_lut_mode=1,
            op=fields.NEOp.BYPASS,
            kernel_mode=1,
            ne_binary_point=NE_BINARY_POINT,
            nonlinear_mode=fields.NENonlinear.LUT,
            arg_output_select=NE_OUTPUT_SELECT,
            post_scale=fp16(1.0, "post_scale"),
        ),
        SRC_DMA.pack(
            tile_dma_src_hint=SOURCE_HINT,
            tile_dma_src_base_addr_lo=low,
            tile_dma_src_base_addr_hi=high,
            tile_dma_src_data_set_id=DATA_SET_ID,
            tile_dma_src_cache_hint=SOURCE_CACHE_HINT,
            tile_dma_src_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_src_trunc=F16_TRUNC,
            tile_dma_src_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_src_interleave=INTERLEAVE,
        ),
        DST_DMA.pack(
            tile_dma_dst_enable=1,
            tile_dma_dst_base_addr_lo=dest_low,
            tile_dma_dst_base_addr_hi=dest_high,
            tile_dma_dst_hint=DEST_HINT,
            tile_dma_dst_data_set_id=DATA_SET_ID,
            tile_dma_dst_fifo_mode=1,
            tile_dma_dst_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_dst_trunc=F16_TRUNC,
            tile_dma_dst_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_dst_zero_pad_first=1,
            tile_dma_dst_zero_pad_last=1,
            tile_dma_dst_interleave=INTERLEAVE,
        ),
        KERNEL_DMA.pack(
            kernel_dma_src_enable=1,
            kernel_dma_armed=1,
            kernel_dma_coeff_present=1,
            kernel_dma_coeff_tag=1,
            nonlinear_lut_enable=1,
            nonlinear_lut_loaded=1,
            nonlinear_lut_size=dram_units(lut_bytes),
        ),
        KERNEL_DMA.pack(
            kernel_dma_lut_base=0,
            kernel_dma_aux_base_lo=0,
            kernel_dma_aux_base_hi=0,
        ),
    )


def multiply(
    channels: int,
    height: int,
    width: int,
    *,
    second_base: int,
    result_base: int,
    patch: tuple[int, int],
    transpose: bool,
    result_config: int = 0,
    result_stride: int | None = None,
    read_config: int = 0,
    to_dram: bool = False,
    lut_armed: bool = False,
    index: int = 0,
    width_stride: int | None = None,
    address: int = 0,
    second_address: int = 0,
) -> dict[int, int]:
    """Multiply two staged sources and write the product where the plan says.

    The operand layout, the rasterization patch, the transposition flags and the
    result slot all come from the memory allocator, so the caller passes them;
    only the register encoding is derived here.  ``result_config`` carries the
    allocator's L2 result configuration word when it is not the plain read form.
    """
    low, high = split_address(address)
    dest_low, dest_high = split_address(second_address)
    return splice(
        COMMON.pack(
            win=width,
            hin=height,
            din=1,
            cin=channels,
            cout=channels,
            wout=width,
            hout=height,
            dout=1,
            tile_height=1,
            patch_width=patch[0],
            patch_height=patch[1],
            **({} if not transpose else {"pe_src1_transpose": 1, "pe_src2_transpose": 1}),
            source_routing=1,
            task_type=fields.TaskType.PE_EW_BINARY,
            node_id=0,
        ),
        L2.pack(
            l2_src_type=fields.L2Access.L2_READ,
            l2_src_alias_conv_rslt=1,
            l2_src_interleave=INTERLEAVE,
            l2_src2_base_address=l2_units(second_base),
            **({} if not read_config else {"l2_src_channel_stride": read_config}),
            **(
                {}
                if not result_config
                else {
                    "l2_result_type": (
                        fields.L2Access.DMA_READ if to_dram else fields.L2Access.L2_READ
                    ),
                    **({} if not to_dram else {"l2_result_buffer_mode": 1}),
                    "l2_result_dma_fmt": fields.TensorFormatMode.COMPRESSED,
                    "l2_result_interleave": INTERLEAVE,
                }
            ),
            l2_result_address=l2_units(result_base),
            **(
                {}
                if result_stride is None
                else {"l2_result_channel_stride": l2_units(result_stride)}
            ),
        ),
        PE.pack(op=fields.PEOp.MUL, second_source=2),
        {} if not lut_armed else KERNEL_DMA.pack(kernel_dma_armed=1),
        (
            {}
            if not to_dram
            else DST_DMA.pack(
                tile_dma_dst_enable=1,
                tile_dma_dst_base_addr_lo=dest_low,
                tile_dma_dst_base_addr_hi=dest_high,
                tile_dma_dst_hint=DEST_HINT,
                tile_dma_dst_data_set_id=DATA_SET_ID,
                tile_dma_dst_user_tag=index,
                tile_dma_dst_fifo_mode=1,
                tile_dma_dst_format_mode=fields.TensorFormatMode.COMPRESSED,
                tile_dma_dst_trunc=F16_TRUNC,
                tile_dma_dst_memory_format=fields.TensorMemoryFormat.TILED,
                tile_dma_dst_zero_pad_first=1,
                tile_dma_dst_zero_pad_last=1,
                tile_dma_dst_interleave=INTERLEAVE,
            )
        ),
    )


def end_of_network() -> dict[int, int]:
    """Mark the descriptor that finishes the stream.

    Only the last descriptor of a stream carries this word; earlier descriptors
    leave it at reset, so a multi-descriptor stream does not repeat it.
    """
    return KERNEL_DMA.pack(kernel_dma_armed=KERNEL_ARMED)


def compute_pe(
    kind: fields.TaskType,
    output: TensorDesc,
    *,
    op: fields.PEOp | None = None,
    second_source: int | None = None,
    source_routing: int = 1,
    declared_in: TensorDesc | None = None,
    declared_out: TensorDesc | None = None,
    tile_height: int | None = None,
    pool_mode: fields.PoolMode = fields.PoolMode.NONE_OR_SUM,
    compare: int | None = None,
    window: tuple[int, int, int, int] | None = None,
    padding: tuple[int, int] = (0, 0),
    patch: tuple[int, int] | None = None,
    pre_scale: float = 1.0,
    scale: float = 1.0,
    bias: float = 0.0,
    absolute: bool = False,
    square: bool = False,
    clamp: tuple[float, float] | None = None,
    source_relu: bool = False,
    final_scale: float = 1.0,
) -> dict[int, int]:
    """Program one planar engine pass over an already loaded feature map.

    A reduction multiplies its result by the fp32 ``final_scale``, whose word is
    left unwritten at one.
    """
    output.validate_feature()
    geometry_in = output if declared_in is None else declared_in
    geometry_out = output if declared_out is None else declared_out
    geometry_in.validate_feature()
    geometry_out.validate_feature()
    shape = (
        {}
        if window is None
        else {
            "kw": window[0],
            "kh": window[1],
            "sx": window[2],
            "sy": window[3],
            "pad_left": padding[0],
            "pad_top": padding[1],
            "out_offset_x": 1 if window is not None else 0,
            "out_offset_y": 1 if window is not None else 0,
        }
    )
    size = {} if patch is None else {"patch_width": patch[0], "patch_height": patch[1]}
    low, high = (0.0, float("inf")) if clamp is None else clamp
    if not low <= high:
        raise ValueError(f"clamp bounds {clamp} are not ordered")
    return splice(
        COMMON.pack(
            **shape,
            **size,
            win=geometry_in.width,
            hin=geometry_in.height,
            din=1,
            cin=geometry_in.channels,
            wout=geometry_out.width,
            hout=geometry_out.height,
            dout=1,
            cout=geometry_out.channels,
            common_in_fmt=COMMON_FORMAT_F16,
            common_src2_in_fmt=COMMON_FORMAT_F16,
            common_out_fmt=COMMON_FORMAT_F16,
            tile_height=output.height if tile_height is None else tile_height,
            source_routing=source_routing,
            task_type=kind,
            node_id=0,
        ),
        L2.pack(
            pe_src1_re_lu=int(source_relu),
            **({} if compare is None else {"l2_reduce_compare": compare}),
        ),
        PE.pack(
            **({} if second_source is None else {"second_source": second_source}),
            pool_mode=pool_mode,
            op=fields.PEOp.SUM_SQUARE if square else op or fields.PEOp.ADD,
            clamp_enable=int(clamp is not None),
            abs_enable=int(absolute),
            pre_scale=fp16(pre_scale, "pre_scale"),
            scale=fp16(scale, "scale"),
            bias=fp16(bias, "bias"),
            clamp_low=fp16(low, "clamp_low"),
            clamp_high=fp16(high, "clamp_high"),
            **({} if final_scale == 1.0 else {"final_scale": fp32(final_scale, "final_scale")}),
        ),
    )

def compute_ne(
    op: fields.NEOp,
    output: TensorDesc,
    *,
    nonlinear: fields.NENonlinear = fields.NENonlinear.NONE,
    kernel_mode: int = 0,
    post_scale: float = 1.0,
    kernel_format: fields.KernelFormat | None = None,
    source_relu: bool = False,
    width_code: int | None = None,
    active_nodes: int = 4,
    declared: TensorDesc | None = None,
    transpose: bool = False,
    unicast_channels: int = 1,
    ocg_size: int | None = 0,
) -> dict[int, int]:
    """Program one neural engine pass over an already staged feature map.

    ``kernel_format`` stays unset for a descriptor without a coefficient stream,
    which is what the reference containers do. ``declared`` is the geometry
    before the output transpose, including the row that selects the width code.
    A transposing pass sets every shift slice to the width code; a pass that keeps
    its axes sets only the maximum slice, as the reference identity bypasses do.
    """
    output.validate_feature()
    if output.channels < NE_CHANNEL_GROUP:
        raise ValueError(
            f"neural engine pass over {output.channels} channels is not decoded yet"
        )
    shown = output if declared is None else declared
    shown.validate_feature()
    dense_row = shown.width * FP16_BYTES
    ratio = NE_WIDTH_UNIT // dense_row
    if NE_WIDTH_UNIT % dense_row or ratio < 1 or ratio & (ratio - 1):
        raise ValueError(f"neural width code for dense row {dense_row} bytes is not decoded yet")
    expected = ratio.bit_length() - 1
    if width_code is None:
        width_code = expected
    elif type(width_code) is not int or not 0 <= width_code <= 4:
        raise ValueError(f"width_code {width_code!r} must be within [0, 4]")
    elif width_code != expected:
        raise ValueError(f"width_code {width_code} contradicts dense row {dense_row}; expected {expected}")
    return splice(
        COMMON.pack(
            common_in_fmt=COMMON_FORMAT_F16,
            common_src2_in_fmt=COMMON_FORMAT_F16,
            common_out_fmt=COMMON_FORMAT_F16,
            win=shown.width,
            hin=shown.height,
            wout=shown.width,
            hout=shown.height,
            cout=shown.channels,
            **({} if ocg_size is None else {"ne_ocg_size": ocg_size}),
            num_groups=1,
            unicast_en=1,
            unicast_cin=unicast_channels,
            tile_height=output.height,
            source_routing=1,
            task_type=fields.TaskType.NE,
            shift_maximum=width_code,
            **({"shift_minimum": width_code, "shift_preferred": width_code} if transpose else {}),
            active_nodes=active_nodes,
            output_transpose=1 if transpose else 0,
            fill_lower_nodes_first=1 if transpose and unicast_channels > 1 else 0,
            node_id=1,
        ),
        L2.pack(pe_src1_re_lu=int(source_relu)),
        NE.pack(
            op=op,
            kernel_mode=kernel_mode,
            ne_binary_point=NE_BINARY_POINT,
            nonlinear_mode=nonlinear,
            arg_output_select=NE_OUTPUT_SELECT,
            post_scale=fp16(post_scale, "post_scale"),
            **({} if kernel_format is None else {"kernel_format": kernel_format}),
        ),
    )


__all__ = [
    "NE_LUT_BYTES",
    "activation",
    "align_up",
    "coefficient_dma",
    "compute_ne",
    "compute_pe",
    "convolution",
    "end_of_network",
    "fp16",
    "fp32",
    "load",
    "load_chained",
    "load_second_chained",
    "multiply",
    "require_row_width",
    "splice",
    "store",
]
