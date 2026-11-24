"""Hardware primitives: the second layer between TIR and a task descriptor.

A task descriptor bundles the data movement and the arithmetic of one engine pass
into a single VLIW instruction.  The primitives below each program one group of
configuration words and return them as ``{byte address: word}``:

``load(surface)``       read a feature map: the L2 source buffer and the source DMA
``store(surface)``      write a feature map: the L2 result buffer and the result DMA
``compute_pe(kind, ...)``  planar engine arithmetic
``compute_ne(op, ...)``    neural engine arithmetic

Words produced by different primitives that land on the same address are combined
with :func:`splice`, which ORs them; the fields of one word never overlap, so the
result is the same as programming them one after another.

Every value here is expressed through :mod:`tilefoundry_apple.ane.isa.fields`, so
a bit position appears exactly once, in the field declaration.

A feature map has one canonical surface geometry: rows are padded to 64 bytes,
channels are ``height`` rows apart, and the extent is ``channels * channel``.
L2 staging uses the same shape with 16-byte padded rows and one extra channel of
slack, which is what the staged buffer needs for the engine's read-ahead.
"""

import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import IntEnum

from . import fields

FP16_BYTES = 2
SURFACE_ROW_ALIGN = 64
L2_ROW_ALIGN = 16
L2_UNIT = 16
STAGING_ROW_GUARD = 32
STREAM_TAG = 1
DATA_SET_ID = 14
PLANAR_MIN_ROW_BYTES = 32
NEURAL_MIN_ROW_BYTES = 16
NE_ROW_GUARD = 16
NE_GUARD_PERIOD = 1024
#: The neural engine expresses a row width as a power-of-two subdivision of this.
NE_WIDTH_UNIT = 256
NE_PIPELINE_CODE = 7
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


class Engine(IntEnum):
    """The engine a staged buffer is prepared for."""

    PLANAR = 0
    NEURAL = 1


#: Channels one neural-engine pass covers.
NE_CHANNEL_GROUP = 16

L2 = fields.L2()
PE = fields.PE()
NE = fields.NE()
COMMON = fields.Common()
SRC_DMA = fields.TileDMASrc()
DST_DMA = fields.TileDMADst()
KERNEL_DMA = fields.KernelDMA()


def align_up(value: int, alignment: int) -> int:
    """Round a byte size up to its required alignment."""
    return (value + alignment - 1) // alignment * alignment


def fp16(value: float, name: str) -> int:
    """Encode one immediate as an fp16 bit pattern."""
    try:
        return int.from_bytes(struct.pack("<e", value), "little")
    except OverflowError as error:
        raise ValueError(f"{name} {value} does not fit fp16") from error


def splice(*parts: Mapping[int, int]) -> dict[int, int]:
    """Combine primitive words; words at the same address are OR-ed."""
    image: dict[int, int] = {}
    for part in parts:
        for addr, word in part.items():
            image[addr] = image.get(addr, 0) | word
    return image


@dataclass(frozen=True)
class Surface:
    """One feature map as the DMA and the L2 buffer address it."""

    channels: int
    height: int
    width: int
    row: int
    channel: int
    tile_pitch: int | None = None

    @property
    def extent(self) -> int:
        """Bytes the DRAM surface spans."""
        return self.channels * self.channel

    @property
    def dram_row_stride(self) -> int:
        """The row stride a DMA programs.

        The row stride is only meaningful while every extent of the surface is
        greater than one; a surface with a one-extent leaves the field at zero,
        which is what the reference descriptors do for a reduced result.
        """
        return 0 if min(self.width, self.height, self.channels) == 1 else self.row

    @property
    def element_strides(self) -> tuple[int, int, int, int]:
        """The strides a caller declares for this surface, in elements."""
        return (0, self.channel // FP16_BYTES, self.row // FP16_BYTES, 1)

    @property
    def l2_channel(self) -> int:
        """Bytes between adjacent staged channels.

        A tile narrower than the feature map stages its columns into the L2
        image, so ``tile_pitch`` overrides the dense row while the declared
        window and the DRAM strides keep the logical width.
        """
        if self.tile_pitch is not None:
            return self.tile_pitch
        return align_up(self.width * FP16_BYTES, L2_ROW_ALIGN)

    def staged(self, width: int | None = None, rows: int | None = None) -> "Surface":
        """Return the same feature map staged as a tile of ``width`` and ``rows``.

        A descriptor that stages a tile narrower than the logical feature map
        keeps the declared geometry for its window and uses the tile for the L2
        layout, so the staged image is a separate surface.
        """
        if width is None or width == self.width:
            return self
        if width < 1:
            raise ValueError(f"staged width {width} is empty")
        if width > self.width:
            return self
        return replace(self, tile_pitch=align_up(width * FP16_BYTES, L2_ROW_ALIGN))

    def staging_row(self, engine: Engine = Engine.PLANAR, rows: int | None = None) -> int:
        """Bytes between adjacent staged rows for one engine.

        The planar engine stages one channel slab of ``channels * l2_channel``
        bytes followed by a constant ``STAGING_ROW_GUARD`` band.

        The neural guard band below is **known incomplete**: it reproduces the
        thirteen shapes it was read from, and is refuted outside them (for
        example a 31x15x32 image stages 144 guard bytes, not the 0 this predicts,
        while a 24x11x64 image stages none, not the 16).  The band is produced by
        the L2 tile allocator, which sits upstream of the descriptor, so a closed
        form has to come from that allocator rather than from more samples.
        """
        partial = rows is not None and rows < self.height
        depth = self.channels * self.l2_channel
        if engine is Engine.PLANAR:
            return depth + STAGING_ROW_GUARD
        if engine is not Engine.NEURAL:
            raise ValueError(f"unknown engine {engine!r}")
        if partial:
            return depth
        slack = self.l2_channel if self.channels % NE_CHANNEL_GROUP == 0 else 0
        guard = NE_ROW_GUARD if depth % NE_GUARD_PERIOD == 0 else 0
        return depth + slack + guard

    def staging_extent(self, engine: Engine = Engine.PLANAR, rows: int | None = None) -> int:
        """Bytes the staged buffer spans.

        ``rows`` is the number of rows one descriptor stages, which is the tile
        height rather than the whole surface.
        """
        return self.staging_row(engine, rows) * (self.height if rows is None else rows)

    @property
    def l2_row(self) -> int:
        """Bytes between adjacent staged rows for the planar engine."""
        return self.staging_row(Engine.PLANAR)

    @property
    def l2_extent(self) -> int:
        """Bytes the planar engine's staged buffer spans."""
        return self.staging_extent(Engine.PLANAR)


def surface(channels: int, height: int, width: int) -> Surface:
    """Derive the canonical fp16 surface of one feature map."""
    if min(channels, height, width) < 1:
        raise ValueError(f"feature map {channels}x{height}x{width} has an empty extent")
    row = align_up(width * FP16_BYTES, SURFACE_ROW_ALIGN)
    return Surface(channels, height, width, row, height * row)


def require_row_width(surface: Surface, minimum: int) -> None:
    """Reject a feature map whose dense row selects an undecoded format entry."""
    dense = surface.width * FP16_BYTES
    if dense < minimum:
        raise ValueError(
            f"feature map {surface.channels}x{surface.height}x{surface.width} has a {dense} "
            f"byte dense row; the format entry for rows below {minimum} bytes is not decoded"
        )


def row_stride(name: str, byte_stride: int) -> dict[str, int]:
    """The row-stride field, omitted for a surface whose extent the hardware ignores."""
    return {} if byte_stride == 0 else {name: _dram_units(byte_stride)}


def _dram_units(byte_stride: int) -> int:
    """Express a DRAM stride in the unit its field holds."""
    if byte_stride % SURFACE_ROW_ALIGN:
        raise ValueError(f"stride {byte_stride} is not a multiple of {SURFACE_ROW_ALIGN}")
    return byte_stride // SURFACE_ROW_ALIGN


def _l2_units(byte_stride: int) -> int:
    """Express an L2 stride in the unit its field holds."""
    if byte_stride % L2_UNIT:
        raise ValueError(f"stride {byte_stride} is not a multiple of {L2_UNIT}")
    return byte_stride // L2_UNIT


def _split_address(address: int) -> tuple[int, int]:
    """Split a byte address into the low and high words the DMA programs."""
    if not 0 <= address < (1 << 64):
        raise ValueError(f"address {address:#x} does not fit 64 bits")
    return address & 0xFFFFFFFF, address >> 32


def _staged_flags(fresh: bool, planar_result: bool = False) -> dict[str, int]:
    """Return the buffer-reuse bits a descriptor programs for its staged slots.

    A descriptor that stages into fresh L2 buffers marks each slot as first use;
    a descriptor that continues a stream reuses the buffers its predecessor left
    behind, so it clears those bits and tags its DMA words with the stream index.
    """
    alias = 1 if fresh else 0
    share = 1 if planar_result else alias
    return {
        "l2_src_alias_conv_src": alias,
        "l2_src_alias_conv_rslt": alias,
        "l2_src_alias_planar_src": alias,
        "l2_src_alias_planar_rslt": share,
        "l2_result_alias_src": alias,
        "l2_result_alias_rslt": alias,
        "l2_result_alias_planar_src": alias,
        "l2_result_alias_planar_rslt": alias,
    }


def _src_tags(fresh: bool) -> dict[str, int]:
    """Return the stream tag a descriptor stamps on its first source transfer."""
    return {"tile_dma_src_user_tag": 0 if fresh else STREAM_TAG}


def _dst_tags(fresh: bool) -> dict[str, int]:
    """Return the stream tag a descriptor stamps on its result transfer."""
    return {"tile_dma_dst_user_tag": 0 if fresh else STREAM_TAG}


def load(
    source: Surface,
    engine: Engine = Engine.PLANAR,
    *,
    address: int = 0,
    fresh: bool = True,
    rows: int | None = None,
    planar_result: bool = False,
) -> dict[int, int]:
    """Program reading one feature map: the L2 source buffer and the source DMA."""
    require_row_width(
        source, NEURAL_MIN_ROW_BYTES if engine is Engine.NEURAL else PLANAR_MIN_ROW_BYTES
    )
    low, high = _split_address(address)
    return splice(
        COMMON.pack(
            win=source.width,
            hin=source.height,
            din=1,
            cin=source.channels,
            common_in_fmt=COMMON_FORMAT_F16,
            common_src2_in_fmt=COMMON_FORMAT_F16,
        ),
        L2.pack(
            l2_src_type=fields.L2Access.DMA_READ,
            **_staged_flags(fresh, planar_result),
            l2_src_dma_format=fields.TensorFormatMode.COMPRESSED,
            l2_src_interleave=INTERLEAVE,
            l2_src_channel_stride=_l2_units(source.l2_channel),
            l2_src_row_stride=_l2_units(source.staging_row(engine, rows)),
            l2_src_depth_stride=_l2_units(source.channels * source.l2_channel),
            l2_src_group_stride=_l2_units(source.channels * source.l2_channel),
        ),
        SRC_DMA.pack(
            **_src_tags(fresh),
            tile_dma_src_hint=SOURCE_HINT,
            tile_dma_src_base_addr_lo=low,
            tile_dma_src_base_addr_hi=high,
            tile_dma_src_data_set_id=DATA_SET_ID,
            tile_dma_src_cache_hint=SOURCE_CACHE_HINT,
            **row_stride("tile_dma_src_row_stride", source.dram_row_stride),
            tile_dma_src_channel_stride=_dram_units(source.channel),
            tile_dma_src_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_src_trunc=F16_TRUNC,
            tile_dma_src_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_src_interleave=INTERLEAVE,
        ),
    )


def slot_offset(
    surface: Surface,
    slots: int,
    engine: Engine = Engine.PLANAR,
    rows: int | None = None,
    *,
    separated: bool = False,
) -> int:
    """Return the byte offset a staged slot starts at.

    Slots are one staged image apart.  A tile that stages fewer rows than the
    surface spans carries an extra guard band between slots, which the compiler
    emits so that adjacent staged images stay on separate row boundaries.
    """
    extent = surface.staging_extent(engine, rows)
    partial = rows is not None and rows < surface.height
    separator = STAGING_ROW_GUARD if separated and partial else 0
    return slots * extent + separator


def load_chained(
    source: Surface, engine: Engine = Engine.PLANAR, *, rows: int | None = None
) -> dict[int, int]:
    """Read a source the previous descriptor left staged instead of reloading it.

    A chained pass names the L2 buffer its predecessor filled, so it programs no
    source DMA at all: the L2 source type alone selects the chain path.  The
    encoding is read from the interface enum; the shape of the staged image is
    the one the predecessor wrote, so the strides come from the same surface.
    """
    require_row_width(source, PLANAR_MIN_ROW_BYTES)
    return splice(
        COMMON.pack(
            win=source.width,
            hin=source.height,
            din=1,
            cin=source.channels,
            common_in_fmt=COMMON_FORMAT_F16,
            common_src2_in_fmt=COMMON_FORMAT_F16,
        ),
        L2.pack(
            l2_src_type=fields.L2Access.L2_CHAIN_READ,
            l2_src_channel_stride=_l2_units(source.l2_channel),
            l2_src_row_stride=_l2_units(source.staging_row(engine, rows)),
            l2_src_depth_stride=_l2_units(source.channels * source.l2_channel),
            l2_src_group_stride=_l2_units(source.channels * source.l2_channel),
        ),
    )


def load_second(
    source: Surface,
    engine: Engine = Engine.PLANAR,
    *,
    rows: int | None = None,
    address: int = 0,
    fresh: bool = True,
) -> dict[int, int]:
    """Program a second staged source, placed one staged slot after the first."""
    require_row_width(source, PLANAR_MIN_ROW_BYTES)
    low, high = _split_address(address)
    offset = _l2_units(slot_offset(source, 1, engine, rows, separated=True))
    return splice(
        L2.pack(
            l2_src_second_stream=1,
            l2_src2_base_address=offset,
            l2_src2_channel_stride=_l2_units(source.l2_channel),
            l2_src2_row_stride=_l2_units(source.staging_row(engine, rows)),
            l2_src2_depth_stride=_l2_units(source.channels * source.l2_channel),
            l2_src2_group_stride=_l2_units(source.channels * source.l2_channel),
        ),
        SRC_DMA.pack(
            **_src_tags(fresh),
            tile_dma_src2_user_tag=0 if fresh else STREAM_TAG,
            tile_dma_src2_hint=SECOND_SOURCE_HINT,
            tile_dma_src2_data_set_id=DATA_SET_ID,
            tile_dma_src2_cache_hint=SOURCE_CACHE_HINT,
            tile_dma_src2_base_addr_lo=low,
            tile_dma_src2_base_addr_hi=high,
            **row_stride("tile_dma_src2_row_stride", source.dram_row_stride),
            tile_dma_src2_channel_stride=_dram_units(source.channel),
            tile_dma_src2_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_src2_trunc=F16_TRUNC,
            tile_dma_src2_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_src2_interleave=INTERLEAVE,
        ),
    )


def store(
    destination: Surface,
    engine: Engine = Engine.PLANAR,
    *,
    staged: int = 1,
    staging: Surface | None = None,
    rows: int | None = None,
    guard: bool = False,
    address: int = 0,
    fresh: bool = True,
    separated: bool | None = None,
    declare: bool = True,
    declared: Surface | None = None,
) -> dict[int, int]:
    """Program writing one feature map: the L2 result buffer and the result DMA.

    The result buffer starts after the staged images, so ``staging`` names the
    surface whose staged extent locates it; it defaults to the destination.
    """
    low, high = _split_address(address)
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
            l2_result_type=fields.L2Access.DMA_READ,
            l2_result_buffer_mode=1,
            **_staged_flags(fresh),
            l2_result_dma_fmt=fields.TensorFormatMode.COMPRESSED,
            l2_result_interleave=INTERLEAVE,
            l2_result_base_addr=_l2_units(
                slot_offset(
                    staging or destination,
                    staged,
                    engine,
                    rows,
                    separated=staged >= 2 if separated is None else separated,
                )
            ),
        ),
        DST_DMA.pack(
            **_dst_tags(fresh),
            tile_dma_dst_enable=1,
            tile_dma_dst_base_addr_lo=low,
            tile_dma_dst_base_addr_hi=high,
            tile_dma_dst_hint=DEST_HINT,
            tile_dma_dst_data_set_id=DATA_SET_ID,
            tile_dma_dst_fifo_mode=FIFO_MODE,
            **row_stride("tile_dma_dst_row_stride", destination.dram_row_stride),
            tile_dma_dst_channel_stride=_dram_units(destination.channel),
            tile_dma_dst_format_mode=fields.TensorFormatMode.COMPRESSED,
            tile_dma_dst_trunc=F16_TRUNC,
            tile_dma_dst_memory_format=fields.TensorMemoryFormat.TILED,
            tile_dma_dst_zero_pad_first=1,
            tile_dma_dst_zero_pad_last=1,
            tile_dma_dst_interleave=INTERLEAVE,
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
    output: Surface,
    *,
    op: fields.PEOp | None = None,
    second_source: int | None = None,
    source_routing: int = 1,
    declared_in: Surface | None = None,
    declared_out: Surface | None = None,
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
) -> dict[int, int]:
    """Program one planar engine pass over an already loaded feature map."""
    geometry_in = output if declared_in is None else declared_in
    geometry_out = output if declared_out is None else declared_out
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
        ),
    )


def compute_ne(
    op: fields.NEOp,
    output: Surface,
    *,
    nonlinear: fields.NENonlinear = fields.NENonlinear.NONE,
    kernel_mode: int = 0,
    post_scale: float = 1.0,
    kernel_format: fields.KernelFormat | None = None,
    source_relu: bool = False,
    tile_height: int | None = None,
    width_code: int | None = None,
    declared: Surface | None = None,
    transpose: bool = False,
    unicast_channels: int = 1,
    ocg_size: int | None = 0,
    address: int = 0,
) -> dict[int, int]:
    """Program one neural engine pass over an already staged feature map.

    ``kernel_format`` stays unset for a descriptor without a coefficient stream,
    which is what the reference containers do.
    """
    if output.channels < NE_CHANNEL_GROUP:
        raise ValueError(
            f"neural engine pass over {output.channels} channels is below one "
            f"{NE_CHANNEL_GROUP}-channel group; that configuration is not decoded"
        )
    if width_code is None:
        ratio = NE_WIDTH_UNIT // (output.width * FP16_BYTES)
        if ratio < 1 or ratio & (ratio - 1):
            raise ValueError(
                f"feature map row {output.width * FP16_BYTES} bytes is not a power-of-two "
                f"divisor of {NE_WIDTH_UNIT}; the neural width code is not decoded"
            )
        width_code = ratio.bit_length() - 1
    shown = output if declared is None else declared
    return splice(
        COMMON.pack(
            win=shown.width,
            hin=shown.height,
            wout=shown.width,
            hout=shown.height,
            cout=shown.channels,
            **({} if ocg_size is None else {"ne_ocg_size": ocg_size}),
            num_groups=1,
            unicast_en=1,
            unicast_cin=unicast_channels,
            tile_height=output.height if tile_height is None else tile_height,
            source_routing=1,
            task_type=fields.TaskType.NE,
            ne_width_code=width_code,
            ne_full_group=1,
            output_transpose=1 if transpose else 0,
            input_transpose=1 if transpose else 0,
            node_id=1,
        ),
        L2.pack(pe_src1_re_lu=int(source_relu)),
        NE.pack(
            op=op,
            kernel_mode=kernel_mode,
            ne_pipeline_code=NE_PIPELINE_CODE,
            nonlinear_mode=nonlinear,
            arg_output_select=NE_OUTPUT_SELECT,
            post_scale=fp16(post_scale, "post_scale"),
            **({} if kernel_format is None else {"kernel_format": kernel_format}),
        ),
    )


__all__ = [
    "Engine",
    "Surface",
    "align_up",
    "compute_ne",
    "compute_pe",
    "require_row_width",
    "fp16",
    "load",
    "load_chained",
    "load_second",
    "splice",
    "store",
    "surface",
]
