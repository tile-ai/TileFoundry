"""Bit fields and configuration blocks for the H14 ANE."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar


def check_word(value: int) -> None:
    """Reject values that cannot be written as an unsigned 32-bit word."""
    if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{value!r} is not an unsigned 32-bit word")


@dataclass(frozen=True)
class BitField:
    """An inclusive bit range in a word at a byte address."""

    addr: int
    hi: int
    lo: int
    allowed: range | frozenset[int] | type[IntEnum] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= self.hi < 32:
            raise ValueError(f"invalid bit range [{self.hi}:{self.lo}]")

    @property
    def mask(self) -> int:
        """Return the field mask at its position in the word."""
        return ((1 << (self.hi - self.lo + 1)) - 1) << self.lo

    def get(self, word: int) -> int:
        """Read the field as unshifted bits."""
        return (word & self.mask) >> self.lo

    def put(self, word: int, val: int) -> int:
        """Replace the field without changing other bits of the word."""
        check_word(word)
        if not isinstance(val, int) or not 0 <= val <= self.mask >> self.lo:
            raise ValueError(f"{val!r} does not fit bits [{self.hi}:{self.lo}] at {self.addr:#06x}")
        if self.allowed is not None and val not in self.allowed:
            raise ValueError(
                f"{val} is not allowed in bits [{self.hi}:{self.lo}] at {self.addr:#06x}"
            )
        return (word & ~self.mask) | (val << self.lo)


class Block:
    """A contiguous range of configuration words with named fields."""

    base: ClassVar[int]
    end: ClassVar[int]
    fields: ClassVar[dict[str, BitField]]

    def __init_subclass__(cls, *, base: int, end: int) -> None:
        super().__init_subclass__()
        if not 0 <= base < end or base % 4 or end % 4:
            raise ValueError(f"{cls.__name__}: invalid block bounds {base:#x}..{end:#x}")
        cls.base, cls.end = base, end
        fields: dict[str, BitField] = {}
        masks: dict[int, int] = {}
        for name, declaration in vars(cls).items():
            if isinstance(declaration, BitField):
                entries: Iterable[tuple[str, BitField]] = ((name, declaration),)
            elif isinstance(declaration, tuple) and all(
                isinstance(field, BitField) for field in declaration
            ):
                entries = ((f"{name}[{i}]", field) for i, field in enumerate(declaration))
            else:
                continue
            for field_name, field in entries:
                if field.addr % 4 or not base <= field.addr < end:
                    raise ValueError(f"{cls.__name__}.{field_name}: field outside the block")
                mask = masks.get(field.addr, 0)
                if mask & field.mask:
                    raise ValueError(f"{cls.__name__}.{field_name}: overlapping fields")
                masks[field.addr] = mask | field.mask
                fields[field_name] = field
        cls.fields = fields

    def pack(self, **name_val: int) -> dict[int, int]:
        """Combine named field values into words starting from zero."""
        image: dict[int, int] = {}
        for name, val in name_val.items():
            field = self.fields[name]
            image[field.addr] = field.put(image.get(field.addr, 0), val)
        return image


class TaskType(IntEnum):
    """Select the engine and task mode."""

    NE = 0
    PE_POOL_UNARY = 1
    PE_EW_UNARY = 2
    PE_EW_BINARY = 3
    PE_GOC = 4
    PE_EW_REDUCE_UNARY = 5
    PE_EW_REDUCE_BINARY = 6
    L2_SPILL_FILL = 7


class NEOp(IntEnum):
    """Select the neural engine operation."""

    CONV = 0
    EW_SQUARE = 1
    EW_MUL = 2
    POOL = 3
    BYPASS = 4


class PEOp(IntEnum):
    """Select the planar engine arithmetic operation."""

    ADD = 0
    MUL = 1
    MAX = 2
    MIN = 3
    SUM_SQUARE = 4


class PoolMode(IntEnum):
    """Select the pooling or reduction mode."""

    NONE_OR_SUM = 0
    AVG = 1
    MAX = 2
    MIN = 3


class PENonlinear(IntEnum):
    """Select the function applied after reduction."""

    NONE = 0
    INVERSE = 1
    RSQRT = 2
    SQRT = 3


class NENonlinear(IntEnum):
    """Select the neural engine activation."""

    NONE = 0
    RELU = 1
    LUT = 2


class KernelFormat(IntEnum):
    """Select the weight element format."""

    UINT8 = 0
    INT8 = 1
    FP16 = 2


class L2Access(IntEnum):
    """How a surface reaches the L2 buffer."""

    L2_READ = 0
    DMA_READ2 = 1
    DMA_READ = 2
    L2_CHAIN_READ = 3


class TensorFormatMode(IntEnum):
    """Format mode of a transferred surface."""

    NONE = 0
    COMPRESSED = 1
    LOSSY = 2


class TensorMemoryFormat(IntEnum):
    """Memory layout of a transferred surface."""

    A = 0
    B = 1
    TILED = 2
    C = 3


SPATIAL_EXTENTS = range(1, 16385)
CHANNEL_EXTENTS = range(1, 65537)
KERNEL_EXTENTS = range(1, 33)
CONV_STRIDES = range(1, 3)


class Common(Block, base=0x0000, end=0x004C):
    """Tensor geometry, convolution window and task selection."""

    win = BitField(0x0000, 14, 0, SPATIAL_EXTENTS)
    hin = BitField(0x0000, 30, 16, SPATIAL_EXTENTS)
    din = BitField(0x0004, 14, 0, SPATIAL_EXTENTS)
    common_in_fmt = BitField(0x0008, 1, 0)
    common_src2_in_fmt = BitField(0x0008, 3, 2)
    common_out_fmt = BitField(0x0008, 5, 4)
    cin = BitField(0x000C, 16, 0, CHANNEL_EXTENTS)
    cout = BitField(0x0010, 16, 0, CHANNEL_EXTENTS)
    wout = BitField(0x0014, 14, 0, SPATIAL_EXTENTS)
    hout = BitField(0x0014, 30, 16, SPATIAL_EXTENTS)
    dout = BitField(0x0018, 14, 0, SPATIAL_EXTENTS)
    kw = BitField(0x0020, 5, 0, KERNEL_EXTENTS)
    kh = BitField(0x0020, 11, 6, KERNEL_EXTENTS)
    sx = BitField(0x0020, 14, 13, CONV_STRIDES)
    sy = BitField(0x0020, 16, 15, CONV_STRIDES)
    pad_left = BitField(0x0020, 21, 17)
    pad_top = BitField(0x0020, 26, 22)
    out_offset_x = BitField(0x0020, 29, 28)
    out_offset_y = BitField(0x0020, 31, 30)
    ne_ocg_size = BitField(0x001C, 2, 0)
    num_groups = BitField(0x0028, 12, 0)
    unicast_en = BitField(0x0028, 14, 14)
    unicast_cin = BitField(0x0028, 31, 16)
    tile_height = BitField(0x002C, 14, 0)
    patch_width = BitField(0x0030, 3, 0)
    patch_height = BitField(0x0030, 7, 4)
    pe_src1_transpose = BitField(0x0034, 8, 8)
    pe_src2_transpose = BitField(0x0034, 9, 9)
    pe_output_cto_w = BitField(0x0034, 10, 10)
    source_routing = BitField(0x0038, 1, 0)
    task_type = BitField(0x0038, 6, 4, TaskType)
    ne_width_code = BitField(0x0038, 18, 16)
    ne_full_group = BitField(0x0038, 21, 21)
    source_transposed = BitField(0x0038, 2, 2)
    output_transpose = BitField(0x0038, 28, 28)
    input_transpose = BitField(0x0038, 29, 29)
    node_id = BitField(0x003C, 27, 20)


class L2(Block, base=0x0500, end=0x0564):
    """On-chip buffer addresses, strides and formats."""

    pe_src1_re_lu = BitField(0x0500, 0, 0)
    l2_reduce_compare = BitField(0x0500, 3, 2)
    pe_src2_re_lu = BitField(0x0500, 4, 4)

    l2_src_type = BitField(0x0504, 1, 0, L2Access)
    l2_src_dependent = BitField(0x0504, 3, 2)
    l2_src_alias_conv_src = BitField(0x0504, 4, 4)
    l2_src_alias_conv_rslt = BitField(0x0504, 5, 5)
    l2_src_dma_format = BitField(0x0504, 7, 6)
    l2_src_interleave = BitField(0x0504, 11, 8)
    l2_src_alias_planar_src = BitField(0x0504, 20, 20)
    l2_src_alias_planar_rslt = BitField(0x0504, 22, 22)
    l2_src_second_stream = BitField(0x0504, 24, 24)
    l2_src_compression = BitField(0x0504, 26, 25)
    l2_src_base_address = BitField(0x0508, 20, 4)
    l2_src_channel_stride = BitField(0x050C, 20, 4)
    l2_src_row_stride = BitField(0x0510, 20, 4)
    l2_src_depth_stride = BitField(0x0514, 20, 4)
    l2_src_group_stride = BitField(0x0518, 20, 4)
    l2_src2_base_address = BitField(0x051C, 20, 4)
    l2_src2_channel_stride = BitField(0x0520, 20, 4)
    l2_src2_row_stride = BitField(0x0524, 20, 4)
    l2_src2_depth_stride = BitField(0x0528, 20, 4)
    l2_src2_group_stride = BitField(0x052C, 20, 4)

    l2_result_type = BitField(0x0530, 1, 0, L2Access)
    l2_result_buffer_mode = BitField(0x0530, 3, 3)
    l2_result_alias_src = BitField(0x0530, 4, 4)
    l2_result_alias_rslt = BitField(0x0530, 5, 5)
    l2_result_dma_fmt = BitField(0x0530, 7, 6)
    l2_result_interleave = BitField(0x0530, 11, 8)
    l2_result_alias_planar_src = BitField(0x0530, 20, 20)
    l2_result_alias_planar_rslt = BitField(0x0530, 22, 22)
    l2_result_compression = BitField(0x0530, 26, 25)
    l2_result_base_addr = BitField(0x0534, 20, 4)
    l2_result_channel_stride = BitField(0x0538, 20, 4)
    l2_result_row_stride = BitField(0x053C, 20, 4)
    l2_result_depth_stride = BitField(0x0540, 20, 4)
    l2_result_group_stride = BitField(0x0544, 20, 4)


class PE(Block, base=0x0900, end=0x0914):
    """Planar engine element-wise, pooling and reduction fields."""

    pool_mode = BitField(0x0900, 1, 0, PoolMode)
    op = BitField(0x0900, 4, 2, PEOp)
    clamp_enable = BitField(0x0900, 5, 5)
    abs_enable = BitField(0x0900, 6, 6)
    nl_mode = BitField(0x0900, 13, 12, PENonlinear)
    first_source = BitField(0x0900, 16, 16)
    second_source = BitField(0x0900, 19, 18)
    bias = BitField(0x0904, 15, 0)
    scale = BitField(0x0904, 31, 16)
    reduction_epsilon = BitField(0x0908, 15, 0)
    pre_scale = BitField(0x0908, 31, 16)
    final_scale = BitField(0x090C, 31, 0)
    clamp_low = BitField(0x0910, 15, 0)
    clamp_high = BitField(0x0910, 31, 16)


class NE(Block, base=0x0D00, end=0x0D14):
    """Neural engine operation, kernel format and activation fields."""

    kernel_format = BitField(0x0D00, 1, 0, KernelFormat)
    kernel_sparse_fmt = BitField(0x0D00, 8, 8)
    op = BitField(0x0D04, 2, 0, NEOp)
    kernel_mode = BitField(0x0D04, 3, 3)
    ne_bias = BitField(0x0D04, 4, 4)
    ne_binary_point = BitField(0x0D04, 9, 8)
    ne_pipeline_code = BitField(0x0D04, 12, 10)
    arg_output_select = BitField(0x0D04, 23, 20)
    post_scale = BitField(0x0D10, 15, 0)
    ne_post_scale = BitField(0x0D04, 14, 14)
    nonlinear_mode = BitField(0x0D04, 17, 16, NENonlinear)
    min_pool_flag = BitField(0x0D04, 19, 19)


class TileDMASrc(Block, base=0x1100, end=0x11D4):
    """Tile DMA reads from unified memory."""

    tile_dma_src_hint = BitField(0x1100, 7, 0)
    tile_dma_src_data_set_id = BitField(0x1100, 15, 8)
    tile_dma_src_user_tag = BitField(0x1100, 23, 16)
    tile_dma_src_cache_hint = BitField(0x1108, 7, 0)
    tile_dma_src_base_addr_lo = BitField(0x1110, 31, 0)
    tile_dma_src_base_addr_hi = BitField(0x1114, 31, 0)
    tile_dma_src_row_stride = BitField(0x1118, 31, 6)
    tile_dma_src_channel_stride = BitField(0x111C, 31, 6)
    tile_dma_src_depth_stride = BitField(0x1120, 31, 6)
    tile_dma_src_group_stride = BitField(0x1124, 31, 6)
    tile_dma_src2_hint = BitField(0x1104, 7, 0)
    tile_dma_src2_data_set_id = BitField(0x1104, 15, 8)
    tile_dma_src2_user_tag = BitField(0x1104, 23, 16)
    tile_dma_src2_cache_hint = BitField(0x110C, 7, 0)
    tile_dma_src2_base_addr_lo = BitField(0x1128, 31, 0)
    tile_dma_src2_base_addr_hi = BitField(0x112C, 31, 0)
    tile_dma_src2_row_stride = BitField(0x1130, 31, 6)
    tile_dma_src2_channel_stride = BitField(0x1134, 31, 6)
    tile_dma_src2_format_mode = BitField(0x1154, 1, 0)
    tile_dma_src2_trunc = BitField(0x1154, 6, 4)
    tile_dma_src2_shift = BitField(0x1154, 11, 8)
    tile_dma_src2_memory_format = BitField(0x1154, 13, 12)
    tile_dma_src2_offset_ch = BitField(0x1154, 18, 16)
    tile_dma_src2_interleave = BitField(0x1154, 27, 24)
    tile_dma_src2_cmp_vec = BitField(0x1154, 31, 28)
    tile_dma_src_format_mode = BitField(0x1150, 1, 0)
    tile_dma_src_trunc = BitField(0x1150, 6, 4)
    tile_dma_src_shift = BitField(0x1150, 11, 8)
    tile_dma_src_memory_format = BitField(0x1150, 13, 12)
    tile_dma_src_offset_ch = BitField(0x1150, 18, 16)
    tile_dma_src_interleave = BitField(0x1150, 27, 24)
    tile_dma_src_cmp_vec = BitField(0x1150, 31, 28)


class TileDMADst(Block, base=0x1500, end=0x1528):
    """Tile DMA writes to unified memory."""

    tile_dma_dst_enable = BitField(0x1500, 0, 0)
    tile_dma_dst_hint = BitField(0x1500, 7, 4)
    tile_dma_dst_data_set_id = BitField(0x1500, 15, 8)
    tile_dma_dst_user_tag = BitField(0x1500, 23, 16)
    tile_dma_dst_fifo_mode = BitField(0x1500, 24, 24)
    tile_dma_dst_base_addr_lo = BitField(0x1508, 31, 0)
    tile_dma_dst_base_addr_hi = BitField(0x150C, 31, 0)
    tile_dma_dst_row_stride = BitField(0x1510, 31, 6)
    tile_dma_dst_channel_stride = BitField(0x1514, 31, 6)
    tile_dma_dst_depth_stride = BitField(0x1518, 31, 6)
    tile_dma_dst_group_stride = BitField(0x151C, 31, 6)
    tile_dma_dst_format_mode = BitField(0x1520, 1, 0)
    tile_dma_dst_trunc = BitField(0x1520, 6, 4)
    tile_dma_dst_shift = BitField(0x1520, 11, 8)
    tile_dma_dst_memory_format = BitField(0x1520, 13, 12)
    tile_dma_dst_offset_ch = BitField(0x1520, 18, 16)
    tile_dma_dst_zero_pad_first = BitField(0x1520, 20, 20)
    tile_dma_dst_zero_pad_last = BitField(0x1520, 21, 21)
    tile_dma_dst_interleave = BitField(0x1520, 27, 24)
    tile_dma_dst_cmp_vec = BitField(0x1520, 31, 28)
    tile_dma_dst_marker = BitField(0x1524, 15, 0)


class KernelDMA(Block, base=0x1900, end=0x1A18):
    """Kernel DMA coefficient streams and auxiliary tables."""

    kernel_dma_src_enable = BitField(0x1900, 6, 6)
    kernel_dma_armed = BitField(0x1900, 7, 7)
    kernel_dma_coeff_present = BitField(0x1900, 9, 9)
    kernel_dma_coeff_tag = BitField(0x1900, 23, 16)
    kernel_stride_registers = (BitField(0x1910, 31, 6), BitField(0x1914, 31, 6))
    kernel_dma_src_coeff_dma_en = tuple(BitField(addr, 0, 0) for addr in range(0x1918, 0x1958, 4))
    kernel_dma_src_coeff_loaded = tuple(BitField(addr, 5, 5) for addr in range(0x1918, 0x1958, 4))
    kernel_dma_src_coeff_tag = tuple(BitField(addr, 23, 16) for addr in range(0x1918, 0x1958, 4))
    kernel_dma_src_coeff_base_offset = tuple(
        BitField(addr, 31, 6) for addr in range(0x1958, 0x1998, 4)
    )
    kernel_dma_src_coeff_mem_buffer_size = tuple(
        BitField(addr, 31, 6) for addr in range(0x1998, 0x19D8, 4)
    )
    nonlinear_lut_enable = BitField(0x19E4, 0, 0)
    nonlinear_lut_size = BitField(0x1A14, 31, 6)


CONFIG_BLOCKS = (Common, L2, PE, NE, TileDMASrc, TileDMADst, KernelDMA)


def block_at(addr: int) -> type[Block] | None:
    """Find the block containing an aligned configuration word address."""
    if not isinstance(addr, int) or addr % 4:
        return None
    return next((block for block in CONFIG_BLOCKS if block.base <= addr < block.end), None)


__all__ = [
    "CONFIG_BLOCKS",
    "BitField",
    "Block",
    "Common",
    "KernelDMA",
    "KernelFormat",
    "L2",
    "NE",
    "NENonlinear",
    "NEOp",
    "PE",
    "PENonlinear",
    "PEOp",
    "PoolMode",
    "TaskType",
    "TileDMADst",
    "TileDMASrc",
    "block_at",
    "check_word",
]
