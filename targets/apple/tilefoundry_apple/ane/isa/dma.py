"""Decode verified tile transfers and their register-only memory relationships."""

from collections.abc import Mapping
from dataclasses import dataclass

from . import fields
from .arith import align_up

L2_CAPACITY = 2 * 1024 * 1024
PADDED_CONV = (3, 3, 1, 1, 1, 1)


def value(image: Mapping[int, int], field: fields.BitField) -> int:
    """Read a field from an effective register image."""
    return field.get(image.get(field.addr, 0))


@dataclass(frozen=True)
class Transfer:
    """One fp16 feature map; a nonzero register identifies an active DRAM transfer."""

    name: str
    register: int
    channels: int
    height: int
    width: int
    row: int
    channel: int

    @property
    def end(self) -> int:
        """Exclusive end of the effective elements relative to the DMA base."""
        return (self.channels - 1) * self.channel + (self.height - 1) * self.row + self.width * 2


def geometry(image: Mapping[int, int]) -> tuple[Transfer, ...]:
    """Decode active readers and writers, refusing modes with undetermined footprints.

    A descriptor without DMA that programs L2 pitches keeps its source and result in
    L2; the common block alone gives their geometry. A zero-padded convolution encodes
    only its top and left padding and its source DMA reads the unpadded map; the result
    geometry implies equal bottom and right padding. ``PADDED_CONV`` is the one decoded
    padded window: kernel, stride and padding as (kh, kw, sy, sx, top, left). A pool
    pads the same way; its emitter limits the windows it admits.
    """
    common, l2 = fields.Common, fields.L2
    access = value(image, l2.l2_src_type)
    second = value(image, l2.l2_src_second_stream)
    enabled = value(image, fields.TileDMADst.tile_dma_dst_enable)
    pitched = any(value(image, field) for field in (l2.l2_src_channel_stride, l2.l2_src_row_stride,
                                                   l2.l2_result_channel_stride, l2.l2_result_row_stride))
    if not (access or second or enabled or pitched):
        return ()
    task = value(image, common.task_type)
    result_access = value(image, l2.l2_result_type)
    continuation = access == fields.L2Access.L2_READ or result_access == fields.L2Access.L2_READ
    binary = task == fields.TaskType.PE_EW_BINARY
    resident_binary = binary and access == fields.L2Access.L2_READ and not second
    if continuation:
        if (access not in (fields.L2Access.DMA_READ, fields.L2Access.L2_READ)
                or result_access not in (fields.L2Access.DMA_READ, fields.L2Access.L2_READ)
                or enabled != (result_access == fields.L2Access.DMA_READ)
                or task not in (fields.TaskType.PE_EW_UNARY, fields.TaskType.NE) and not resident_binary):
            raise ValueError("L2 continuation access mapping is not decoded yet")
    elif access != fields.L2Access.DMA_READ or not enabled:
        raise ValueError("tile DMA access mapping is not decoded yet")
    for field in (common.pe_src1_transpose, common.pe_src2_transpose,
                  common.pe_output_cto_w, common.small_source_mode, l2.l2_src_dependent):
        if value(image, field):
            raise ValueError(f"tile DMA mode at {field.addr:#x}[{field.hi}:{field.lo}] is not decoded yet")
    if (result_access == fields.L2Access.DMA_READ and not value(image, l2.l2_result_buffer_mode)
            or result_access == fields.L2Access.L2_READ and value(image, l2.l2_result_buffer_mode)):
        raise ValueError("L2 result DMA mapping is not decoded yet")
    if task not in (fields.TaskType.NE, fields.TaskType.PE_POOL_UNARY,
                    fields.TaskType.PE_EW_UNARY, fields.TaskType.PE_EW_BINARY,
                    fields.TaskType.PE_EW_REDUCE_UNARY):
        raise ValueError(f"task_type {task} DMA geometry is not decoded yet")
    if second != (binary and not resident_binary):
        raise ValueError("second-stream task mapping is not decoded yet")
    if value(image, common.source_routing) != (3 if second else 1):
        raise ValueError("source_routing DMA mapping is not decoded yet")
    if binary and value(image, fields.PE.second_source) not in (2, 3):
        raise ValueError("PE second_source DMA mapping is not decoded yet")
    for field in (common.din, common.dout):
        if value(image, field) != 1:
            raise ValueError("tile DMA depth other than one is not decoded yet")
    for field in (common.common_in_fmt, common.common_src2_in_fmt, common.common_out_fmt):
        if value(image, field) != 2:
            raise ValueError("tile DMA common format is not decoded yet")
    source = tuple(value(image, field) for field in (common.cin, common.hin, common.win))
    result = tuple(value(image, field) for field in (common.cout, common.hout, common.wout))
    if min(*source, *result) <= 0:
        raise ValueError("tile DMA geometry must have positive channel, height and width")
    tile = value(image, common.tile_height)
    if not 1 <= tile <= result[1]:
        raise ValueError(f"tile_height {tile} must be within [1, {result[1]}]")
    if continuation and (tile != source[1] or source != result):
        raise ValueError("partial or unequal L2 continuation geometry is not decoded yet")
    transpose = value(image, common.output_transpose)
    if not transpose and value(image, common.fill_lower_nodes_first):
        raise ValueError("fill_lower_nodes_first DMA mapping is not decoded yet")
    if transpose:
        if (task != fields.TaskType.NE or value(image, fields.NE.op) != fields.NEOp.BYPASS
                or source != result or access != fields.L2Access.DMA_READ):
            raise ValueError("output transpose DMA mapping is not decoded yet")
        result = result[2], result[1], result[0]
    elif task in (fields.TaskType.PE_EW_UNARY, fields.TaskType.PE_EW_BINARY,
                  fields.TaskType.PE_EW_REDUCE_UNARY) and source != result:
        raise ValueError("elementwise and reduction declared output geometry must equal input")
    if task == fields.TaskType.PE_EW_REDUCE_UNARY:
        result = result[0], 1, 1
    if task in (fields.TaskType.PE_POOL_UNARY, fields.TaskType.NE):
        neural_op = value(image, fields.NE.op)
        if task == fields.TaskType.NE and neural_op not in (fields.NEOp.CONV, fields.NEOp.BYPASS):
            raise ValueError("neural operation DMA geometry is not decoded yet")
        if task == fields.TaskType.NE and neural_op == fields.NEOp.BYPASS:
            if (value(image, common.num_groups) != 1 or not value(image, common.unicast_en)
                    or not transpose and value(image, common.unicast_cin) != 1):
                raise ValueError("bypass unicast DMA mapping is not decoded yet")
        if task == fields.TaskType.PE_POOL_UNARY and source[0] != result[0]:
            raise ValueError("pool output channels must equal input channels")
        if task == fields.TaskType.PE_POOL_UNARY or neural_op == fields.NEOp.CONV:
            if (value(image, common.out_offset_x), value(image, common.out_offset_y)) != (1, 1):
                raise ValueError("kernel output offset DMA mapping is not decoded yet")
            kw, kh, sx, sy = (value(image, f) for f in (common.kw, common.kh, common.sx, common.sy))
            top, left = value(image, common.pad_top), value(image, common.pad_left)
            if (top or left) and task != fields.TaskType.PE_POOL_UNARY and (
                task != fields.TaskType.NE or (kh, kw, sy, sx, top, left) != PADDED_CONV
            ):
                raise ValueError("padded DMA footprint is not decoded yet")
            if min(kw, kh, sx, sy) <= 0:
                raise ValueError("kernel window and strides must be positive")
            if result[1:] != ((source[1] + 2 * top - kh) // sy + 1, (source[2] + 2 * left - kw) // sx + 1):
                raise ValueError("output geometry contradicts the kernel window, strides and padding")
        elif not transpose and source != result:
            raise ValueError("bypass output geometry must equal input")
    decoded = [
        decode(image, "src", source) if access == fields.L2Access.DMA_READ else staged("src", source),
        decode(image, "dst", result) if enabled else staged("dst", result),
    ]
    if binary:
        decoded.insert(1, decode(image, "src2", source) if second else staged("src2", source))
    return tuple(decoded)


def staged(name: str, shape: tuple[int, int, int]) -> Transfer:
    """Describe an L2-only feature map without declaring any DRAM access."""
    channels, height, width = shape
    row = align_up(width * 2, 64)
    return Transfer(name, 0, channels, height, width, row, height * row)


def transfers(image: Mapping[int, int]) -> tuple[Transfer, ...]:
    """Return only enabled DRAM transfers after validating the complete geometry."""
    return tuple(transfer for transfer in geometry(image) if transfer.register)


def decode(image: Mapping[int, int], name: str, shape: tuple[int, int, int]) -> Transfer:
    """Decode the verified tiled fp16 format and check non-overlapping DMA strides."""
    block = fields.TileDMADst if name == "dst" else fields.TileDMASrc
    prefix = f"tile_dma_{name}_"
    for suffix, expected in (("format_mode", 1), ("trunc", 3), ("shift", 0),
                             ("memory_format", 2), ("offset_ch", 0), ("interleave", 1),
                             ("cmp_vec", 0)):
        field = block.fields[prefix + suffix]
        if value(image, field) != expected:
            raise ValueError(f"{prefix}{suffix}={value(image, field)} is not decoded yet")
    if name == "dst":
        for suffix in ("zero_pad_first", "zero_pad_last", "fifo_mode"):
            if value(image, block.fields[prefix + suffix]) != 1:
                raise ValueError(f"{prefix}{suffix} mode is not decoded yet")
    if name != "src2":
        for suffix in ("depth_stride", "group_stride"):
            if value(image, block.fields[prefix + suffix]):
                raise ValueError(f"{prefix}{suffix} mapping is not decoded yet")
    channels, height, width = shape
    row = value(image, block.fields[prefix + "row_stride"]) * 64
    if not row and min(shape) == 1:
        row = align_up(width * 2, 64)
    channel = value(image, block.fields[prefix + "channel_stride"]) * 64
    if row < width * 2:
        raise ValueError(f"{prefix}row_stride {row} does not cover width {width} fp16 elements")
    if channel < height * row:
        raise ValueError(f"{prefix}channel_stride {channel} does not cover a {height * row} byte plane")
    return Transfer(name, block.fields[prefix + "base_addr_lo"].addr,
                    channels, height, width, row, channel)


def check_l2(image: Mapping[int, int], decoded: tuple[Transfer, ...]) -> tuple[tuple[str, int, int], ...]:
    """Check staged pitches and disjoint live intervals within L2 capacity.

    A staged channel spans its 16-byte aligned row and a staged row spans every channel,
    so any row stride that covers that depth is valid. A DMA result that leaves its row
    stride zero occupies the depth, or one source row for a reduction. A planar unary
    that reads its source by DMA and keeps its result in L2 may write the result over
    exactly the interval its source lands in.
    """
    if not decoded:
        return ()
    common, l2 = fields.Common, fields.L2
    task = value(image, common.task_type)
    tile = value(image, common.tile_height)
    source_rows = tile
    if task == fields.TaskType.PE_POOL_UNARY:
        source_rows = (tile - 1) * value(image, common.sy) + value(image, common.kh)
    intervals = []
    source_row = 0
    for transfer in decoded:
        name = "result" if transfer.name == "dst" else transfer.name
        prefix = f"l2_{name}_"
        base_field = l2.l2_result_address if name == "result" else l2.fields[prefix + "base_address"]
        base = value(image, base_field) * 16
        rows = min(tile if name == "result" else source_rows, transfer.height)
        channel = align_up(transfer.width * 2, 16)
        depth = transfer.channels * channel
        row = value(image, l2.fields[prefix + "row_stride"]) * 16
        if name == "result" and not row and transfer.register:
            row = source_row if task == fields.TaskType.PE_EW_REDUCE_UNARY else depth
        if row < depth:
            raise ValueError(f"{prefix}row_stride {row} does not cover the {depth} byte staged depth")
        for suffix, expected in (("channel_stride", channel), ("depth_stride", depth), ("group_stride", depth)):
            actual = value(image, l2.fields[prefix + suffix]) * 16
            if name == "result" and not actual:
                if not transfer.register and suffix == "channel_stride":
                    raise ValueError("l2_result_channel_stride must explicitly match the allocated L2 layout")
                continue
            if name != "result" and not transfer.register and suffix != "channel_stride":
                expected = 0
            if actual != expected:
                raise ValueError(f"{prefix}{suffix} {actual} contradicts the staged layout; expected {expected}")
        for suffix, expected in (("interleave", 1), ("compression", 0),
                                 ("dma_fmt" if name == "result" else "dma_format", 1)):
            if name == "src" and not transfer.register and suffix == "dma_format":
                expected = 0
            if name != "src2" and value(image, l2.fields[prefix + suffix]) != expected:
                raise ValueError(f"{prefix}{suffix} mode is not decoded yet")
        if name == "src":
            source_row = row
        end = base + rows * row
        if end > L2_CAPACITY:
            raise ValueError(f"L2 {name} range [{base}, {end}) exceeds capacity {L2_CAPACITY}")
        intervals.append((name, base, end))
    in_place = task == fields.TaskType.PE_EW_UNARY and decoded[0].register and not decoded[-1].register
    for index, (name, start, end) in enumerate(intervals):
        for other, low, high in intervals[:index]:
            if start < high and low < end and not (in_place and (low, high) == (start, end)):
                raise ValueError(f"L2 {other} and {name} ranges overlap")
    return tuple(intervals)


def check_continuations(images) -> None:
    """Require each L2 reader to match an unmodified earlier result buffer.

    Equal intervals over equal complete-height shapes give the reader the writer's pitches.
    """
    live = []
    for image in images:
        decoded = geometry(image)
        intervals = check_l2(image, decoded)
        if not intervals:
            continue
        result = decoded[-1]
        if value(image, fields.L2.l2_src_type) == fields.L2Access.L2_READ:
            for source, (_, low, high) in zip(decoded[:-1], intervals):
                if (low, high, (source.channels, source.height, source.width)) not in live:
                    raise ValueError("L2 source requires a matching unmodified preceding result")
        writes = [interval for transfer, interval in zip(decoded, intervals)
                  if transfer.register or transfer is result]
        for _, low, high in writes:
            live = [item for item in live if item[1] <= low or high <= item[0]]
        if value(image, fields.L2.l2_result_type) == fields.L2Access.L2_READ:
            live.append((*intervals[-1][1:], (result.channels, result.height, result.width)))


def check_coefficient_lanes(lanes: int, enabled_slots: int) -> None:
    """Require equal packed-lane and enabled-slot counts in a coefficient stream."""
    if lanes != enabled_slots:
        raise ValueError(f"coefficient lane count {lanes} with {enabled_slots} DMA slots is not decoded yet")


__all__ = ["L2_CAPACITY", "Transfer", "check_coefficient_lanes", "check_continuations", "check_l2", "geometry", "transfers"]
