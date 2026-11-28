"""H14 packet encoding and canonical configuration word grouping."""

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .fields import (
    COEFFICIENT_RELOCATIONS,
    CONFIG_BLOCKS,
    BitField,
    KernelDMA,
    TileDMADst,
    TileDMASrc,
    block_at,
    check_word,
)

WORD_BYTES = struct.calcsize("<I")
TASK_ALIGN_GRANULE = 16
DESCRIPTOR_ALIGN = 64
PKT_BASE = BitField(0, 14, 0)
PKT_COUNT = BitField(0, 20, 15)
PKT_FLAGS = BitField(0, 30, 21)
PKT_WINDOW = BitField(0, 30, 15)
PKT_TAG = BitField(0, 31, 31)
ADDR_MASK = PKT_BASE.mask
RUN_MAX = (PKT_COUNT.mask >> PKT_COUNT.lo) + 1
MASK_WIDTH = PKT_WINDOW.hi - PKT_WINDOW.lo + 1
RELOCATION_SLOTS = MappingProxyType(
    {
        TileDMASrc.tile_dma_src_base_addr_lo.addr: 0x114,
        TileDMASrc.tile_dma_src2_base_addr_lo.addr: 0x118,
        TileDMADst.tile_dma_dst_base_addr_lo.addr: 0x110,
        KernelDMA.kernel_dma_src_base_lo.addr: 0x118,
        KernelDMA.kernel_dma_aux_base_lo.addr: 0x118,
    }
)
RELOCATION_SINGLES = MappingProxyType({KernelDMA.kernel_dma_lut_base.addr: 0x118})
IO_RELOCATION = 0x110
IO_RELOCATION_STEP = 4
EXTENDED_ORDER = (
    *(addr for addr, width in COEFFICIENT_RELOCATIONS.items() if width == 8),
    TileDMASrc.tile_dma_src_base_addr_lo.addr,
    TileDMASrc.tile_dma_src2_base_addr_lo.addr,
    TileDMADst.tile_dma_dst_base_addr_lo.addr,
)
MARKER_ADDR = TileDMADst.tile_dma_dst_marker.addr
PAD_SENTINEL = 0xDEAD


@dataclass(frozen=True)
class Packet:
    """A RUN or MASK of writes in ascending byte address order."""

    kind: str
    writes: tuple[tuple[int, int], ...]
    flags: int = 0

    def words(self) -> tuple[int, ...]:
        """Encode the packet header followed by its values."""
        if not self.writes:
            raise ValueError("empty packet")
        addrs = [addr for addr, _ in self.writes]
        if any(
            not isinstance(addr, int)
            or addr % WORD_BYTES
            or not 0 <= addr // WORD_BYTES <= ADDR_MASK
            for addr in addrs
        ):
            raise ValueError("packet address outside the word address space")
        if block_at(addrs[0]) is None or any(block_at(addr) is not block_at(addrs[0]) for addr in addrs):
            raise ValueError("packet addresses must lie inside one declared configuration block")
        start = addrs[0]
        base = PKT_BASE.put(0, start // WORD_BYTES)
        if self.kind == "run":
            count = len(addrs)
            if count > RUN_MAX or addrs != list(
                range(start, start + WORD_BYTES * count, WORD_BYTES)
            ):
                raise ValueError(f"RUN requires 1..{RUN_MAX} consecutive words")
            head = PKT_FLAGS.put(PKT_COUNT.put(base, count - 1), self.flags)
        elif self.kind == "mask":
            offsets = [addr // WORD_BYTES - start // WORD_BYTES for addr in addrs[1:]]
            if (
                self.flags
                or offsets != sorted(set(offsets))
                or any(not 1 <= offset <= MASK_WIDTH for offset in offsets)
            ):
                raise ValueError("MASK requires ascending writes in its window and has no flags")
            window = sum(1 << (offset - 1) for offset in offsets)
            head = PKT_TAG.put(PKT_WINDOW.put(base, window), 1)
        else:
            raise ValueError(f"unknown packet kind {self.kind!r}")
        for _, value in self.writes:
            check_word(value)
        return (head, *(value for _, value in self.writes))


def io_relocation(segment: int) -> int:
    """The relocation flags that bind a base to the I/O segment at index *segment*.

    The flags hold the segment's memory id: the first I/O segment, which is the
    first output, has id 4 and each later segment the next id.
    """
    return IO_RELOCATION + IO_RELOCATION_STEP * segment


def serialize(
    image: Mapping[int, int], relocations: Mapping[int, int] = RELOCATION_SLOTS
) -> list[Packet]:
    """Group words by block, choosing RUN on a tie with the MASK window.

    *relocations* gives the flags of each base pair emitted as an extended packet.
    """
    for addr, value in image.items():
        if block_at(addr) is None:
            raise ValueError(f"{addr!r} is not a configuration word address")
        check_word(value)

    packets: list[Packet] = []
    slotted = {addr: flag for addr, flag in relocations.items() if addr in image}
    singles = {addr: flag for addr, flag in RELOCATION_SINGLES.items() if addr in image}
    paired = set(slotted) | {addr + WORD_BYTES for addr in slotted} | set(singles)
    pairs = [
        Packet("run", ((addr, image[addr]), (addr + WORD_BYTES, image[addr + WORD_BYTES])), flag)
        for addr, flag in sorted(slotted.items(), key=lambda item: EXTENDED_ORDER.index(item[0])
                                 if item[0] in EXTENDED_ORDER else len(EXTENDED_ORDER))
    ]
    for block in STAGE_ORDER:
        pending = sorted(
            addr for addr in image if block.base <= addr < block.end and addr not in paired
        )
        packets.extend(block_packets(pending, image))
    packets.extend(pairs)
    packets.extend(
        Packet("run", ((addr, image[addr]),), flag) for addr, flag in sorted(singles.items())
    )
    return packets


STAGE_ORDER = tuple(sorted(CONFIG_BLOCKS, key=lambda block: block.pipeline_rank))


def block_packets(pending: list[int], image: Mapping[int, int]) -> list[Packet]:
    """Pack one block's words into RUN and MASK packets.

    A window that holds every pending word becomes a single MASK unless the whole
    block is contiguous, in which case it becomes one RUN.  When the words span
    more than one window, a leading run of two or more words is emitted as a RUN
    so the remainder fits one MASK; a window holding a single word is emitted as
    a RUN too, because a MASK with no offsets carries no information.
    """
    out: list[Packet] = []
    while pending:
        start = pending[0]
        run = 1
        while (
            run < min(len(pending), RUN_MAX)
            and pending[run] == start + run * WORD_BYTES
        ):
            run += 1
        window = 1
        while window < len(pending) and pending[window] <= start + MASK_WIDTH * WORD_BYTES:
            window += 1
        if window == len(pending):
            kind, count = ("run", run) if run == len(pending) else ("mask", window)
        else:
            kind, count = ("run", run) if (run > 1 or window == 1) else ("mask", window)
        out.append(Packet(kind, tuple((addr, image[addr]) for addr in pending[:count])))
        pending = pending[count:]
    return out


def sentinel_packets(cursor: int, task_bytes: int, *, alignment: int = TASK_ALIGN_GRANULE) -> int:
    """Count the marker packets a descriptor appends when it starts at ``cursor``.

    A descriptor that ends exactly on a ``DESCRIPTOR_ALIGN`` boundary after being
    aligned to ``alignment`` is padded instead of left alone: an 8-byte pad (one
    marker packet of two words) when it was less than 8 bytes short of that
    boundary, 16 bytes (two packets) when it was less than 16 short, and nothing
    at all once it is 16 or more bytes short.
    """
    end = cursor + task_bytes
    aligned = (end + alignment - 1) & ~(alignment - 1)
    if aligned % DESCRIPTOR_ALIGN:
        return 0
    pad = aligned - end
    applied = 8 if pad < 8 else (16 if pad < 16 else 0)
    return applied // 8


__all__ = [
    "DESCRIPTOR_ALIGN",
    "PAD_SENTINEL",
    "MARKER_ADDR",
    "RELOCATION_SLOTS",
    "TASK_ALIGN_GRANULE",
    "WORD_BYTES",
    "Packet",
    "io_relocation",
    "STAGE_ORDER",
    "sentinel_packets",
    "serialize",
]
