"""H14 packet encoding and canonical configuration word grouping."""

import struct
from collections.abc import Mapping
from dataclasses import dataclass

from .fields import (
    L2,
    NE,
    PE,
    Common,
    KernelDMA,
    TileDMADst,
    TileDMASrc,
    block_at,
    check_word,
)

WORD_BYTES = struct.calcsize("<I")
ADDR_MASK = (1 << 15) - 1
RUN_MAX = 64
MASK_WIDTH = 16
MASK_TAG = 1 << 31
TASK_ALIGN_GRANULE = 16
DESCRIPTOR_ALIGN = 64
RELOCATION_SLOTS = {
    0x1110: 0x114,
    0x1128: 0x118,
    0x1508: 0x110,
    0x1908: 0x118,
    0x1A00: 0x118,
}
RELOCATION_SINGLES = {0x19F4: 0x118}
"""Base-address word pairs that are patched after emission, keyed by their low word.

Each relocated base address is a 64-bit value in two adjacent words.  The pair
travels as one flagged run packet of its own, after the ordinary words and after
the padding sentinels, so the patcher can rewrite both halves without disturbing
a neighbouring packet.  The flag names the relocation slot the pair belongs to.
"""
MARKER_ADDR = 0x1524
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
        start = addrs[0]
        base = start // WORD_BYTES
        if self.kind == "run":
            count = len(addrs)
            if count > RUN_MAX or addrs != list(
                range(start, start + WORD_BYTES * count, WORD_BYTES)
            ):
                raise ValueError("RUN requires 1..64 consecutive words")
            if not isinstance(self.flags, int) or not 0 <= self.flags < (1 << 10):
                raise ValueError("RUN flags do not fit 10 bits")
            head = (self.flags << 21) | ((count - 1) << 15) | base
        elif self.kind == "mask":
            offsets = [addr // WORD_BYTES - base for addr in addrs[1:]]
            if (
                self.flags
                or offsets != sorted(set(offsets))
                or any(not 1 <= offset <= MASK_WIDTH for offset in offsets)
            ):
                raise ValueError("MASK requires ascending writes in its window and has no flags")
            mask = sum(1 << (offset - 1) for offset in offsets)
            head = MASK_TAG | (mask << 15) | base
        else:
            raise ValueError(f"unknown packet kind {self.kind!r}")
        for _, value in self.writes:
            check_word(value)
        return (head, *(value for _, value in self.writes))

    @classmethod
    def from_words(cls, words: tuple[int, ...], off: int) -> tuple["Packet", int]:
        """Decode a packet and return the index after its last value."""
        if not 0 <= off < len(words):
            raise ValueError(f"missing packet header at word {off}")
        head = words[off]
        base = head & ADDR_MASK
        if head & MASK_TAG:
            addrs = [base, *(base + 1 + bit for bit in range(MASK_WIDTH) if head >> (15 + bit) & 1)]
            kind, flags = "mask", 0
        else:
            count = ((head >> 15) & (RUN_MAX - 1)) + 1
            addrs = range(base, base + count)
            kind, flags = "run", head >> 21
        end = off + 1 + len(addrs)
        if end > len(words) or addrs[-1] > ADDR_MASK:
            raise ValueError(f"invalid packet at word {off}: payload or address range overrun")
        return cls(
            kind,
            tuple((WORD_BYTES * addr, value) for addr, value in zip(addrs, words[off + 1 : end])),
            flags,
        ), end


#: The flagged address packets follow the pipeline too: coefficients first, then
#: the source surfaces, then the destination.
EXTENDED_ORDER = (0x1908, 0x19F8, 0x1A00, 0x1110, 0x1128, 0x1508)


def serialize(image: Mapping[int, int]) -> list[Packet]:
    """Group words by block, choosing RUN on a tie with the MASK window."""
    for addr, value in image.items():
        if block_at(addr) is None:
            raise ValueError(f"{addr!r} is not a configuration word address")
        check_word(value)

    packets: list[Packet] = []
    slotted = {addr: flag for addr, flag in RELOCATION_SLOTS.items() if addr in image}
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
        packets.extend(_block_packets(pending, image))
    packets.extend(pairs)
    packets.extend(
        Packet("run", ((addr, image[addr]),), flag) for addr, flag in sorted(singles.items())
    )
    return packets


#: The blocks are emitted in pipeline order, not address order: coefficients
#: first, then the shared configuration, the source DMA and its trigger, the
#: L2 read setup, the compute engine (PE or NE), and the destination DMA.
STAGE_ORDER = (KernelDMA, Common, TileDMASrc, L2, PE, NE, TileDMADst)


def _block_packets(pending: list[int], image: Mapping[int, int]) -> list[Packet]:
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
    "STAGE_ORDER",
    "sentinel_packets",
    "serialize",
]
