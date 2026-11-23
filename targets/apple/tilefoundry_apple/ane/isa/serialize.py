"""H14 packet encoding and canonical configuration word grouping."""

import struct
from collections.abc import Mapping
from dataclasses import dataclass

from .fields import CONFIG_BLOCKS, block_at, check_word

WORD_BYTES = struct.calcsize("<I")
ADDR_MASK = (1 << 15) - 1
RUN_MAX = 64
MASK_WIDTH = 16
MASK_TAG = 1 << 31
TASK_ALIGN_GRANULE = 16
DESCRIPTOR_ALIGN = 64
RELOCATION_SLOTS = {0x1110: 0x114, 0x1128: 0x118, 0x1508: 0x110}
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


def serialize(image: Mapping[int, int]) -> list[Packet]:
    """Group words by block, choosing RUN on a tie with the MASK window."""
    for addr, value in image.items():
        if block_at(addr) is None:
            raise ValueError(f"{addr!r} is not a configuration word address")
        check_word(value)

    packets: list[Packet] = []
    slotted = {addr: flag for addr, flag in RELOCATION_SLOTS.items() if addr in image}
    paired = set(slotted) | {addr + WORD_BYTES for addr in slotted}
    pairs = [
        Packet("run", ((addr, image[addr]), (addr + WORD_BYTES, image[addr + WORD_BYTES])), flag)
        for addr, flag in sorted(slotted.items())
    ]
    for block in CONFIG_BLOCKS:
        pending = sorted(
            addr for addr in image if block.base <= addr < block.end and addr not in paired
        )
        while pending:
            start = pending[0]
            run = 1
            while run < min(len(pending), RUN_MAX) and pending[run] == start + run * WORD_BYTES:
                run += 1
            window = 1
            while window < len(pending) and pending[window] <= start + MASK_WIDTH * WORD_BYTES:
                window += 1
            kind, count = ("run", run) if run >= window else ("mask", window)
            packets.append(Packet(kind, tuple((addr, image[addr]) for addr in pending[:count])))
            pending = pending[count:]
    packets.extend(pairs)
    return packets


def sentinel_words(end: int) -> int:
    """Count the padding sentinels a descriptor appends when it ends at ``end``.

    The encoder aligns the end of a descriptor up to ``TASK_ALIGN_GRANULE``; only
    a descriptor that lands on a ``DESCRIPTOR_ALIGN`` boundary is padded at all,
    and the padding is one or two 8-byte steps.  Each step contributes exactly
    one sentinel word.
    """
    aligned = (end + TASK_ALIGN_GRANULE - 1) & ~(TASK_ALIGN_GRANULE - 1)
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
    "sentinel_words",
    "serialize",
]
