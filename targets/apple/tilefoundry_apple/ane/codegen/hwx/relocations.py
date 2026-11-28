"""H14 HWX relocations encoding."""

from collections.abc import Mapping
from dataclasses import dataclass

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa import arith

from .macho import RELOCATION

DMA_BASES = (
    isa.TileDMASrc.tile_dma_src_base_addr_lo.addr,
    isa.TileDMASrc.tile_dma_src2_base_addr_lo.addr,
    isa.TileDMADst.tile_dma_dst_base_addr_lo.addr,
)


@dataclass(frozen=True)
class Reloc:
    """A 64-bit placeholder at a word offset from the start of the TD stream."""

    word_offset: int
    symbol: str


@dataclass(frozen=True)
class CoefficientReloc:
    """An encoded coefficient slot bound to one planned coefficient block."""

    word_offset: int
    coefficient: int


def packet_slots(program: isa.Program):
    """Yield each packet of the encoded stream and the word offset of its first payload."""
    start = len(isa.PREAMBLE) // 4
    for index, descriptor in enumerate(program.tds):
        first = start + descriptor.header_size_words()
        for packet in descriptor.encoded_packets(start * 4):
            yield index, first + 1, packet
            first += 1 + len(packet.writes)
        start = arith.align_pow2(start + descriptor.encoded_words(start * 4), 2)


def base_slots(program: isa.Program) -> dict[int, tuple[int, int, int]]:
    """Map encoded base slots to their descriptor, register and byte addend."""
    widths = {
        **dict.fromkeys(DMA_BASES, 8),
        **isa.COEFFICIENT_RELOCATIONS,
    }
    slots = {}
    for td_index, first, packet in packet_slots(program):
        packet.words()
        addresses = [addr for addr, _ in packet.writes]
        for low, width in widths.items():
            if width == 8 and low + 4 in addresses and low not in addresses:
                raise ValueError(f"base register {low:#x} has an orphan high word")
        if (packet.flags and addresses[0] not in DMA_BASES
                and packet.flags != isa.RELOCATION_SLOTS.get(addresses[0])):
            raise ValueError("relocation packet flags disagree with its base register")
        for index, (addr, value) in enumerate(packet.writes):
            if addr in DMA_BASES and (not packet.flags or len(packet.writes) != 2):
                raise ValueError(f"DMA base register {addr:#x} requires its relocation packet flags")
            if addr not in widths:
                continue
            if widths[addr] == 8:
                if index + 1 == len(packet.writes) or packet.writes[index + 1][0] != addr + 4:
                    raise ValueError(f"base register {addr:#x} needs its adjacent high word")
                value |= packet.writes[index + 1][1] << 32
            slots[first + index] = td_index, addr, value
    return slots


def coefficient_relocations(
    program: isa.Program, consumers: Mapping[int, int]
) -> tuple[CoefficientReloc, ...]:
    """Bind every encoded coefficient slot to its consuming descriptor's block."""
    slots = base_slots(program)
    relocs = tuple(CoefficientReloc(word, consumers[index])
                   for word, (index, addr, _) in slots.items()
                   if addr in isa.COEFFICIENT_RELOCATIONS and index in consumers)
    expected = {word for word, (_, addr, _) in slots.items() if addr in isa.COEFFICIENT_RELOCATIONS}
    seen = {slots[reloc.word_offset][0] for reloc in relocs}
    if {reloc.word_offset for reloc in relocs} != expected or seen != set(consumers):
        raise ValueError("coefficient slots and weight consumers disagree")
    return relocs


def relocation_table(
    program: isa.Program, relocs: tuple[CoefficientReloc, ...], section: int
) -> bytes:
    """Encode one Mach-O relocation per coefficient slot, in stream order.

    Each entry addresses the slot's byte offset in the stream and keeps the encoded
    addend; its info word holds the section ordinal, ``r_pcrel`` and the log2 width.
    """
    slots = base_slots(program)
    entries = []
    for word in sorted(reloc.word_offset for reloc in relocs):
        width = isa.COEFFICIENT_RELOCATIONS[slots[word][1]]
        info = section | 1 << 24 | (width.bit_length() - 1) << 25
        entries.append(RELOCATION.pack(address=word * 4, info=info))
    return b"".join(entries)


__all__ = [
    "DMA_BASES", "CoefficientReloc", "Reloc", "base_slots", "coefficient_relocations",
    "packet_slots", "relocation_table",
]
