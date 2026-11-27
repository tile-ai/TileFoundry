"""H14 task descriptors and their aligned program stream."""

import struct
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType

from . import rules
from .fields import (
    DEPENDENCY_ENABLE,
    DEPENDENCY_ID,
    FLAGS,
    RESET_IMAGE,
    TASK_SIZE,
    TID,
    BitField,
    block_at,
    check_configuration_word,
    check_word,
)
from .serialize import (
    MARKER_ADDR,
    PAD_SENTINEL,
    WORD_BYTES,
    Packet,
    sentinel_packets,
    serialize,
)

MAX_TASK_WORDS = TASK_SIZE.get(TASK_SIZE.mask)
HEADER_WORDS = 8
HEADER_DEFAULTS = MappingProxyType({0x1C: 0x00000001})
ALIGNMENT = 16
PREAMBLE = bytes.fromhex("01000000000000000000000000000000")


def aligned_end(off: int) -> int:
    """Round a stream offset up to the next descriptor boundary."""
    return (off + ALIGNMENT - 1) & -ALIGNMENT


@dataclass(frozen=True)
class TD:
    """A task descriptor: a header followed by an ordered packet sequence."""

    packets: tuple[Packet, ...] = ()
    _: KW_ONLY
    tid: int = 0
    flags: int = 0
    header: Mapping[int, int] | None = None
    dependency: int | None = None

    def __post_init__(self) -> None:
        words = dict(HEADER_DEFAULTS if self.header is None else self.header)
        dependency = self.dependency
        if dependency is None:
            word = words.get(DEPENDENCY_ID.addr)
            dependency = None if word is None else DEPENDENCY_ID.get(word)
        else:
            addr = DEPENDENCY_ENABLE.addr
            words[addr] = DEPENDENCY_ENABLE.put(words.get(addr, 0), 1)
            words[DEPENDENCY_ID.addr] = DEPENDENCY_ID.put(
                words.get(DEPENDENCY_ID.addr, 0), dependency
            )
        object.__setattr__(self, "packets", tuple(self.packets))
        object.__setattr__(self, "header", MappingProxyType(words))
        object.__setattr__(self, "dependency", dependency)

    @classmethod
    def from_values(
        cls,
        values: Mapping[BitField | int, int],
        *,
        base: Mapping[int, int] | None = None,
        tid: int = 0,
    ) -> "TD":
        """Apply raw words first, overlay fields, then omit words equal to base."""
        initial = {} if base is None else base
        image: dict[int, int] = {}
        for key, value in values.items():
            addr = key.addr if isinstance(key, BitField) else key
            if block_at(addr) is None:
                raise ValueError(f"{addr!r} is not a configuration word address")
            if isinstance(key, int):
                check_word(value)
                image[key] = value
        for key, value in values.items():
            if isinstance(key, BitField):
                word = image.get(key.addr, initial.get(key.addr, 0))
                image[key.addr] = key.put(word, value)
        for addr, word in image.items():
            check_configuration_word(addr, word)
        td = cls(
            serialize({addr: word for addr, word in image.items() if initial.get(addr) != word}),
            tid=tid,
        )
        td.validate()
        return td

    def validate(self) -> None:
        """Validate packets, named field domains and effective register relationships."""
        for packet in self.packets:
            packet.words()
            for addr, word in packet.writes:
                check_configuration_word(addr, word)
        rules.check(self.values())

    def values(self) -> dict[int, int]:
        """Return this descriptor's effective configuration, including address addends."""
        return dict(RESET_IMAGE) | {
            addr: word for packet in self.packets for addr, word in packet.writes
        }

    def header_size_words(self) -> int:
        """Count the fixed header and the optional dependency word."""
        return HEADER_WORDS + (DEPENDENCY_ID.addr in self.header)

    def size_words(self) -> int:
        """Count the words this descriptor states, marker packets included."""
        return self.header_size_words() + sum(1 + len(packet.writes) for packet in self.packets)

    def marker_packets(self) -> int:
        """Count the marker packets already present in the packet list."""
        return sum(
            1 for packet in self.packets if dict(packet.writes).get(MARKER_ADDR) == PAD_SENTINEL
        )

    def base_words(self) -> int:
        """Count the words this descriptor states, marker packets excluded."""
        return self.header_size_words() + sum(
            1 + len(packet.writes)
            for packet in self.packets
            if dict(packet.writes).get(MARKER_ADDR) != PAD_SENTINEL
        )

    def padding_packets(self, off: int) -> int:
        """Count the marker packets this descriptor carries when it starts at ``off``."""
        present = self.marker_packets()
        if present:
            return present
        return sentinel_packets(off, self.base_words() * WORD_BYTES)

    def encoded_words(self, off: int = 0) -> int:
        """Count the words this descriptor occupies in a stream that starts it at ``off``."""
        return self.base_words() + 2 * self.padding_packets(off)

    def encoded_packets(self, off: int = 0) -> tuple[Packet, ...]:
        """Insert alignment sentinels before the flagged relocation packets."""
        count = self.padding_packets(off)
        packets = [
            packet
            for packet in self.packets
            if dict(packet.writes).get(MARKER_ADDR) != PAD_SENTINEL
        ]
        if count:
            marker = MARKER_ADDR
            close = tuple(Packet("run", ((marker, PAD_SENTINEL),)) for _ in range(count))
            at = next((i for i, packet in enumerate(packets) if packet.flags), len(packets))
            packets[at:at] = close
        return tuple(packets)

    def to_bytes(self, off: int = 0) -> bytes:
        """Encode the descriptor as little-endian words, padded to its aligned end."""
        packets = self.encoded_packets(off)
        has_dependency = DEPENDENCY_ENABLE.get(self.header.get(DEPENDENCY_ENABLE.addr, 0))
        if has_dependency != (DEPENDENCY_ID.addr in self.header):
            raise ValueError("dependency flag and dependency word disagree")
        size = self.encoded_words(off)
        first = TID.put(FLAGS.put(TASK_SIZE.put(0, size), self.flags), self.tid)
        header = [
            self.header.get(off, 0)
            for off in range(WORD_BYTES, WORD_BYTES * self.header_size_words(), WORD_BYTES)
        ]
        for value in header:
            check_word(value)
        words = [first, *header]
        for packet in packets:
            words.extend(packet.words())
        return struct.pack(f"<{len(words)}I", *words)

    def writes(self) -> dict[int, int]:
        """Return literal writes, with the last literal write to each word winning."""
        return {
            addr: val for packet in self.packets if not packet.flags for addr, val in packet.writes
        }


@dataclass(frozen=True)
class Program:
    """One stream of descriptors with a preamble and alignment padding."""

    tds: tuple[TD, ...]

    def to_bytes(self) -> bytes:
        """Frame the descriptors as a stream, with zero-filled alignment gaps.

        The section ends at the last descriptor word: a trailing gap is only
        present between descriptors, never after the final one.
        """
        stream = bytearray(PREAMBLE)
        for index, td in enumerate(self.tds):
            stream.extend(td.to_bytes(len(stream)))
            if index + 1 < len(self.tds):
                stream.extend(bytes(aligned_end(len(stream)) - len(stream)))
        return bytes(stream)


__all__ = ["MAX_TASK_WORDS", "PREAMBLE", "Program", "TD"]
