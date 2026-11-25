"""H14 task descriptors and their aligned program stream."""

import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .fields import BitField, block_at, check_word
from .serialize import (
    MARKER_ADDR,
    PAD_SENTINEL,
    WORD_BYTES,
    Packet,
    sentinel_packets,
    serialize,
)

TID = BitField(0, 15, 0)
TASK_SIZE = BitField(0, 26, 16)
MAX_TASK_WORDS = TASK_SIZE.get(TASK_SIZE.mask)
FLAGS = BitField(0, 31, 27)
DEPENDENCY_ENABLE = BitField(0x1C, 1, 1)
DEPENDENCY_ID = BitField(0x20, 15, 0)

HEADER_WORDS = 8
#: Only the task-status bit is on by default; the event masks and the
#: engine-enable field are supplied per descriptor by the codegen policy.
HEADER_DEFAULTS = MappingProxyType({0x1C: 0x00000001})
ALIGNMENT = 16
PREAMBLE = bytes.fromhex("01000000000000000000000000000000")


def aligned_end(off: int) -> int:
    """Round a stream offset up to the next descriptor boundary."""
    return (off + ALIGNMENT - 1) & -ALIGNMENT


class TD:
    """A task descriptor: a header followed by an ordered packet sequence."""

    def __init__(
        self,
        packets: Iterable[Packet] = (),
        *,
        tid: int = 0,
        flags: int = 0,
        header: Mapping[int, int] | None = None,
        dependency: int | None = None,
    ) -> None:
        self.packets = tuple(packets)
        self.tid = tid
        self.flags = flags
        self.header = dict(HEADER_DEFAULTS if header is None else header)
        if dependency is not None:
            self.dependency = dependency

    @property
    def dependency(self) -> int | None:
        """Return the dependency task ID when the header carries one."""
        word = self.header.get(DEPENDENCY_ID.addr)
        return None if word is None else DEPENDENCY_ID.get(word)

    @dependency.setter
    def dependency(self, dtid: int | None) -> None:
        addr = DEPENDENCY_ENABLE.addr
        control = DEPENDENCY_ENABLE.put(self.header.get(addr, 0), int(dtid is not None))
        if dtid is None:
            self.header.pop(DEPENDENCY_ID.addr, None)
        else:
            self.header[DEPENDENCY_ID.addr] = DEPENDENCY_ID.put(
                self.header.get(DEPENDENCY_ID.addr, 0), dtid
            )
        self.header[addr] = control

    @classmethod
    def build(
        cls, values: Mapping[BitField | int, int], *, base: Mapping[int, int] | None = None
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
        return cls(
            serialize({addr: word for addr, word in image.items() if initial.get(addr) != word})
        )

    def header_size_words(self) -> int:
        """Count the fixed header and the optional dependency word."""
        return HEADER_WORDS + (DEPENDENCY_ID.addr in self.header)

    def size_words(self) -> int:
        """Count the words this descriptor states, marker packets included."""
        return self.header_size_words() + sum(1 + len(packet.writes) for packet in self.packets)

    def marker_packets(self) -> int:
        """Count the marker packets already present in the packet list."""
        return sum(1 for packet in self.packets if dict(packet.writes).get(MARKER_ADDR) == PAD_SENTINEL)

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

    def to_bytes(self, off: int = 0) -> bytes:
        """Encode the descriptor as little-endian words, padded to its aligned end."""
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
        has_dependency = DEPENDENCY_ENABLE.get(self.header.get(DEPENDENCY_ENABLE.addr, 0))
        if has_dependency != (DEPENDENCY_ID.addr in self.header):
            raise ValueError("dependency flag and dependency word disagree")
        size = self.base_words() + 2 * count
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

    @classmethod
    def decode(cls, buf: bytes, off: int = 0) -> tuple["TD", int]:
        """Read one descriptor at an aligned offset and return the next aligned offset."""
        if off < 0 or off % ALIGNMENT:
            raise ValueError(f"unaligned TD offset {off}")
        if off + HEADER_WORDS * WORD_BYTES > len(buf):
            raise ValueError(f"truncated TD header at {off:#x}")
        first = struct.unpack_from("<I", buf, off)[0]
        control = struct.unpack_from("<I", buf, off + DEPENDENCY_ENABLE.addr)[0]
        start = HEADER_WORDS + DEPENDENCY_ENABLE.get(control)
        size = TASK_SIZE.get(first)
        if size < start or off + size * WORD_BYTES > len(buf):
            raise ValueError(f"invalid task size {size} at {off:#x}")
        words = struct.unpack_from(f"<{size}I", buf, off)
        header = {i * WORD_BYTES: words[i] for i in range(1, start)}
        packets: list[Packet] = []
        i = start
        while i < size:
            packet, i = Packet.from_words(words, i)
            packets.append(packet)
        return cls(packets, tid=TID.get(first), flags=FLAGS.get(first), header=header), aligned_end(
            off + size * WORD_BYTES
        )


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

    @classmethod
    def from_bytes(cls, buf: bytes) -> "Program":
        """Decode a framed stream and validate its preamble and padding."""
        if not buf.startswith(PREAMBLE):
            raise ValueError("invalid TD stream preamble")
        tds: list[TD] = []
        off = len(PREAMBLE)
        while off < len(buf):
            td, end = TD.decode(buf, off)
            raw_end = off + td.size_words() * WORD_BYTES
            if raw_end == len(buf):
                # A stream ends at its last descriptor word: only the gaps
                # between descriptors carry alignment padding.
                tds.append(td)
                break
            if end > len(buf):
                raise ValueError(f"truncated TD padding at {off:#x}")
            if any(buf[raw_end:end]):
                raise ValueError(f"nonzero TD padding at {off:#x}")
            tds.append(td)
            off = end
        return cls(tuple(tds))


__all__ = ["MAX_TASK_WORDS", "PREAMBLE", "Program", "TD"]
