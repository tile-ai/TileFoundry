"""H14 HWX macho encoding."""

import struct
from dataclasses import dataclass

from tilefoundry_apple.ane.isa import arith

HEADER = struct.Struct("<8I")
SEGMENT = struct.Struct("<II16sQQQQiiII")
SECTION = struct.Struct("<16s16sQQ8I")
SYMTAB = struct.Struct("<6I")
SYMBOL = struct.Struct("<IBBHQ")
RELOCATION = struct.Struct("<iI")
MAGIC = 0xBEEFFACE
CPU = 0x80
H14 = 5
FILE_TYPE = 2
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 2
VM_BASE = 0x30000000
PAGE = 0x4000
TEXT_VM_SIZE = 0x8000
TEXT_CONST_SIZE = 0x4000
TEXT_CONST_ALIGN = 0x40
SEG_PAGEZERO_FLAGS = 0x4
SEG_FVMLIB_FLAGS = 0x6
SEG_TEXT_FLAGS_PLAIN = 0x4
SEG_TEXT_FLAGS_KERNEL = 0x0
SEG_KERNEL_FLAGS = 0x4
SECT_DATA_FLAGS = 0x23
SECT_CONST_FLAGS = 0x21
SECT_TEXT_FLAGS = 0x28
SECT_TEXT_FLAGS_RELOC = 0x128
SECT_TEXT_CONST_FLAGS = 0x26
SECT_KERNEL_FLAGS = 0x26
SECT_ALIGN_IO = 14
SECT_ALIGN_TEXT = 14
SECT_ALIGN_TEXT_CONST = 6
SECT_ALIGN_KERNEL = 6


def name16(name: bytes, limit: int = 16) -> bytes:
    """Encode a fixed-width Mach-O name field."""
    return name[: limit - 1].ljust(limit, b"\0")


@dataclass(frozen=True)
class Section:
    """One section of one segment command."""

    name: bytes
    segment: bytes
    address: int
    size: int
    offset: int
    alignment: int
    flags: int
    reloc_offset: int = 0
    reloc_count: int = 0


@dataclass(frozen=True)
class Segment:
    """One LC_SEGMENT_64 command."""

    name: bytes
    address: int
    size: int
    file_offset: int
    file_size: int
    max_protection: int
    initial_protection: int
    flags: int
    sections: tuple[Section, ...] = ()

    def command(self) -> bytes:
        """Encode the segment command followed by its section commands."""
        out = bytearray(
            SEGMENT.pack(
                LC_SEGMENT_64,
                SEGMENT.size + SECTION.size * len(self.sections),
                name16(self.name),
                self.address,
                self.size,
                self.file_offset,
                self.file_size,
                self.max_protection,
                self.initial_protection,
                len(self.sections),
                self.flags,
            )
        )
        for section in self.sections:
            out += SECTION.pack(
                name16(section.name),
                name16(section.segment),
                section.address,
                section.size,
                section.offset,
                section.alignment,
                section.reloc_offset,
                section.reloc_count,
                section.flags,
                0,
                0,
                0,
            )
        return bytes(out)


def pagezero() -> Segment:
    """Encode the guard page Mach-O requires below the image."""
    return Segment(b"__PAGEZERO", 0, PAGE, 0, 0, 0, 0, SEG_PAGEZERO_FLAGS)


def file_offset(metadata_end: int, surface_count: int) -> int | None:
    """Pick the first segment file offset that clears the metadata.

    The compiler places __TEXT at 0x4000 when a program has at most two
    surfaces and at 0x8000 otherwise, then steps further out only when the
    metadata would not clear it.
    """
    candidates = (PAGE, 2 * PAGE, 4 * PAGE, 8 * PAGE)
    first = 0 if surface_count <= 2 else 1
    for candidate in candidates[first:]:
        if metadata_end <= candidate:
            return candidate
    return None


def text_segment(
    address: int,
    offset: int,
    size: int,
    reloc_offset: int,
    reloc_count: int,
    has_kernel: bool,
) -> Segment:
    """Describe the instruction stream and its aligned zero-filled const section."""
    const_start = arith.align_up(size, TEXT_CONST_ALIGN)
    return Segment(
        b"__TEXT",
        address,
        TEXT_VM_SIZE,
        offset,
        TEXT_VM_SIZE,
        5,
        5,
        SEG_TEXT_FLAGS_KERNEL if has_kernel else SEG_TEXT_FLAGS_PLAIN,
        (
            Section(
                b"__text",
                b"__TEXT",
                address,
                size,
                offset,
                SECT_ALIGN_TEXT,
                SECT_TEXT_FLAGS_RELOC if reloc_count else SECT_TEXT_FLAGS,
                reloc_offset,
                reloc_count,
            ),
            Section(
                b"__const",
                b"__TEXT",
                address + const_start,
                TEXT_CONST_SIZE,
                offset + const_start,
                SECT_ALIGN_TEXT_CONST,
                SECT_TEXT_CONST_FLAGS,
            ),
        ),
    )


def kernel_segment(address: int, offset: int, size: int) -> Segment:
    """Reserve at least two pages for the coefficient section."""
    span = max(2 * PAGE, arith.align_up(size, PAGE))
    return Segment(
        b"__KERN_0",
        address,
        span,
        offset,
        span,
        1,
        1,
        SEG_KERNEL_FLAGS,
        (
            Section(
                b"__kern_0",
                b"__KERN_0",
                address,
                size,
                offset,
                SECT_ALIGN_KERNEL,
                SECT_KERNEL_FLAGS,
            ),
        ),
    )


__all__ = [
    "CPU",
    "FILE_TYPE",
    "H14",
    "HEADER",
    "LC_SEGMENT_64",
    "LC_SYMTAB",
    "MAGIC",
    "PAGE",
    "RELOCATION",
    "SECT_ALIGN_IO",
    "SECT_CONST_FLAGS",
    "SECT_DATA_FLAGS",
    "SEG_FVMLIB_FLAGS",
    "SEGMENT",
    "SYMBOL",
    "SYMTAB",
    "Section",
    "Segment",
    "TEXT_CONST_ALIGN",
    "TEXT_CONST_SIZE",
    "TEXT_VM_SIZE",
    "VM_BASE",
    "file_offset",
    "kernel_segment",
    "pagezero",
    "text_segment",
]
