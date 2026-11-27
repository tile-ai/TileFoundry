"""The Mach-O variant of an H14 HWX container: fixed-layout records, load commands, segments."""

import struct
from collections.abc import Iterable
from dataclasses import dataclass

from tilefoundry_apple.ane.isa import arith

VM_BASE = 0x30000000
PAGE = 0x4000
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2
N_EXT = 0x1
N_SECT = 0xE
N_GSYM = 0x20
N_LSYM = 0x80
SG_FVMLIB = 0x2
SG_NORELOC = 0x4
S_ATTR_LOC_RELOC = 0x100


@dataclass(frozen=True)
class Field:
    """One little-endian field at a byte offset; *format* is a ``struct`` format.

    A named field takes its value from ``Record.pack`` and falls back to *default*;
    an unnamed field always holds *default*, a constant whose meaning is not decoded.
    """

    offset: int
    format: str
    name: str = ""
    default: int | bytes | tuple[int, ...] = 0


class Record:
    """Fields at fixed offsets; the record ends where its last field ends."""

    def __init__(self, *fields: Field) -> None:
        self.fields = fields
        self.size = max(field.offset + struct.calcsize("<" + field.format) for field in fields)

    def pack(self, **values: int | bytes | tuple[int, ...]) -> bytes:
        """Encode the named *values*; every other field holds its default."""
        unknown = values.keys() - {field.name for field in self.fields if field.name}
        if unknown:
            raise TypeError(f"record has no fields {sorted(unknown)}")
        out = bytearray(self.size)
        for field in self.fields:
            value = values.get(field.name, field.default)
            items = value if isinstance(value, tuple) else (value,)
            struct.pack_into("<" + field.format, out, field.offset, *items)
        return bytes(out)

    def unpack(self, data: bytes, offset: int = 0) -> dict[str, int | bytes | tuple[int, ...]]:
        """Decode the named fields of the record that starts at *offset* of *data*."""
        values = {}
        for field in self.fields:
            if field.name:
                items = struct.unpack_from("<" + field.format, data, offset + field.offset)
                values[field.name] = items[0] if len(items) == 1 else items
        return values


HEADER = Record(
    Field(0x00, "I", "magic", 0xBEEFFACE),
    Field(0x04, "I", "cpu", 0x80),
    Field(0x08, "I", "subtype", 5),
    Field(0x0C, "I", "file_type", 2),
    Field(0x10, "I", "command_count"),
    Field(0x14, "I", "command_size"),
    Field(0x18, "I", "flags", 0x00200000),
    Field(0x1C, "I"),
)
COMMAND = Record(Field(0x0, "I", "command"), Field(0x4, "I", "size"))
SEGMENT = Record(
    Field(0x00, "I", "command", LC_SEGMENT_64),
    Field(0x04, "I", "size"),
    Field(0x08, "16s", "name"),
    Field(0x18, "Q", "address"),
    Field(0x20, "Q", "vm_size"),
    Field(0x28, "Q", "offset"),
    Field(0x30, "Q", "file_size"),
    Field(0x38, "I", "max_protection"),
    Field(0x3C, "I", "protection"),
    Field(0x40, "I", "section_count"),
    Field(0x44, "I", "flags"),
)
SECTION = Record(
    Field(0x00, "16s", "name"),
    Field(0x10, "16s", "segment"),
    Field(0x20, "Q", "address"),
    Field(0x28, "Q", "size"),
    Field(0x30, "I", "offset"),
    Field(0x34, "I", "align"),
    Field(0x38, "I", "reloc_offset"),
    Field(0x3C, "I", "reloc_count"),
    Field(0x40, "I", "flags"),
    Field(0x44, "3I", default=(0, 0, 0)),
)
SYMTAB = Record(
    Field(0x00, "I", "command", LC_SYMTAB),
    Field(0x04, "I", "size"),
    Field(0x08, "I", "symbol_offset"),
    Field(0x0C, "I", "symbol_count"),
    Field(0x10, "I", "string_offset"),
    Field(0x14, "I", "string_size"),
)
SYMBOL = Record(
    Field(0x0, "I", "name"),
    Field(0x4, "B", "type"),
    Field(0x5, "B", "section"),
    Field(0x6, "H", "desc"),
    Field(0x8, "Q", "value"),
)
RELOCATION = Record(Field(0x0, "i", "address"), Field(0x4, "I", "info"))


def load_command(
    record: Record, tail: bytes = b"", **values: int | bytes | tuple[int, ...]
) -> bytes:
    """Encode *record* followed by *tail*; its ``size`` field covers both, 8-byte aligned."""
    size = arith.align_up(record.size + len(tail), 8)
    return (record.pack(size=size, **values) + tail).ljust(size, b"\0")


@dataclass(frozen=True)
class Section:
    """A virtual range of its segment; *align* is in bytes, *relocations* counts its entries."""

    name: bytes
    address: int
    size: int
    align: int
    flags: int
    relocations: int = 0


@dataclass(frozen=True)
class Segment:
    """One LC_SEGMENT_64 command; a mapped segment's file bytes mirror its virtual range."""

    name: bytes
    address: int
    size: int
    protection: int
    sections: tuple[Section, ...] = ()
    flags: int = SG_NORELOC
    mapped: bool = False

    @property
    def command_size(self) -> int:
        """The bytes of the command and its section headers."""
        return SEGMENT.size + SECTION.size * len(self.sections)

    def file_offset(self, address: int, slide: int) -> int:
        """Return the file offset of *address*; an unmapped segment records zero."""
        return address + slide if self.mapped else 0

    def command(self, slide: int, relocations: int) -> bytes:
        """Encode the command; *slide* turns a mapped address into its file offset.

        A section with relocations points at the table at file offset *relocations*,
        carries ``S_ATTR_LOC_RELOC`` and clears ``SG_NORELOC`` of its segment.
        """
        sections = b"".join(
            SECTION.pack(
                name=section.name, segment=self.name, address=section.address, size=section.size,
                offset=self.file_offset(section.address, slide),
                align=section.align.bit_length() - 1,
                reloc_offset=relocations if section.relocations else 0,
                reloc_count=section.relocations,
                flags=section.flags | (S_ATTR_LOC_RELOC if section.relocations else 0),
            )
            for section in self.sections
        )
        relocated = any(section.relocations for section in self.sections)
        return load_command(
            SEGMENT, sections, name=self.name, address=self.address, vm_size=self.size,
            offset=self.file_offset(self.address, slide), file_size=self.size if self.mapped else 0,
            max_protection=self.protection, protection=self.protection,
            section_count=len(self.sections),
            flags=self.flags & ~SG_NORELOC if relocated else self.flags,
        )


PAGEZERO = Segment(b"__PAGEZERO", 0, PAGE, 0)


def symbol_table(symbols: Iterable[tuple[str, int, int, int, int]]) -> tuple[bytes, bytes]:
    """Encode ``(name, type, section, desc, value)`` entries and the strings they index."""
    entries, strings = bytearray(), bytearray(b"\0")
    for name, kind, section, desc, value in symbols:
        entries += SYMBOL.pack(
            name=len(strings), type=kind, section=section, desc=desc, value=value
        )
        strings += name.encode("ascii") + b"\0"
    return bytes(entries), bytes(strings)


def first_page(metadata_end: int, surfaces: int) -> int:
    """Pick the file offset of the first segment so that it clears the metadata.

    The reference compiler starts at 0x4000 for at most two surfaces and at 0x8000
    otherwise, and doubles the offset up to 0x20000 while the metadata does not fit.
    """
    offset = PAGE if surfaces <= 2 else 2 * PAGE
    while offset < metadata_end:
        offset *= 2
    if offset > 8 * PAGE:
        raise ValueError("image metadata does not fit below the __TEXT segment")
    return offset


__all__ = [
    "COMMAND", "HEADER", "LC_SEGMENT_64", "LC_SYMTAB", "N_EXT", "N_GSYM", "N_LSYM", "N_SECT",
    "PAGE", "PAGEZERO", "RELOCATION", "SEGMENT", "SG_FVMLIB", "SG_NORELOC", "SYMBOL", "SYMTAB",
    "VM_BASE", "Field", "Record", "Section", "Segment", "first_page", "load_command",
    "symbol_table",
]
