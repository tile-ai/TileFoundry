"""H14 HWX image: a descriptor program, its I/O bindings and coefficients, laid out as bytes."""

from dataclasses import dataclass, replace

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa import arith

from .bindings import BINDING_ROLE_INPUT, BINDING_ROLE_OUTPUT, TensorBinding
from .coefficients import COEFF_LANE_ALIGN, Coefficient, coefficient_section
from .macho import (
    COMMAND,
    HEADER,
    LC_SEGMENT_64,
    PAGE,
    PAGEZERO,
    SEGMENT,
    SYMBOL,
    SYMTAB,
    VM_BASE,
    Section,
    Segment,
    first_page,
    load_command,
    symbol_table,
)
from .records import DEFAULT_RECORD_COUNT, TOOL_NAME, commands, symbols
from .relocations import CoefficientReloc, Reloc, relocation_table
from .validation import validate

TEXT_VM_SIZE = 0x8000
TEXT_CONST_SIZE = 0x4000
TEXT_CONST_ALIGN = 0x40
KERNEL_SEGMENT = b"__KERN_0"


@dataclass(frozen=True)
class Image:
    """A complete H14 container assembled from a descriptor program and its bindings."""

    program: isa.Program
    io: tuple[TensorBinding, ...]
    relocs: tuple[Reloc, ...]
    procedure: str = "main"
    output_count: int = 1
    scratch_bytes: int = 0
    record_count: int = DEFAULT_RECORD_COUNT
    format_code: int | None = None
    kernel_symbol: str = "coeff"
    kernel_digest: str | None = None
    coefficients: tuple[Coefficient, ...] = ()
    coefficient_relocs: tuple[CoefficientReloc, ...] = ()
    tool: str = TOOL_NAME
    version: str = "1.0"
    layout: str = "canonical"

    def bindings(self) -> tuple[TensorBinding, ...]:
        """Order signature outputs before inputs and assign their virtual addresses."""
        count = self.output_count
        if type(count) is not int or not 0 <= count <= len(self.io):
            raise ValueError(f"output count {count!r} does not fit {len(self.io)} I/O buffers")
        split = len(self.io) - count
        inputs, outputs = self.io[:split], self.io[split:]
        binding_list: list[TensorBinding] = []
        address = VM_BASE
        for role, group in ((BINDING_ROLE_OUTPUT, outputs), (BINDING_ROLE_INPUT, inputs)):
            for item in group:
                binding_list.append(replace(item, role=role, address=address))
                address += item.size
        return tuple(binding_list)

    def to_bytes(self) -> bytes:
        """Validate the image, then lay out and encode the container.

        Virtual memory holds the surfaces from ``VM_BASE``, the text reservation and the
        coefficients. The file holds the header, load commands, symbols, strings and
        relocations, then the text and kernel reservations from the first page that clears
        them, so a mapped segment's file offset is its address plus one slide. The kernel
        reservation is kept even without coefficients, and its section is the last one.
        """
        validate(self)
        bindings = self.bindings()
        text = self.program.to_bytes()
        kernel = coefficient_section(self.coefficients)
        text_address = VM_BASE + sum(binding.size for binding in bindings)
        const_address = text_address + arith.align_up(len(text), TEXT_CONST_ALIGN)
        kernel_address = text_address + TEXT_VM_SIZE
        kernel_span = max(2 * PAGE, arith.align_up(len(kernel), PAGE))
        if const_address + TEXT_CONST_SIZE > kernel_address:
            raise ValueError("__TEXT cannot hold its const section")
        text_segment = Segment(b"__TEXT", text_address, TEXT_VM_SIZE, 5, (
            Section(b"__text", text_address, len(text), PAGE, 0x28, len(self.coefficient_relocs)),
            Section(b"__const", const_address, TEXT_CONST_SIZE, TEXT_CONST_ALIGN, 0x26),
        ), mapped=True)
        kernel_segment = Segment(KERNEL_SEGMENT, kernel_address, kernel_span, 1, (
            Section(b"__kern_0", kernel_address, len(kernel), COEFF_LANE_ALIGN, 0x26),
        ), mapped=True)
        segments = (PAGEZERO, *(binding.segment() for binding in bindings), text_segment)
        segments += (kernel_segment,) if kernel else ()
        kernel_section = sum(len(segment.sections) for segment in segments)
        records = commands(
            self, bindings, text, kernel, text_address, const_address, kernel_address
        )
        names, strings = symbol_table(symbols(self, bindings, kernel_address, kernel_section))
        relocations = relocation_table(self.program, self.coefficient_relocs, kernel_section)
        command_size = sum(segment.command_size for segment in segments)
        command_size += sum(map(len, records)) + SYMTAB.size
        symbol_offset = HEADER.size + command_size
        string_offset = symbol_offset + len(names)
        relocation_offset = string_offset + len(strings)
        text_offset = first_page(relocation_offset + len(relocations), len(bindings))
        slide = text_offset - text_address
        metadata = b"".join((
            HEADER.pack(command_count=len(segments) + len(records) + 1, command_size=command_size),
            *(segment.command(slide, relocation_offset) for segment in segments),
            *records,
            load_command(
                SYMTAB, symbol_offset=symbol_offset, symbol_count=len(names) // SYMBOL.size,
                string_offset=string_offset, string_size=len(strings),
            ),
            names,
            strings,
            relocations,
        ))
        return (
            metadata.ljust(text_offset, b"\0")
            + text.ljust(TEXT_VM_SIZE, b"\0")
            + kernel.ljust(kernel_span, b"\0")
        )


def coefficient_offset(payload: bytes) -> int:
    """Locate the coefficient segment in a validated serialized container."""
    cursor = HEADER.size
    for _ in range(HEADER.unpack(payload)["command_count"]):
        command = COMMAND.unpack(payload, cursor)
        if command["command"] == LC_SEGMENT_64:
            segment = SEGMENT.unpack(payload, cursor)
            if segment["name"].rstrip(b"\0") == KERNEL_SEGMENT:
                return segment["offset"]
        cursor += command["size"]
    raise ValueError("image has no coefficient segment")


__all__ = ["Image", "coefficient_offset"]
