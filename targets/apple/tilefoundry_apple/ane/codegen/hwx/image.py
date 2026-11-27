"""H14 HWX image encoding."""

from dataclasses import dataclass, replace

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa import arith

from .bindings import BINDING_ROLE_INPUT, BINDING_ROLE_OUTPUT, TensorBinding, binding_segment
from .coefficients import Coefficient, coefficient_section
from .macho import (
    CPU,
    FILE_TYPE,
    H14,
    HEADER,
    LC_SEGMENT_64,
    LC_SYMTAB,
    MAGIC,
    PAGE,
    RELOCATION,
    SEGMENT,
    SYMBOL,
    SYMTAB,
    TEXT_CONST_ALIGN,
    TEXT_CONST_SIZE,
    TEXT_VM_SIZE,
    VM_BASE,
    Segment,
    file_offset,
    kernel_segment,
    pagezero,
    text_segment,
)
from .records import (
    DEFAULT_RECORD_COUNT,
    RESOURCE_SLOTS,
    TOOL_NAME,
    binding_record,
    buffer_reference,
    compiler_metadata,
    runtime_version,
    segment_record,
    source_model_info,
    validate_names,
)
from .relocations import DMA_BASES, CoefficientReloc, Reloc, base_slots, coeff_slots
from .symbols import symbol_table


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

    def relocation_table(self, text: bytes, kernel_section: int | None) -> bytes:
        """Encode section-relative coefficient relocations, preserving each addend."""
        records = bytearray()
        if self.coefficient_relocs and kernel_section is None:
            raise ValueError("kernel relocations need a kernel section")
        slots = coeff_slots(self.program)

        words = sorted(reloc.word_offset for reloc in self.coefficient_relocs)
        used = {record * 4 for record in (reloc.word_offset for reloc in self.relocs)}
        for word in words:
            if type(word) is not int or word < 0:
                raise ValueError(f"kernel relocation word {word!r} is not a word offset")
            off = word * 4
            width = isa.COEFFICIENT_RELOCATIONS.get(slots.get(word))
            if width is None:
                raise ValueError(
                    f"kernel relocation word {word} does not address a coefficient slot"
                )
            if off + width > len(text):
                raise ValueError(f"kernel relocation at word {word} leaves the descriptor")
            if off in used:
                raise ValueError(f"overlapping kernel relocation at word {word}")

            used.add(off)
            flags = (1 << 24) | ((width.bit_length() - 1) << 25) | kernel_section
            records.extend(RELOCATION.pack(off, flags))
        return bytes(records)

    def validate(self) -> None:
        """Reject invalid procedure metadata, empty programs and incompatible I/O."""
        if not self.procedure or not self.procedure.isascii() or not self.procedure.isprintable():
            raise ValueError(f"invalid procedure name {self.procedure!r}")
        if type(self.record_count) is not int or not 0 <= self.record_count < 2**32:
            raise ValueError(f"record count {self.record_count!r} is not a u32")
        if self.format_code is not None and (
            type(self.format_code) is not int or not 0 <= self.format_code < 2**32
        ):
            raise ValueError(f"format code {self.format_code!r} is not a u32")
        if type(self.scratch_bytes) is not int or not 0 <= self.scratch_bytes < 2**32:
            raise ValueError(f"scratch size {self.scratch_bytes!r} is not a u32")
        if self.scratch_bytes:
            raise ValueError("nonzero scratch_bytes mapping is not decoded yet")
        validate_names(self.procedure, tuple(item.name for item in self.io))
        if not self.program.tds:
            raise ValueError("image text must contain a descriptor")
        if len({item.name for item in self.io}) != len(self.io):
            raise ValueError("duplicate image I/O symbol")
        for item in self.io:
            item.validate(self.layout)
        bindings = self.bindings()
        slots = base_slots(self.program)
        outputs = {item.name for item in bindings if item.role == BINDING_ROLE_OUTPUT}
        for reloc in self.relocs:
            slot = slots.get(reloc.word_offset)
            if slot is not None and slot[1] == isa.TileDMADst.tile_dma_dst_base_addr_lo.addr and reloc.symbol not in outputs:
                raise ValueError(f"GMEM intermediate {reloc.symbol!r} mapping is not decoded yet")
        for td in self.program.tds:
            td.validate()
        isa.dma.check_continuations(td.values() for td in self.program.tds)
        self.require_relocated()

    def require_relocated(self) -> None:
        """Reject a program with a base address that no relocation binds to a buffer.

        The loader maps only the buffers this container declares. A DMA whose
        base stays at its placeholder reads or writes outside every mapping,
        which the engine's address translation reports as a fatal fault.
        """
        slots = base_slots(self.program)
        keys = [(index, addr) for index, addr, _ in slots.values()]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate base register in a descriptor")
        kernel = isa.KernelDMA
        accesses = {}
        for index, td in enumerate(self.program.tds):
            values = td.values()
            for enabled, base in (
                (kernel.kernel_dma_src_enable, kernel.kernel_dma_src_base_lo),
                (kernel.nonlinear_lut_enable, kernel.kernel_dma_lut_base),
            ):
                if enabled.get(values.get(enabled.addr, 0)) and (index, base.addr) not in keys:
                    raise ValueError("enabled coefficient reader has no relocated base")
            for transfer in isa.dma.transfers(values):
                accesses[index, transfer.register] = transfer
                if (index, transfer.register) not in keys:
                    raise ValueError(f"enabled {transfer.name} DMA has no base word")
        dma = {word for word, (_, addr, _) in slots.items() if addr in DMA_BASES}
        bound = [reloc.word_offset for reloc in self.relocs]
        if len(bound) != len(set(bound)):
            raise ValueError("duplicate DMA binding")
        if dma - set(bound):
            raise ValueError("DMA base is bound to no I/O buffer")
        if set(bound) - dma:
            raise ValueError("DMA binding does not address a DMA slot")
        buffers = {item.name: item for item in self.io}
        inputs = self.io[:len(self.io) - self.output_count]
        outputs = self.io[len(self.io) - self.output_count:]
        mapped = {
            isa.TileDMASrc.tile_dma_src_base_addr_lo.addr: inputs[:1],
            isa.TileDMASrc.tile_dma_src2_base_addr_lo.addr: inputs[1:2],
            isa.TileDMADst.tile_dma_dst_base_addr_lo.addr: outputs,
        }
        for reloc in self.relocs:
            if reloc.symbol not in buffers:
                raise ValueError(f"unknown I/O symbol {reloc.symbol!r}")
            index, register, addend = slots[reloc.word_offset]
            transfer = accesses.get((index, register))
            if transfer is None:
                raise ValueError(f"DMA base {register:#x} has no enabled transfer")
            if len(mapped[register]) != 1 or mapped[register][0].name != reloc.symbol:
                raise ValueError(f"{transfer.name} DMA symbol-to-signature mapping is not decoded yet")
            desc = buffers[reloc.symbol].desc
            desc.validate_feature()
            if not desc.is_canonical:
                raise ValueError("declared DMA tensor layout is not decoded yet")
            if desc.shape[1:] != (transfer.channels, transfer.height, transfer.width):
                raise ValueError(f"{transfer.name} DMA geometry contradicts TensorDesc {reloc.symbol!r}")
            if (transfer.row, transfer.channel) != (desc.row, desc.channel):
                raise ValueError(f"{transfer.name} DMA strides contradict TensorDesc {reloc.symbol!r}")
            end = addend + transfer.end
            if addend < 0 or end > desc.storage_bytes:
                raise ValueError(f"{transfer.name} DMA range [{addend}, {end}) exceeds "
                                 f"TensorDesc {reloc.symbol!r} storage_bytes {desc.storage_bytes}")
        end = 0
        for coefficient in self.coefficients:
            coefficient.validate()
            if coefficient.layout.base != end:
                raise ValueError("coefficient blocks must be consecutive in first-use order")
            end += coefficient.layout.size
        expected = {word for word, (_, addr, _) in slots.items()
                    if addr in isa.COEFFICIENT_RELOCATIONS}
        bound = [reloc.word_offset for reloc in self.coefficient_relocs]
        if len(bound) != len(set(bound)):
            raise ValueError("duplicate coefficient binding")
        if expected != set(bound):
            raise ValueError("coefficient base is bound to no coefficient section or slot")
        first_uses = tuple(dict.fromkeys(reloc.coefficient for reloc in sorted(
            self.coefficient_relocs, key=lambda item: item.word_offset
        )))
        if first_uses != tuple(range(len(self.coefficients))):
            raise ValueError("coefficient blocks do not match first consumption order")
        for reloc in self.coefficient_relocs:
            if type(reloc.coefficient) is not int or not 0 <= reloc.coefficient < len(self.coefficients):
                raise ValueError("coefficient binding names a missing block")
            layout = self.coefficients[reloc.coefficient].layout
            td_index, register, addend = slots[reloc.word_offset]
            values = {addr: value for packet in self.program.tds[td_index].packets
                      for addr, value in packet.writes}
            if register == kernel.kernel_dma_src_base_lo.addr:
                if addend != layout.base:
                    raise ValueError("coefficient base disagrees with its planned block")
                isa.dma.check_coefficient_lanes(layout.lanes, sum(
                    field.get(values.get(field.addr, 0)) for field in kernel.kernel_dma_src_coeff_dma_en
                ))
                for enabled, offset, size in zip(
                    kernel.kernel_dma_src_coeff_dma_en,
                    kernel.kernel_dma_src_coeff_base_offset,
                    kernel.kernel_dma_src_coeff_mem_buffer_size,
                ):
                    if not enabled.get(values.get(enabled.addr, 0)):
                        continue
                    start = offset.get(values.get(offset.addr, 0)) * 64
                    length = size.get(values.get(size.addr, 0)) * 64 or 64
                    if start % layout.stride or length != layout.stride or start + length > layout.size:
                        raise ValueError("coefficient DMA range leaves its planned block")
            elif register == kernel.kernel_dma_lut_base.addr and layout.lut:
                length = kernel.nonlinear_lut_size.get(values.get(kernel.nonlinear_lut_size.addr, 0)) * 64
                if not length or not layout.base <= addend < addend + length <= layout.base + layout.size:
                    raise ValueError("lookup table range leaves its planned block")
            else:
                raise ValueError(f"coefficient register {register:#x} read range is not decoded yet")

    def records(
        self,
        bindings: tuple[TensorBinding, ...],
        text: bytes,
        blob: bytes,
        text_address: int,
        text_const_address: int,
        kernel_address: int,
    ) -> tuple[bytes, ...]:
        """Encode ordered surface references, procedure records and provenance commands."""
        resources = [binding.address for binding in bindings]
        if self.coefficients and len(resources) < RESOURCE_SLOTS:
            resources.append(kernel_address)
        command_payloads = tuple(buffer_reference(binding) for binding in bindings)
        segment_payload = segment_record(
            text_address,
            text_const_address,
            tuple(resources),
            len(text) // 4,
            len(self.program.tds),
            self.procedure,
            self.record_count,
            self.format_code
            if self.format_code is not None
            else max(descriptor.header.get(0x04, 0) for descriptor in self.program.tds),
        )
        io_bindings = tuple(
            binding for binding in bindings if binding.role == BINDING_ROLE_INPUT
        ) + tuple(binding for binding in bindings if binding.role == BINDING_ROLE_OUTPUT)
        binding_payloads = tuple(
            binding_record(binding, self.procedure, self.layout == "declared")
            for binding in io_bindings
        )
        binding_bytes = sum(binding.size for binding in bindings)
        metadata_payload = compiler_metadata(
            self.tool, self.version, len(blob), binding_bytes, len(self.program.tds)
        )
        source_payload = source_model_info(
            self.tool, self.version, self.procedure, len(self.program.tds)
        )
        runtime_payload = runtime_version()

        return (
            *command_payloads,
            segment_payload,
            *binding_payloads,
            metadata_payload,
            source_payload,
            runtime_payload,
        )

    def segments(
        self,
        bindings: tuple[TensorBinding, ...],
        text_size: int,
        blob_size: int,
        *,
        text_file_offset: int = 0,
        reloc_offset: int = 0,
        reloc_count: int = 0,
    ) -> tuple[Segment, ...]:
        """Build guard, surface, text and optional coefficient segment commands."""
        address = VM_BASE + sum(binding.size for binding in bindings)
        segments = [pagezero(), *(binding_segment(binding) for binding in bindings)]
        segments.append(
            text_segment(
                address,
                text_file_offset,
                text_size,
                reloc_offset,
                reloc_count,
                bool(self.coefficients),
            )
        )
        if self.coefficients:
            segments.append(
                kernel_segment(
                    address + TEXT_VM_SIZE,
                    text_file_offset + TEXT_VM_SIZE,
                    blob_size,
                )
            )
        return tuple(segments)

    def to_bytes(self) -> bytes:
        """Plan segment addresses and assemble the little-endian container."""
        self.validate()
        text = self.program.to_bytes()
        has_kern = bool(self.coefficients)
        bindings = self.bindings()
        text_address = VM_BASE + sum(binding.size for binding in bindings)
        text_const_address = arith.align_up(text_address + len(text), TEXT_CONST_ALIGN)
        if text_const_address + TEXT_CONST_SIZE > text_address + TEXT_VM_SIZE:
            raise ValueError("__TEXT cannot hold its const section")
        kernel_address = text_address + TEXT_VM_SIZE
        blob = coefficient_section(self.coefficients)

        kernel_size = max(2 * PAGE, arith.align_up(len(blob), PAGE))
        payloads = self.records(
            bindings, text, blob, text_address, text_const_address, kernel_address
        )
        segments = self.segments(bindings, len(text), len(blob))
        command_size = (
            sum(len(segment.command()) for segment in segments)
            + sum(map(len, payloads))
            + SYMTAB.size
        )
        command_count = len(segments) + len(payloads) + 1
        kernel_section = len(bindings) + 3 if has_kern else None
        symbols, strings = symbol_table(self, bindings, kernel_section)
        reloc_data = self.relocation_table(text, kernel_section)
        reloc_count = len(reloc_data) // RELOCATION.size

        text_file_offset = file_offset(
            HEADER.size + command_size + len(symbols) + len(strings) + len(reloc_data),
            len(bindings),
        )
        if text_file_offset is None:
            raise ValueError("image metadata does not fit below the __TEXT segment")
        symbol_offset = HEADER.size + command_size
        string_offset = symbol_offset + len(symbols)
        reloc_offset = string_offset + len(strings) if reloc_data else 0
        if reloc_offset + len(reloc_data) > text_file_offset:
            raise ValueError("image symbols and relocations cross the __TEXT segment")
        text_const_file_offset = text_file_offset + arith.align_up(len(text), TEXT_CONST_ALIGN)
        if text_const_file_offset + TEXT_CONST_SIZE > text_file_offset + TEXT_VM_SIZE:
            raise ValueError("__TEXT cannot hold its const section on disk")
        kernel_file_offset = text_file_offset + TEXT_VM_SIZE

        segments = self.segments(
            bindings,
            len(text),
            len(blob),
            text_file_offset=text_file_offset,
            reloc_offset=reloc_offset,
            reloc_count=reloc_count,
        )
        commands = bytearray(
            b"".join(segment.command() for segment in segments) + b"".join(payloads)
        )
        commands += SYMTAB.pack(
            LC_SYMTAB,
            SYMTAB.size,
            symbol_offset,
            len(symbols) // SYMBOL.size,
            string_offset,
            len(strings),
        )
        if len(commands) != command_size:
            raise ValueError("command area size disagrees with its plan")

        out = bytearray(text_file_offset + TEXT_VM_SIZE + kernel_size)
        out[: HEADER.size] = HEADER.pack(
            MAGIC, CPU, H14, FILE_TYPE, command_count, command_size, 0x00200000, 0
        )
        out[HEADER.size : HEADER.size + command_size] = commands
        out[symbol_offset : symbol_offset + len(symbols)] = symbols
        out[string_offset : string_offset + len(strings)] = strings
        out[reloc_offset : reloc_offset + len(reloc_data)] = reloc_data
        out[text_file_offset : text_file_offset + len(text)] = text
        if has_kern:
            out[kernel_file_offset : kernel_file_offset + len(blob)] = blob
        return bytes(out)


def coefficient_offset(payload: bytes) -> int:
    """Locate the coefficient segment in a validated serialized container."""
    cursor = HEADER.size
    for _ in range(HEADER.unpack_from(payload)[4]):
        command, size = HEADER.unpack_from(payload, cursor)[:2]
        if command == LC_SEGMENT_64:
            segment = SEGMENT.unpack_from(payload, cursor)
            if segment[2].rstrip(b"\0") == b"__KERN_0":
                return segment[5]
        cursor += size
    raise ValueError("image has no coefficient segment")


__all__ = ["Image", "coefficient_offset"]
