"""The checks an HWX image passes before it is laid out: metadata, I/O, descriptors, bindings."""

from collections.abc import Mapping

from tilefoundry_apple.ane import isa

from .bindings import BINDING_ROLE_OUTPUT
from .records import NAME_BYTES, PROCEDURE_BYTES
from .relocations import DMA_BASES, base_slots, packet_slots

DST = isa.TileDMADst.tile_dma_dst_base_addr_lo.addr


def is_u32(value: object) -> bool:
    """Whether *value* is an int that fits an unsigned 32-bit field."""
    return type(value) is int and 0 <= value < 2**32


def field_value(values: Mapping[int, int], field: isa.BitField) -> int:
    """Read *field* from register words, an unwritten register reading zero."""
    return field.get(values.get(field.addr, 0))


def validate(image: "Image") -> None:
    """Reject invalid procedure metadata, empty programs and incompatible I/O."""
    if not image.procedure or not image.procedure.isascii() or not image.procedure.isprintable():
        raise ValueError(f"invalid procedure name {image.procedure!r}")
    if not is_u32(image.record_count):
        raise ValueError(f"record count {image.record_count!r} is not a u32")
    if image.format_code is not None and not is_u32(image.format_code):
        raise ValueError(f"format code {image.format_code!r} is not a u32")
    if not is_u32(image.scratch_bytes):
        raise ValueError(f"scratch size {image.scratch_bytes!r} is not a u32")
    if image.scratch_bytes:
        raise ValueError("nonzero scratch_bytes mapping is not decoded yet")
    if len(image.procedure) >= PROCEDURE_BYTES:
        raise ValueError("procedure name overlaps the binding record symbol region")
    short = [item.name[:NAME_BYTES] for item in image.io]
    if len(set(short)) != len(short):
        raise ValueError("I/O names collide after truncation to eight bytes")
    if not image.program.tds:
        raise ValueError("image text must contain a descriptor")
    for item in image.io:
        item.validate(image.layout)
    outputs = {item.name for item in image.bindings() if item.role == BINDING_ROLE_OUTPUT}
    slots = base_slots(image.program)
    for reloc in image.relocs:
        slot = slots.get(reloc.word_offset)
        if slot is not None and slot[1] == DST and reloc.symbol not in outputs:
            raise ValueError(f"GMEM intermediate {reloc.symbol!r} mapping is not decoded yet")
    for td in image.program.tds:
        td.validate()
    isa.dma.check_continuations(td.values() for td in image.program.tds)
    require_relocated(image, slots)


def require_relocated(image: "Image", slots: Mapping[int, tuple[int, int, int]]) -> None:
    """Reject a program with a base address that no relocation binds to a buffer.

    The loader maps only the buffers this container declares. A DMA whose
    base stays at its placeholder reads or writes outside every mapping,
    which the engine's address translation reports as a fatal fault.
    """
    keys = [(index, addr) for index, addr, _ in slots.values()]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate base register in a descriptor")
    kernel = isa.KernelDMA
    accesses = {}
    for index, td in enumerate(image.program.tds):
        values = td.values()
        for enabled, base in (
            (kernel.kernel_dma_src_enable, kernel.kernel_dma_src_base_lo),
            (kernel.nonlinear_lut_enable, kernel.kernel_dma_lut_base),
        ):
            if field_value(values, enabled) and (index, base.addr) not in keys:
                raise ValueError("enabled coefficient reader has no relocated base")
        for transfer in isa.dma.transfers(values):
            accesses[index, transfer.register] = transfer
            if (index, transfer.register) not in keys:
                raise ValueError(f"enabled {transfer.name} DMA has no base word")
    require_dma_bindings(image, slots, accesses)
    require_coefficient_bindings(image, slots)


def require_dma_bindings(
    image: "Image", slots: Mapping[int, tuple[int, int, int]], accesses: Mapping
) -> None:
    """Bind each DMA base once, to the signature buffer its transfer reads or writes in range."""
    dma = {word for word, (_, addr, _) in slots.items() if addr in DMA_BASES}
    bound = [reloc.word_offset for reloc in image.relocs]
    if len(bound) != len(set(bound)):
        raise ValueError("duplicate DMA binding")
    if dma - set(bound):
        raise ValueError("DMA base is bound to no I/O buffer")
    if set(bound) - dma:
        raise ValueError("DMA binding does not address a DMA slot")
    buffers = {item.name: item for item in image.io}
    bindings = image.bindings()
    segments = [item.name for item in bindings]
    outputs = {item.name for item in bindings if item.role == BINDING_ROLE_OUTPUT}
    flags = {first: packet.flags for _, first, packet in packet_slots(image.program)}
    for reloc in image.relocs:
        if reloc.symbol not in buffers:
            raise ValueError(f"unknown I/O symbol {reloc.symbol!r}")
        index, register, addend = slots[reloc.word_offset]
        transfer = accesses.get((index, register))
        if transfer is None:
            raise ValueError(f"DMA base {register:#x} has no enabled transfer")
        if (register == DST) != (reloc.symbol in outputs):
            raise ValueError(f"{transfer.name} DMA symbol-to-signature mapping is not decoded yet")
        if flags[reloc.word_offset] != isa.io_relocation(segments.index(reloc.symbol)):
            raise ValueError(f"{transfer.name} DMA relocation flags do not name {reloc.symbol!r}")
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


def require_coefficient_bindings(image: "Image", slots: Mapping[int, tuple[int, int, int]]) -> None:
    """Bind each coefficient slot once, to a consecutive block whose range its reads stay in."""
    end = 0
    for coefficient in image.coefficients:
        coefficient.validate()
        if coefficient.layout.base != end:
            raise ValueError("coefficient blocks must be consecutive in first-use order")
        end += coefficient.layout.size
    expected = {word for word, (_, addr, _) in slots.items() if addr in isa.COEFFICIENT_RELOCATIONS}
    bound = [reloc.word_offset for reloc in image.coefficient_relocs]
    if len(bound) != len(set(bound)):
        raise ValueError("duplicate coefficient binding")
    if expected != set(bound):
        raise ValueError("coefficient base is bound to no coefficient section or slot")
    flags = {first: packet.flags for _, first, packet in packet_slots(image.program) if packet.flags}
    segment = isa.io_relocation(len(image.io))
    if any(flags.get(word, segment) != segment for word in bound):
        raise ValueError("coefficient relocation flags do not name the kernel segment")
    first_uses = tuple(dict.fromkeys(reloc.coefficient for reloc in sorted(
        image.coefficient_relocs, key=lambda item: item.word_offset
    )))
    if first_uses != tuple(range(len(image.coefficients))):
        raise ValueError("coefficient blocks do not match first consumption order")
    kernel = isa.KernelDMA
    for reloc in image.coefficient_relocs:
        if type(reloc.coefficient) is not int or not 0 <= reloc.coefficient < len(image.coefficients):
            raise ValueError("coefficient binding names a missing block")
        layout = image.coefficients[reloc.coefficient].layout
        td_index, register, addend = slots[reloc.word_offset]
        values = {addr: value for packet in image.program.tds[td_index].packets
                  for addr, value in packet.writes}
        if register == kernel.kernel_dma_src_base_lo.addr:
            if addend != layout.base:
                raise ValueError("coefficient base disagrees with its planned block")
            isa.dma.check_coefficient_lanes(layout.lanes, sum(
                field_value(values, field) for field in kernel.kernel_dma_src_coeff_dma_en
            ))
            if (field_value(values, isa.NE.nonlinear_mode) == isa.NENonlinear.LUT) != bool(layout.lut):
                raise ValueError("lookup-table mode disagrees with its coefficient lanes")
            if field_value(values, isa.NE.ne_bias) != layout.bias:
                raise ValueError("bias mode disagrees with its coefficient lanes")
            for enabled, offset, size in zip(
                kernel.kernel_dma_src_coeff_dma_en,
                kernel.kernel_dma_src_coeff_base_offset,
                kernel.kernel_dma_src_coeff_mem_buffer_size,
            ):
                if not field_value(values, enabled):
                    continue
                start = field_value(values, offset) * 64
                length = field_value(values, size) * 64 or 64
                if start % layout.stride or length != layout.stride or start + length > layout.size:
                    raise ValueError("coefficient DMA range leaves its planned block")
        elif register == kernel.kernel_dma_lut_base.addr and layout.lut:
            length = field_value(values, kernel.nonlinear_lut_size) * 64
            if not length or not layout.base <= addend < addend + length <= layout.base + layout.size:
                raise ValueError("lookup table range leaves its planned block")
        else:
            raise ValueError(f"coefficient register {register:#x} read range is not decoded yet")


__all__ = ["require_relocated", "validate"]
