"""Real convolution images bind DMA and coefficient accesses to their buffers."""

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from tests.core import _corpus
from tilefoundry import analysis
from tilefoundry.schedule import finalize
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.codegen import hwx, module
from tilefoundry_apple.ane.codegen.hwx import coefficients, macho


@pytest.fixture
def image():
    path = Path(__file__).parents[1] / "fixtures" / "scheduled" / "conv_1x1.py"
    loaded = _corpus.Case(path).load()
    checked = analysis.check_program(loaded, loaded.entry_function())
    fn = finalize(analysis.analyze(*checked, analysis=("memory",))).entry_function()
    return module.function_image(fn, {"w": b"\x11\x22" * 1024})


def test_coefficient_image_and_symbols(image):
    payload = image.to_bytes()
    slots = hwx.base_slots(image.program)
    coefficient_bases = [slots[reloc.word_offset][2] for reloc in image.coefficient_relocs]
    assert coefficient_bases == [0]
    for index, td in enumerate(image.program.tds):
        values = {addr: value for packet in td.packets for addr, value in packet.writes}
        assert td.tid == index
        assert isa.KernelDMA.kernel_dma_coeff_tag.get(values[0x1900]) == index + 1
        assert isa.TileDMADst.tile_dma_dst_user_tag.get(values[isa.TileDMADst.tile_dma_dst_user_tag.addr]) == index
    bindings = {(slots[reloc.word_offset][0], slots[reloc.word_offset][1]): reloc.symbol for reloc in image.relocs}
    source = isa.TileDMASrc.tile_dma_src_base_addr_lo.addr
    destination = isa.TileDMADst.tile_dma_dst_base_addr_lo.addr
    assert bindings == {(0, source): "x", (0, destination): "out"}
    assert {binding.name: binding.role for binding in image.bindings()} == {"x": 1, "out": 2}
    cursor = macho.HEADER.size
    for _ in range(macho.HEADER.unpack_from(payload)[4]):
        command, size = struct.unpack_from("<II", payload, cursor)
        if command == macho.LC_SEGMENT_64:
            segment = macho.SEGMENT.unpack_from(payload, cursor)
            if segment[2].rstrip(b"\0") == b"__KERN_0":
                kernel_address, kernel_offset = segment[3], segment[5]
        elif command == macho.LC_SYMTAB:
            _, _, symbol_offset, symbol_count, string_offset, string_size = macho.SYMTAB.unpack_from(payload, cursor)
        cursor += size
    strings = payload[string_offset : string_offset + string_size]
    symbols = {}
    for index in range(symbol_count):
        name, _, _, _, address = macho.SYMBOL.unpack_from(payload, symbol_offset + index * macho.SYMBOL.size)
        symbols[strings[name:].split(b"\0", 1)[0].decode()] = address
    section = coefficients.coefficient_section(image.coefficients)
    assert payload[kernel_offset : kernel_offset + len(section)] == section
    for index, coefficient in enumerate(image.coefficients):
        layout = coefficient.layout
        for lane in range(layout.lanes):
            address = symbols[f"coeff_ne_{lane}"]
            assert address == kernel_address + layout.base + lane * layout.stride
            start = kernel_offset + address - kernel_address
            assert payload[start : start + layout.lane_bytes] == coefficient.values[:layout.lane_bytes]


@pytest.mark.parametrize("base", (0, 32 * 18))
def test_convolution_rejects_overlapping_l2_ranges_even_with_alias_bits(image, base):
    td = image.program.tds[0]
    field = isa.L2.l2_result_address
    packets = tuple(replace(packet, writes=tuple(
        (addr, field.put(word, base // 16) if addr == field.addr else word)
        for addr, word in packet.writes
    )) for packet in td.packets)
    invalid = replace(image, program=isa.Program((replace(td, packets=packets),)))
    with pytest.raises(ValueError, match="L2 src and result ranges overlap"):
        invalid.to_bytes()


@pytest.mark.parametrize("fault", ("missing_dma", "duplicate_dma", "unknown_dma", "missing_block", "missing_coefficient", "duplicate_coefficient", "wrong_owner", "overlap"))
def test_invalid_image_bindings_are_rejected(image, fault):
    if fault == "missing_dma":
        invalid = replace(image, relocs=image.relocs[:-1])
    elif fault == "duplicate_dma":
        invalid = replace(image, relocs=(*image.relocs, image.relocs[0]))
    elif fault == "unknown_dma":
        invalid = replace(image, relocs=(replace(image.relocs[0], symbol="missing"), *image.relocs[1:]))
    elif fault == "missing_block":
        invalid = replace(image, coefficients=())
    elif fault == "missing_coefficient":
        invalid = replace(image, coefficient_relocs=image.coefficient_relocs[:-1])
    elif fault == "duplicate_coefficient":
        invalid = replace(image, coefficient_relocs=(*image.coefficient_relocs, image.coefficient_relocs[0]))
    elif fault == "wrong_owner":
        invalid = replace(image, coefficient_relocs=(replace(image.coefficient_relocs[0], coefficient=1),))
    else:
        block = image.coefficients[0]
        invalid = replace(image, coefficients=(replace(block, layout=replace(block.layout, base=64)),))
    with pytest.raises(ValueError):
        invalid.to_bytes()


@pytest.mark.parametrize("register,value", ((0x1908, 4096), (0x1908, 2048 - 64), (0x195C, 2048), (0x1998, 256)))
def test_out_of_range_base_or_lane_read_is_rejected(image, register, value):
    td = image.program.tds[0]
    packets = tuple(replace(packet, writes=tuple((addr, value if addr == register else word) for addr, word in packet.writes)) for packet in td.packets)
    invalid = replace(image, program=isa.Program((replace(td, packets=packets), *image.program.tds[1:])))
    with pytest.raises(ValueError):
        invalid.to_bytes()
