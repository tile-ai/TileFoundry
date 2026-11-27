"""Container generation rejects DMA, L2 and record integrity failures offline."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.core import _corpus
from tilefoundry import analysis
from tilefoundry.schedule import finalize
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.codegen import hwx, module, program
from tilefoundry_apple.ane.codegen.hwx import coefficients
from tilefoundry_apple.ane.isa import intrinsic


def binary_image(shape=(1, 32, 16, 16), changes=None, omitted=()):
    """Build an unchecked descriptor to exercise the container's validation boundary."""
    desc = isa.TensorDesc.from_shape("f16", shape)
    values = intrinsic.splice(
        intrinsic.load(desc), intrinsic.load_second(desc),
        intrinsic.compute_pe(isa.TaskType.PE_EW_BINARY, desc, second_source=2, source_routing=3),
        intrinsic.store(desc, staged=2),
    )
    values.update(changes or {})
    values = {addr: word for addr, word in values.items() if addr not in omitted}
    stream = isa.Program((isa.TD(isa.serialize(values)),))
    names = {0x1110: "x", 0x1128: "z", 0x1508: "y"}
    relocs = tuple(hwx.Reloc(word, names[addr])
                   for word, (_, addr, _) in hwx.base_slots(stream).items())
    return hwx.Image(stream, tuple(hwx.TensorBinding(name, desc) for name in ("x", "z", "y")), relocs)


@pytest.mark.parametrize("fault", ("src", "src2", "dst", "row", "stride", "format", "batch", "dtype"))
def test_a01_dma_end_must_fit_bound_tensor(fault):
    changes = {"src": (0x1110, 32704), "src2": (0x1128, 32704), "dst": (0x1508, 32704),
               "row": (0x1118, 0xFFFFFFC0), "stride": (0x111C, 2048),
               "format": (0x1150, 0)}
    image = binary_image(changes=dict((changes[fault],)) if fault in changes else None)
    if fault in ("batch", "dtype"):
        desc = isa.TensorDesc.from_shape("f32" if fault == "dtype" else "f16",
                                         (2 if fault == "batch" else 1, 32, 16, 16))
        image = replace(image, io=(replace(image.io[0], desc=desc), *image.io[1:]))
    with pytest.raises(ValueError, match="exceeds TensorDesc|does not cover|strides contradict|not decoded yet"):
        image.to_bytes()


@pytest.mark.parametrize("fault", ("src", "src2", "dst", "orphan", "flags"))
def test_a02_enabled_dma_requires_base(fault):
    register = {"src": 0x1110, "src2": 0x1128, "dst": 0x1508}.get(fault, 0x1110)
    if fault in ("src", "src2", "dst"):
        image = binary_image(omitted=(register, register + 4))
    else:
        image = binary_image()
        td = image.program.tds[0]
        packets = tuple(replace(packet, writes=packet.writes[1:]) if fault == "orphan"
                        and packet.writes[0][0] == register else
                        replace(packet, flags=0) if fault == "flags"
                        and packet.writes[0][0] == register else packet for packet in td.packets)
        image = replace(image, program=isa.Program((replace(td, packets=packets),)))
    with pytest.raises(ValueError, match="has no base word|orphan high word|relocation packet flags"):
        image.to_bytes()


@pytest.mark.parametrize("fault", ("capacity", "overlap", "boundary", "tail"))
def test_a03_l2_live_ranges_are_bounded_and_disjoint(fault):
    desc = isa.TensorDesc.from_shape("f16", (1, 32, 1000 if fault == "capacity" else 16, 16))
    values = intrinsic.splice(intrinsic.load(desc),
                             intrinsic.compute_pe(isa.TaskType.PE_EW_UNARY, desc),
                             intrinsic.store(desc))
    if fault == "overlap":
        values[isa.L2.l2_result_base_addr.addr] = 0
    elif fault in ("boundary", "tail"):
        values[isa.L2.l2_result_base_addr.addr] = isa.dma.L2_CAPACITY - desc.staging_extent() + (16 if fault == "tail" else 0)
    stream = isa.Program((isa.TD(isa.serialize(values)),))
    relocs = tuple(hwx.Reloc(word, "y" if addr == 0x1508 else "x")
                   for word, (_, addr, _) in hwx.base_slots(stream).items())
    image = hwx.Image(stream, tuple(hwx.TensorBinding(name, desc) for name in ("x", "y")), relocs)
    if fault == "boundary":
        assert image.to_bytes()
    else:
        with pytest.raises(ValueError, match="L2 .* (exceeds capacity|ranges overlap)"):
            image.to_bytes()


def test_a04_signature_roles_reject_gmem_intermediate_and_preserve_coefficients():
    path = Path(__file__).parents[1] / "fixtures" / "scheduled" / "multi_weight.py"
    loaded = _corpus.Case(path).load()
    fn = finalize(analysis.analyze(*analysis.check_program(loaded, loaded.entry_function()),
                                   analysis=("memory",))).entry_function()
    image = module.function_image(fn, {"first": b"\x11\x22" * 1024, "second": b"\x33\x44" * 1024})
    assert {binding.name: binding.role for binding in image.bindings()} == {"x": 1, "out": 2}
    assert [block.layout.base for block in image.coefficients] == [0, 2048]
    slots = hwx.base_slots(image.program)
    assert [slots[reloc.word_offset][2] for reloc in image.coefficient_relocs] == [0, 2048]
    section = coefficients.coefficient_section(image.coefficients)
    for block in image.coefficients:
        for lane in range(block.layout.lanes):
            start = block.layout.base + lane * block.layout.stride
            assert section[start:start + block.layout.lane_bytes] == block.values[:block.layout.lane_bytes]
    assert image.to_bytes()
    destination = next(reloc for reloc in image.relocs if slots[reloc.word_offset][1] == 0x1508)
    invalid = replace(image, io=(image.io[0], hwx.TensorBinding("scratch", image.io[1].desc), image.io[1]),
                      relocs=tuple(replace(reloc, symbol="scratch") if reloc == destination else reloc
                                   for reloc in image.relocs))
    with pytest.raises(ValueError, match="GMEM intermediate 'scratch' mapping is not decoded yet"):
        invalid.to_bytes()


@pytest.mark.parametrize("fault", ("width", "task", "ne_op", "pe_op", "packet", "row", "plane", "tile"))
def test_a16_raw_words_and_packets_obey_field_and_geometry_rules(fault):
    changes = {"width": (0x0000, 16 << 16), "task": (0x0038, 15 << 4),
               "ne_op": (0x0D04, 7), "pe_op": (0x0900, 7 << 2),
               "row": (0x1118, 64), "plane": (0x111C, 64), "tile": (0x002C, 17)}
    if fault == "packet":
        image = hwx.Image(isa.Program((isa.TD((isa.Packet("run", ((0x4000, 123),)),)),)), (), (), output_count=0)
    else:
        register, word = changes[fault]
        image = binary_image((1, 32, 16, 64), {register: word})
    with pytest.raises(ValueError, match="allowed|configuration block|cover|tile_height|reserved bits"):
        image.to_bytes()
    if fault in ("width", "task", "ne_op", "pe_op"):
        with pytest.raises(ValueError, match="allowed|reserved bits"):
            isa.TD.from_values({register: word})


@pytest.mark.parametrize("fault", ("procedure", "short_names"))
def test_a20_fixed_record_names_must_not_overlap_or_collide(fault):
    image = binary_image()
    if fault == "procedure":
        image = replace(image, procedure="abcdefghijklmno")
    else:
        names = {"x": "abcdefgh1", "z": "abcdefgh2", "y": "y"}
        image = replace(image, io=tuple(replace(item, name=names[item.name]) for item in image.io),
                        relocs=tuple(replace(reloc, symbol=names[reloc.symbol]) for reloc in image.relocs))
    with pytest.raises(ValueError, match="procedure name overlaps|I/O names collide"):
        image.to_bytes()


def test_a21_scratch_mapping_must_be_decoded():
    stream = program.to_program((({},),))
    with pytest.raises(ValueError, match="nonzero scratch_bytes mapping is not decoded yet"):
        hwx.Image(stream, (), (), output_count=0, scratch_bytes=16385).to_bytes()
