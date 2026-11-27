"""L2 placement preserves live intermediates and rejects uninitialized continuations."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.core import _corpus
from tilefoundry import analysis
from tilefoundry.ir import expr, types
from tilefoundry.schedule import finalize
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.codegen import allocation, context, hwx, module
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic


def tensor(name, storage, height=16):
    """Construct a canonical static feature map in the requested storage."""
    desc = isa.TensorDesc.from_shape("f16", (1, 32, height, 16))
    strides = (32 * height * 16, height * 16, 16, 1) if storage is expr.StorageKind.SMEM else desc.strides
    return expr.Var(name, type=types.TensorType(
        desc.shape, expr.DType.f16, types.Layout(desc.shape, strides), storage,
    ))


def calls(height=16):
    """Return two simultaneously live SMEM results and their later consumers."""
    x, out = (tensor(name, expr.StorageKind.GMEM, height) for name in ("x", "out"))
    a, b = (tensor(name, expr.StorageKind.SMEM, height) for name in ("a", "b"))
    return [expr.Call(tir.Unary(), args, type=types.TupleType.Void)
            for args in ((a, x), (b, x), (out, a), (out, b))]


def test_overlapping_smem_lifetimes_have_disjoint_guarded_ranges():
    issued = calls()
    ctx = context.Context()
    ctx.plan(issued)
    a, b = issued[0].args[0], issued[1].args[0]
    size = context.tensor_desc_of(a.type, "a").staging_extent()
    assert abs(ctx.address_of(a) - ctx.address_of(b)) >= size
    for call in issued:
        ctx.emit_call(call, "allocation")
    for parts in ctx.passes:
        values = isa.TD.from_values(intrinsic.splice(*parts)).values()
        isa.rules.check(values)


def test_total_live_buffers_over_capacity_fail_during_compilation():
    with pytest.raises(ValueError, match="L2 live buffers exceed capacity 2097152"):
        context.Context().plan(calls(700))


def test_expired_intervals_reuse_space_and_capacity_boundary_is_inclusive():
    placed = allocation.allocate((allocation.Interval("a", 0, 1, isa.dma.L2_CAPACITY),
                                  allocation.Interval("b", 2, 3, isa.dma.L2_CAPACITY)))
    assert placed == {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="exceed capacity"):
        allocation.allocate((allocation.Interval("a", 0, 1, isa.dma.L2_CAPACITY),
                             allocation.Interval("b", 1, 2, 16)))


def test_smem_read_requires_an_earlier_producer():
    with pytest.raises(ValueError, match="requires a preceding producer"):
        context.Context().plan(calls()[2:])


@pytest.mark.parametrize("height", (1, 16, 1000))
def test_convolution_allocates_complete_neural_source_and_result_buffers(height):
    src, dst = (tensor(name, expr.StorageKind.GMEM, height) for name in ("x", "out"))
    shape = (32, 32, 1, 1)
    weight = expr.Var("w", is_const=True, type=types.TensorType(
        shape, expr.DType.f16, types.Layout(shape, (32, 1, 1, 1)), expr.StorageKind.GMEM,
    ))
    call = expr.Call(tir.Conv(), (dst, src, weight), type=types.TupleType.Void)
    ctx = context.Context()
    extent = context.tensor_desc_of(src.type, "src").staging_extent(isa.Engine.NEURAL)
    if 2 * extent > isa.dma.L2_CAPACITY:
        with pytest.raises(ValueError, match="L2 live buffers exceed capacity"):
            ctx.plan([call])
        return
    ctx.plan([call])
    ctx.emit_call(call, "convolution")
    values = isa.TD.from_values(intrinsic.splice(*ctx.passes[0])).values()
    ranges = isa.dma.check_l2(values, isa.dma.geometry(values))
    assert ranges == (("src", ctx.staging[0, 1], ctx.staging[0, 1] + extent),
                      ("result", ctx.staging[0, 0], ctx.staging[0, 0] + extent))
    assert ranges[0][2] <= ranges[1][1] or ranges[1][2] <= ranges[0][1]


def test_convolution_l2_source_requires_decoded_small_source_mapping():
    dst, src, weight = (tensor("out", expr.StorageKind.GMEM),
                        tensor("h", expr.StorageKind.SMEM), tensor("w", expr.StorageKind.GMEM))
    call = expr.Call(tir.Conv(), (dst, src, weight), type=types.TupleType.Void)
    with pytest.raises(ValueError, match="full-height L2 source small_source_mode mapping is not decoded yet"):
        context.Context().emit_call(call, "convolution")


def chain_image():
    """Compile the scheduled chain through ordinary lowering and HWX construction."""
    path = Path(__file__).parents[1] / "fixtures" / "scheduled" / "unary_chain.py"
    loaded = _corpus.Case(path).load()
    lowered = finalize(analysis.analyze(*analysis.check_program(loaded, loaded.entry_function()),
                                        analysis=("memory",)))
    return module.function_image(lowered.entry_function())


def test_chain_has_only_external_dma_and_matching_producer_consumer_slots():
    image = chain_image()
    assert image.to_bytes()
    first, second = (td.values() for td in image.program.tds)
    l2 = isa.L2
    assert l2.l2_result_address.get(first[l2.l2_result_address.addr]) == l2.l2_src_base_address.get(second[l2.l2_src_base_address.addr])
    assert l2.l2_result_type.get(first[l2.l2_result_type.addr]) == isa.L2Access.L2_READ
    assert l2.l2_src_type.get(second[l2.l2_src_type.addr]) == isa.L2Access.L2_READ
    slots = {index: [] for index in (0, 1)}
    for index, register, _ in hwx.base_slots(image.program).values():
        slots[index].append(register)
    assert slots == {0: [isa.TileDMASrc.tile_dma_src_base_addr_lo.addr],
                         1: [isa.TileDMADst.tile_dma_dst_base_addr_lo.addr]}


def test_image_rejects_l2_source_without_a_matching_producer():
    image = chain_image()
    second = image.program.tds[1]
    with pytest.raises(ValueError, match="matching unmodified preceding result"):
        replace(image, program=isa.Program((second,))).to_bytes()


def test_l2_producer_and_consumer_address_every_element_identically():
    image = chain_image()
    producer, consumer = (td.values() for td in image.program.tds)
    l2 = isa.L2
    written = l2.l2_result_address.get(producer[l2.l2_result_address.addr]) * 16
    read = l2.l2_src_base_address.get(consumer[l2.l2_src_base_address.addr]) * 16
    write_channel = l2.l2_result_channel_stride.get(producer[l2.l2_result_channel_stride.addr]) * 16
    write_row = l2.l2_result_row_stride.get(producer[l2.l2_result_row_stride.addr]) * 16
    read_channel = l2.l2_src_channel_stride.get(consumer[l2.l2_src_channel_stride.addr]) * 16
    read_row = l2.l2_src_row_stride.get(consumer[l2.l2_src_row_stride.addr]) * 16
    _, channels, height, width = image.io[0].desc.shape
    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                assert written + channel * write_channel + row * write_row + column * 2 == (
                    read + channel * read_channel + row * read_row + column * 2
                )


@pytest.mark.parametrize("pitch,wrong", (
    ("channel_stride", 0), ("row_stride", 0), ("channel_stride", 1), ("row_stride", 64),
))
def test_l2_result_pitches_cannot_contradict_the_allocated_layout(pitch, wrong):
    image = chain_image()
    first, second = image.program.tds
    values = first.values()
    field = isa.L2.fields["l2_result_" + pitch]
    values[field.addr] = field.put(values[field.addr], wrong)
    stream = isa.Program((isa.TD(isa.serialize(values)), second))
    relocs = tuple(hwx.Reloc(word, "out" if register == isa.TileDMADst.tile_dma_dst_base_addr_lo.addr else "x")
                   for word, (_, register, _) in hwx.base_slots(stream).items())
    with pytest.raises(ValueError, match="must explicitly match|tile pitch is not decoded yet; expected"):
        replace(image, program=stream, relocs=relocs).to_bytes()


@pytest.mark.parametrize("fault", ("moved", "overwritten"))
def test_image_rejects_moved_or_overwritten_l2_results(fault):
    image = chain_image()
    first, second = image.program.tds
    if fault == "moved":
        values = second.values()
        field = isa.L2.l2_src_base_address
        values[field.addr] = field.put(values[field.addr], 4096)
        descriptors = (first, isa.TD(isa.serialize(values)))
    else:
        desc = image.io[0].desc
        clobber = isa.TD.from_values(intrinsic.splice(
            intrinsic.load(desc), intrinsic.compute_pe(isa.TaskType.PE_EW_UNARY, desc),
            intrinsic.store(desc),
        ))
        descriptors = (first, clobber, second)
    stream = isa.Program(descriptors)
    relocs = tuple(hwx.Reloc(word, "out" if register == isa.TileDMADst.tile_dma_dst_base_addr_lo.addr else "x")
                   for word, (_, register, _) in hwx.base_slots(stream).items())
    with pytest.raises(ValueError, match="matching unmodified preceding result"):
        replace(image, program=stream, relocs=relocs).to_bytes()
