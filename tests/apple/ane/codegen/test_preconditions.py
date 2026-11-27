"""Engine emission refuses undecoded semantics, schedules and conflicting field writes."""

import pytest

from tilefoundry.ir import expr, types
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.codegen import context, hwx, program
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic


def tensor(name, shape=(1, 32, 16, 16), *, constant=False):
    """Construct one canonical feature map or contiguous constant kernel operand."""
    if constant:
        _, channels, height, width = shape
        strides = (channels * height * width, height * width, width, 1)
    else:
        strides = isa.TensorDesc.from_shape("f16", shape).strides
    return expr.Var(name, is_const=constant, type=types.TensorType(
        shape, expr.DType.f16, types.Layout(shape, strides), expr.StorageKind.GMEM,
    ))


def image_bytes(op, shape=(1, 32, 16, 16), *, bias=False, ctx=None):
    """Emit one call and assemble its image through the public container boundary."""
    ctx = context.Context() if ctx is None else ctx
    dst_shape = shape
    if isinstance(op, tir.Reduce):
        dst_shape = (shape[0], shape[1], 1, 1)
    elif isinstance(op, tir.Conv):
        dst_shape = (shape[0], 32, shape[2], shape[3])
    src, dst = tensor("x", shape), tensor("y", dst_shape)
    args = (dst, src)
    inputs = (src,)
    if isinstance(op, tir.Binary):
        rhs = tensor("z", shape)
        args = (*args, rhs)
        inputs = (*inputs, rhs)
    elif isinstance(op, tir.Conv):
        args = (*args, tensor("w", (32, shape[1], 1, 1), constant=True))
        if bias:
            args = (*args, expr.Var("b", type=types.TensorType(
                (32,), expr.DType.f16, types.Layout((32,), (1,)), expr.StorageKind.GMEM,
            )))
    ctx.emit_call(expr.Call(op, args, type=types.TupleType.Void), "precondition")
    stream = program.to_program(ctx.passes)
    names = {0x1110: "x", 0x1128: "z", 0x1508: "y"}
    relocs = tuple(hwx.Reloc(word, names[addr]) for word, (_, addr, _) in hwx.base_slots(stream).items()
                   if addr not in isa.COEFFICIENT_RELOCATIONS)
    blocks = tuple(hwx.Coefficient(layout, bytes(layout.lanes * layout.lane_bytes))
                   for layout in ctx.coefficients.values())
    bindings = tuple(hwx.TensorBinding(var.name, context.tensor_desc_of(var.type, var.name))
                     for var in (*inputs, dst))
    return hwx.Image(stream, bindings, relocs, coefficients=blocks,
                     coefficient_relocs=hwx.coefficient_relocations(stream, {index: index for index in range(len(blocks))})).to_bytes()


def test_a06_convolution_bias_is_refused():
    with pytest.raises(ValueError, match="bias operand encoding is not decoded yet"):
        image_bytes(tir.Conv(), bias=True)


@pytest.mark.parametrize("perm", ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)))
def test_a07_undecoded_bypass_permutations_are_refused(perm):
    with pytest.raises(ValueError, match="perm .* is not decoded yet"):
        image_bytes(tir.Bypass(perm=perm), (1, 32, 32, 32))


@pytest.mark.parametrize("width,code", ((31, None), (63, None), (120, None), (31, 2), (16, 2), (16, -1)))
def test_a08_width_code_requires_exact_shape_subdivision(width, code):
    with pytest.raises(ValueError, match="width code .* is not decoded yet|width_code .* (contradicts|must be)"):
        image_bytes(tir.Bypass(width_code=code), (1, 32, 4, width))


@pytest.mark.parametrize("name", ("unary", "binary", "pool", "bypass"))
@pytest.mark.parametrize("producer", ("none", "matching", "mismatched"))
def test_a09_chaining_requires_a_verified_producer_mapping(name, producer):
    ctx = context.Context()
    if producer != "none":
        shape = (1, 32, 8 if producer == "mismatched" else 16, 16)
        src, dst = tensor("previous", shape), tensor("x", shape)
        op = tir.Unary(tile_height=1 if producer == "mismatched" else None)
        ctx.emit_call(expr.Call(op, (dst, src), type=types.TupleType.Void), "producer")
        if producer == "mismatched":
            ctx.bind(dst, dst, 64)
    ops = {
        "unary": tir.Unary(chained_source=True),
        "binary": tir.Binary(kind=expr.BinaryKind.ADD, chained_source=True),
        "pool": tir.Pool(kernel_size=(1, 1), stride=(1, 1), kind=expr.PoolKind.MAX, chained_source=True),
        "bypass": tir.Bypass(chained_source=True),
    }
    reason = "chained_source requires a preceding pass" if producer == "none" else "continuation mapping is not decoded yet"
    with pytest.raises(ValueError, match=reason):
        image_bytes(ops[name], ctx=ctx)


def test_a10_convolution_tile_plan_must_cover_the_compute_height():
    with pytest.raises(ValueError, match="tile plan is not decoded yet"):
        image_bytes(tir.Conv(tile_height=1))


@pytest.mark.parametrize("shape,rows,width", (
    ((1, 31, 15, 32), None, None), ((1, 24, 11, 64), None, None),
    ((1, 17, 4, 16), None, None), ((1, 32, 16, 16), 1, None),
    ((1, 32, 16, 64), None, 16),
))
def test_a12_neural_guards_require_sampled_staging_geometry(shape, rows, width):
    op = tir.Bypass(tile_height=rows) if width is None else tir.Conv(tile_width=width)
    with pytest.raises(ValueError, match="neural L2 staging guard .* is not decoded yet"):
        image_bytes(op, shape)


def test_a13_coefficient_lanes_must_match_enabled_dma_slots():
    with pytest.raises(ValueError, match="coefficient lane count 16 with 8 DMA slots is not decoded yet"):
        image_bytes(tir.Conv(), (1, 8, 4, 16))
    desc = isa.TensorDesc.from_shape("f16", (1, 32, 4, 16))
    values = intrinsic.splice(
        intrinsic.load(desc, isa.Engine.NEURAL),
        intrinsic.convolution(desc, desc, index=0, ocg_size=1, read_config=False),
    )
    for field in isa.KernelDMA.kernel_dma_src_coeff_dma_en[8:]:
        values[field.addr] = field.put(values[field.addr], 0)
    stream = isa.Program((isa.TD(isa.serialize(values)),))
    slots = hwx.base_slots(stream)
    relocs = tuple(hwx.Reloc(word, "y" if addr == 0x1508 else "x")
                   for word, (_, addr, _) in slots.items() if addr not in isa.COEFFICIENT_RELOCATIONS)
    image = hwx.Image(stream, tuple(hwx.TensorBinding(name, desc) for name in ("x", "y")), relocs,
                      coefficients=(hwx.Coefficient(hwx.CoefficientLayout((32, 32, 1, 1)), bytes(2048)),),
                      coefficient_relocs=hwx.coefficient_relocations(stream, {0: 0}))
    with pytest.raises(ValueError, match="coefficient lane count 16 with 8 DMA slots is not decoded yet"):
        image.to_bytes()


@pytest.mark.parametrize("dtype,batch", (("f32", 1), ("f16", 2)))
def test_a14_low_level_entries_require_fp16_single_batch(dtype, batch):
    desc = isa.TensorDesc.from_shape(dtype, (batch, 32, 16, 16))
    for emit in (
        lambda: intrinsic.load(desc), lambda: intrinsic.load_second(desc),
        lambda: intrinsic.load_chained(desc), lambda: intrinsic.store(desc),
        lambda: intrinsic.compute_pe(isa.TaskType.PE_EW_UNARY, desc),
        lambda: intrinsic.compute_ne(isa.NEOp.BYPASS, desc),
        lambda: intrinsic.staged_copy(desc), lambda: intrinsic.activation(desc),
        lambda: intrinsic.convolution(desc, desc, index=0, ocg_size=1),
    ):
        with pytest.raises(ValueError, match="dtype or batch is not decoded yet; expected fp16 N=1"):
            emit()


@pytest.mark.parametrize("first,second", (
    (isa.PE().pack(op=isa.PEOp.MAX), isa.PE().pack(op=isa.PEOp.MUL)),
    (isa.Common().pack(win=16), isa.Common().pack(win=32)),
    (isa.PE().pack(op=isa.PEOp.ADD), isa.PE().pack(op=isa.PEOp.MUL)),
))
def test_a15_splicing_rejects_conflicts_including_zero_writes(first, second):
    assert intrinsic.splice(first, first) == first
    with pytest.raises(ValueError, match="conflicting primitive writes"):
        program.to_program(((first, second),))
    with pytest.raises(ValueError, match="conflicting primitive writes"):
        intrinsic.splice(intrinsic.splice(first), second)
    combined = intrinsic.splice(isa.Common().pack(win=16), isa.Common().pack(hin=32))
    assert isa.Common.win.get(combined[0]) == 16
    assert isa.Common.hin.get(combined[0]) == 32


@pytest.mark.parametrize("name,attrs", (
    ("binary", {"tile_width": -9, "width_code": -4, "ocg_size": 99, "omit_ocg": True}),
    *((name, {attr: value}) for name in ("unary", "binary", "pool", "reduce")
      for attr, value in (("width_code", 1), ("ocg_size", 1), ("omit_ocg", True))),
    ("reduce", {"chained_source": True}),
    *(("conv", {attr: value}) for attr, value in (("patch", (4, 4)), ("width_code", 1), ("omit_ocg", True))),
    *(("bypass", {attr: value}) for attr, value in (("tile_width", 16), ("patch", (4, 4)),
                                                  ("ocg_size", 2), ("unicast_channels", 0), ("omit_ocg", True))),
    *(("unary", {attr: value}) for attr, value in (("tile_height", 0), ("tile_width", -1),
                                                 ("patch", (4, 4, 4)), ("patch", (15, 15)))),
    ("conv", {"l2_result_base": 1}), ("conv", {"l2_result_base": isa.dma.L2_CAPACITY - 16}),
    ("conv", {"ocg_size": 2}),
))
def test_a18_schedule_attributes_are_validated_or_refused(name, attrs):
    constructors = {
        "unary": tir.Unary, "binary": lambda **kw: tir.Binary(kind=expr.BinaryKind.ADD, **kw),
        "pool": lambda **kw: tir.Pool(kernel_size=(1, 1), stride=(1, 1), kind=expr.PoolKind.MAX, **kw),
        "reduce": lambda **kw: tir.Reduce(axes=(2, 3), keepdim=True, kind=expr.ReduceKind.SUM, **kw),
        "conv": tir.Conv, "bypass": tir.Bypass,
    }
    with pytest.raises(ValueError, match="|".join((*attrs, "L2"))):
        image_bytes(constructors[name](**attrs))
