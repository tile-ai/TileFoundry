"""Codegen for ``T.ane.binary``: one planar engine pass over two operands.

``OPS`` gives each binary kind its planar engine op and its second-source select.
Both sources come from L2 buffers that tasks of one engine filled.
"""

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic

OPS = MappingProxyType(
    {
        expr.BinaryKind.ADD: (isa.PEOp.ADD, 2),
        expr.BinaryKind.SUB: (isa.PEOp.ADD, 3),
        expr.BinaryKind.MUL: (isa.PEOp.MUL, 2),
        expr.BinaryKind.MIN: (isa.PEOp.MIN, 2),
        expr.BinaryKind.MAX: (isa.PEOp.MAX, 2),
    }
)


@codegen.register(target.Target, codegen.Role.EMIT, tir.Binary)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.width_code is not None:
        raise ValueError(f"T.ane.binary: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.binary: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.binary: omit_ocg={op.omit_ocg!r} is not decoded yet")
    lhs, rhs, destination = call.args
    dst = context.tensor_desc_of(destination.type, "dst")
    left = context.tensor_desc_of(lhs.type, "lhs")
    right = context.tensor_desc_of(rhs.type, "rhs")
    extent = (dst.channels, dst.height, dst.width)
    for name, operand in (("lhs", left), ("rhs", right)):
        if (operand.channels, operand.height, operand.width) != extent:
            raise ValueError(f"T.ane.binary: {name} shape must equal dst shape")
    if op.kind not in OPS:
        raise ValueError(f"T.ane.binary: {op.kind} has no planar engine encoding")
    engine_op, second_source = OPS[op.kind]
    base, producer = ctx.source(lhs)
    second, other = ctx.source(rhs)
    result_base, stored = ctx.result(destination)
    ctx.emit(
        intrinsic.load_chained(left, producer=producer, second=other, address=base),
        intrinsic.load_second_chained(right, address=second),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_BINARY,
            dst,
            op=engine_op,
            second_source=second_source,
            patch=context.raster_patch(dst, op.patch),
        ),
        intrinsic.store(
            context.written(dst, stored), result_base=result_base, address=ctx.address_of(stored),
            fresh=ctx.fresh, producer=producer, second=other, to_dram=stored is not None,
        ),
    )


__all__: list[str] = []
