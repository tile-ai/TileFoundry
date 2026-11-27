"""Codegen for ``T.ane.binary``: one planar engine pass over two operands.

``OPS`` gives each binary kind its planar engine op and its second-source select.
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
SOURCE_ROUTING = 3
RESULT_SLOTS = 2


@codegen.register(target.Target, codegen.Role.EMIT, tir.Binary)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.tile_width is not None:
        raise ValueError(f"T.ane.binary: tile_width={op.tile_width!r} is not decoded yet")
    if op.width_code is not None:
        raise ValueError(f"T.ane.binary: width_code={op.width_code!r} is not decoded yet")
    if op.ocg_size is not None:
        raise ValueError(f"T.ane.binary: ocg_size={op.ocg_size!r} is not decoded yet")
    if op.omit_ocg is not False:
        raise ValueError(f"T.ane.binary: omit_ocg={op.omit_ocg!r} is not decoded yet")
    if type(op.chained_source) is not bool:
        raise ValueError("T.ane.binary: chained_source must be a boolean")
    if op.chained_source:
        if ctx.fresh:
            raise ValueError("T.ane.binary: chained_source requires a preceding pass")
        raise ValueError("T.ane.binary: chained_source continuation mapping is not decoded yet")
    destination, lhs, rhs = call.args
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
    rows = context.tile_rows(op, dst)
    ctx.emit(
        intrinsic.load(left, rows=rows, address=ctx.address_of(lhs), fresh=ctx.fresh),
        intrinsic.load_second(right, rows=rows, address=ctx.address_of(rhs), fresh=ctx.fresh),
        intrinsic.compute_pe(
            isa.TaskType.PE_EW_BINARY,
            dst,
            op=engine_op,
            second_source=second_source,
            source_routing=SOURCE_ROUTING,
            tile_height=rows,
            patch=context.raster_patch(op, rows, dst),
        ),
        intrinsic.store(
            dst,
            staged=RESULT_SLOTS,
            rows=rows,
            address=ctx.address_of(destination),
            fresh=ctx.fresh,
            separated=True,
        ),
    )


__all__: list[str] = []
