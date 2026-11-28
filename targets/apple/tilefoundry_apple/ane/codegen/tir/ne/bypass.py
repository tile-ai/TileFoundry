"""Codegen for ``T.ane.bypass``: one neural engine pass-through, optionally permuting."""

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context
from tilefoundry_apple.ane.ir import tir
from tilefoundry_apple.ane.isa import intrinsic

NONLINEAR = MappingProxyType(
    {
        tir.Nonlinear.NONE: isa.NENonlinear.NONE,
        tir.Nonlinear.RELU: isa.NENonlinear.RELU,
    }
)
IDENTITY = (0, 1, 2)


@codegen.register(target.Target, codegen.Role.EMIT, tir.Bypass)
def emit(call: expr.Call, ctx: context.Context) -> None:
    op = call.target
    if op.tile_width is not None:
        raise ValueError(f"T.ane.bypass: tile_width={op.tile_width!r} is not decoded yet")
    if op.patch is not None:
        raise ValueError(f"T.ane.bypass: patch={op.patch!r} is not decoded yet")
    if op.nonlinear not in NONLINEAR:
        raise ValueError(f"T.ane.bypass: nonlinear={op.nonlinear.name} is not decoded yet")
    destination, source = call.args
    if source.type.storage is expr.StorageKind.SMEM:
        raise ValueError("T.ane.bypass: L2 source is not decoded yet")
    perm = tuple(op.perm)
    if sorted(perm) != list(IDENTITY):
        raise ValueError(f"T.ane.bypass: perm {perm} must be a permutation of {IDENTITY}")
    if perm not in (IDENTITY, (2, 1, 0)):
        raise ValueError(f"T.ane.bypass: perm {perm} is not decoded yet")
    permuted = perm != IDENTITY
    src = context.tensor_desc_of(source.type, "src")
    dst = context.tensor_desc_of(destination.type, "dst", strided=permuted)
    extent = (dst.channels, dst.height, dst.width)
    expected = tuple(src.shape[1 + axis] for axis in perm)
    if extent != expected:
        raise ValueError("T.ane.bypass: dst shape must equal src shape")
    rows = context.tile_rows(op, dst)
    declared = src if permuted else None
    ocg_size = isa.rules.channel_groups(src.channels)
    if op.ocg_size is not None and (type(op.ocg_size) is not int or op.ocg_size != ocg_size):
        raise ValueError(f"T.ane.bypass: ocg_size {op.ocg_size!r} contradicts channels; expected {ocg_size}")
    if type(op.omit_ocg) is not bool:
        raise ValueError("T.ane.bypass: omit_ocg must be a boolean")
    if op.omit_ocg and (op.ocg_size is not None or ocg_size != 0):
        raise ValueError("T.ane.bypass: omit_ocg for this channel grouping is not decoded yet")
    unicast = isa.rules.unicast_channels(src.channels) if permuted else 1
    if op.unicast_channels is not None:
        if type(op.unicast_channels) is not int or not 1 <= op.unicast_channels <= 65535:
            raise ValueError("T.ane.bypass: unicast_channels must be within [1, 65535]")
        if op.unicast_channels != unicast:
            raise ValueError(f"T.ane.bypass: unicast_channels {op.unicast_channels} is not decoded yet")
    compute = intrinsic.compute_ne(
        isa.NEOp.BYPASS,
        dst,
        nonlinear=NONLINEAR[op.nonlinear],
        kernel_mode=1,
        tile_height=rows,
        width_code=op.width_code,
        declared=declared,
        transpose=permuted,
        unicast_channels=unicast,
        ocg_size=None if op.omit_ocg else ocg_size,
        address=ctx.address_of(destination),
    )
    ctx.emit(
        intrinsic.load(src, isa.Engine.NEURAL, address=ctx.address_of(source), fresh=ctx.fresh,
                       l2_base=ctx.staging_address(1)),
        compute,
        intrinsic.store(
            dst,
            staging=src,
            rows=rows,
            address=ctx.address_of(destination),
            fresh=ctx.fresh,
            declared=declared,
            result_base=ctx.staging_address(0),
            to_dram=destination.type.storage is expr.StorageKind.GMEM,
        ),
    )


__all__: list[str] = []
