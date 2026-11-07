"""Native MPP (M,N,K)=(16,32,16), non-transposed SIMD-group contracts.

Pure layouts follow MLX a21f892a6e8e83afb117cef1d1ceae3e82f48c32,
steel/gemm/nax.h:401-454. All descriptors use relaxed_precision=true;
F32 input multiplication may truncate mantissas. No native 16x16x16 claim.
Availability requires a separate target capability, not MetalArchitecture inference.
"""
from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect as E
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.nn.matmul import MatMul as HirMatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.pattern import (
    LayoutPattern,
    OrPattern,
    ShardLayoutPattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern.predicates import variable
from tilefoundry.ir.types import DType, Split, StorageKind
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_access,
    iterating,
    matmul_relations,
    register_access_relation,
)
from tilefoundry.visitor_registry.candidates import HirMatch

from .common import (
    MATRIX_EXECUTION,
    MATRIX_MESH,
    G,
    attribute,
    cpu_args,
    dtype_attribute,
    infer,
    write_result,
)
from .matrix import product_access, product_eval

FLOATS = (DType.f16, DType.bf16, DType.f32)


def fragment(dtype, columns):
    offset = int(columns == 32)
    col_shape = (2, 2, 4) if columns == 16 else (2, 2, 2, 4)
    col_strides = (0, 0, 1) if columns == 16 else (8, 0, 0, 1)
    return TensorPattern(shape=(G, 16, columns), dtype=dtype, storage=StorageKind.RMEM,
        layout=ShardLayoutPattern(
            LayoutPattern((G, (2, 2, 4), col_shape), (0, (4, 0, 0), col_strides)),
            (Split(0), Split(5 + offset), Split(3), Split(4 + offset), Split(2)),
            MATRIX_MESH))

def transfer_fragment():
    return SwitchPattern("columns", {
        n: fragment(WildcardPattern("dtype"), n) for n in (16, 32)})


def memory():
    row = WildcardPattern("row_stride", predicates=(
        variable("row_stride") >= variable("columns"),))
    batch = WildcardPattern("batch_stride", predicates=(
        variable("batch_stride") >= 16 * variable("row_stride"),))
    return TensorPattern(shape=(G, 16, WildcardPattern("columns")),
        dtype=WildcardPattern("dtype"),
        storage=OrPattern(StorageKind.GMEM, StorageKind.SMEM),
        layout=LayoutPattern((G, 16, WildcardPattern("columns")), (batch, row, 1)))


@register_op(dialect="T", category="nn", namespace="metal", name="matmul_m16n32k16")
class MatMul16x32x16(Op):
    execution_mesh = MATRIX_EXECUTION
    relaxed_precision = attribute(bool, True, True)
    a_dtype = attribute(DType, WildcardPattern("a_dtype", predicates=(OrPattern(*FLOATS),)), DType.f32)
    b_dtype = attribute(DType, WildcardPattern("b_dtype", predicates=(
        OrPattern(*FLOATS),
        (variable("a_dtype") == variable("b_dtype")) | (variable("a_dtype") == DType.f32),
    )), DType.f32)

    hir_matches = (HirMatch(HirMatMul, when={"a_layout":"MK", "b_layout":"KN"},
        operands={"dst":"result", "lhs":"lhs", "rhs":"rhs"}),)
    """F32 destination = A16x16 @ B16x32, with relaxed input precision."""
    capability = OpCapability("metal_matmul_m16n32k16")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=fragment(DType.f32, 32))
    lhs = ParamDef(kind="input", effect=E.READ, pattern=fragment(WildcardPattern("a_dtype"), 16))
    rhs = ParamDef(kind="input", effect=E.READ, pattern=fragment(WildcardPattern("b_dtype"), 32))


@register_op(dialect="T", category="nn", namespace="metal", name="mma_m16n32k16")
class Mma16x32x16(Op):
    execution_mesh = MATRIX_EXECUTION
    relaxed_precision = attribute(bool, True, True)
    a_dtype = attribute(DType, WildcardPattern("a_dtype", predicates=(OrPattern(*FLOATS),)), DType.f32)
    b_dtype = attribute(DType, WildcardPattern("b_dtype", predicates=(
        OrPattern(*FLOATS),
        (variable("a_dtype") == variable("b_dtype")) | (variable("a_dtype") == DType.f32),
    )), DType.f32)

    capability = OpCapability("metal_mma_m16n32k16")
    acc = ParamDef(kind="input", effect=E.READ | E.WRITE, pattern=fragment(DType.f32, 32))
    lhs = ParamDef(kind="input", effect=E.READ, pattern=fragment(WildcardPattern("a_dtype"), 16))
    rhs = ParamDef(kind="input", effect=E.READ, pattern=fragment(WildcardPattern("b_dtype"), 32))


@register_op(dialect="T", category="memory", namespace="metal", name="ldmatrix16")
class LdMatrix16(Op):
    execution_mesh = MATRIX_EXECUTION
    columns = attribute(int, OrPattern(16, 32), 16, name="columns")
    dtype = dtype_attribute(FLOATS, DType.f32)

    hir_matches = tuple(HirMatch(Reshard, attributes={"columns":n}, operands={"dst":"result", "src":"x"}) for n in (16,32))
    capability = OpCapability("metal_ldmatrix16")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=transfer_fragment())
    src = ParamDef(kind="input", effect=E.READ, pattern=memory())


@register_op(dialect="T", category="memory", namespace="metal", name="stmatrix16")
class StMatrix16(Op):
    execution_mesh = MATRIX_EXECUTION
    columns = attribute(int, OrPattern(16, 32), 16, name="columns")
    dtype = dtype_attribute(FLOATS, DType.f32)

    hir_matches = tuple(HirMatch(Reshard, attributes={"columns":n}, operands={"dst":"result", "src":"x"}) for n in (16,32))
    capability = OpCapability("metal_stmatrix16")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=memory())
    src = ParamDef(kind="input", effect=E.READ, pattern=transfer_fragment())


register_typeinfer(LdMatrix16)(infer)
register_typeinfer(StMatrix16)(infer)
register_typeinfer(MatMul16x32x16)(infer)
register_typeinfer(Mma16x32x16)(infer)
register_access_relation(MatMul16x32x16)(product_access)
register_schedule_eval(MatMul16x32x16)(product_eval)


@register_access_relation(LdMatrix16)
@register_access_relation(StMatrix16)
def transfer_access(call, ctx):
    shape = ctx.type_of(call.args[0]).shape
    same = identity_access(3)
    return iterating(shape, (same, same, same))


@register_schedule_eval(LdMatrix16)
@register_schedule_eval(StMatrix16)
def transfer_eval(ctx):
    _, src = cpu_args(ctx)
    return write_result(ctx, src.clone())


@register_access_relation(Mma16x32x16)
def mma_access(call, ctx):
    lhs, rhs, dst = matmul_relations(ctx.type_of(call.args[1]).shape,
        ctx.type_of(call.args[2]).shape, (-2, -1, -1, -2))
    return dst, lhs, rhs, dst


@register_schedule_eval(Mma16x32x16)
def mma_eval(ctx):
    acc, lhs, rhs = cpu_args(ctx)
    return write_result(ctx, lhs.to(torch.float64) @ rhs.to(torch.float64) + acc.to(torch.float64))
