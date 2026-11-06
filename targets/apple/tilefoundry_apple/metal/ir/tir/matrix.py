"""8x8 matrix primitives with fully specified Apple GPU8 lane layouts."""
import isl
from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation, identity_access, iterating, matmul_relations, register_access_relation,
)
from .common import infer, cpu_args, write_result
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect as E, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.nn.matmul import MatMul as HirMatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import DType
from tilefoundry.visitor_registry.candidates import HirMatch
from .common import MATRIX_EXECUTION, attribute, dtype_attribute, matrix, memory_matrix
from tilefoundry.ir.pattern import OrPattern, WildcardPattern

@register_op(dialect="T", category="memory", namespace="metal", name="ldmatrix")
class LdMatrix(Op):
    execution_mesh = MATRIX_EXECUTION
    capability = OpCapability("metal_ldmatrix")
    hir_matches = (HirMatch(Reshard, attributes={"transpose": False}, operands={"dst": "result", "src": "x"}),)
    transpose = attribute(bool, OrPattern(False,True), False)
    dtype = dtype_attribute((DType.f16, DType.f32), DType.f32)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=matrix())
    src = ParamDef(kind="input", effect=E.READ, pattern=memory_matrix())

@register_op(dialect="T", category="memory", namespace="metal", name="stmatrix")
class StMatrix(Op):
    execution_mesh = MATRIX_EXECUTION
    capability = OpCapability("metal_stmatrix")
    hir_matches = (HirMatch(Reshard, attributes={"transpose": False}, operands={"dst": "result", "src": "x"}),)
    transpose = attribute(bool, OrPattern(False,True), False)
    dtype = dtype_attribute((DType.f16, DType.f32), DType.f32)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=memory_matrix())
    src = ParamDef(kind="input", effect=E.READ, pattern=matrix())

@register_op(dialect="T", category="nn", namespace="metal", name="matmul")
class MatMul(Op):
    execution_mesh = MATRIX_EXECUTION
    capability = OpCapability("metal_matmul")
    hir_matches = (HirMatch(HirMatMul, when={"a_layout": "MK", "b_layout": "KN"},
        operands={"dst": "result", "lhs": "lhs", "rhs": "rhs"}),)
    dtype = dtype_attribute((DType.f16, DType.f32), DType.f32)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=matrix())
    lhs = ParamDef(kind="input", effect=E.READ, pattern=matrix())
    rhs = ParamDef(kind="input", effect=E.READ, pattern=matrix())

@register_op(dialect="T", category="nn", namespace="metal", name="mma_m8n8k8")
class Mma8x8x8(Op):
    execution_mesh = MATRIX_EXECUTION
    capability = OpCapability("metal_mma_m8n8k8")
    lhs_dtype = attribute(DType, WildcardPattern("dtype", predicates=(OrPattern(DType.f16, DType.f32),)), DType.f32)
    rhs_dtype = attribute(DType, WildcardPattern("dtype"), DType.f32)
    acc_dtype = attribute(DType, WildcardPattern("dtype"), DType.f32)
    dst_dtype = attribute(DType, WildcardPattern("dtype"), DType.f32)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=matrix())
    lhs = ParamDef(kind="input", effect=E.READ, pattern=matrix())
    rhs = ParamDef(kind="input", effect=E.READ, pattern=matrix())
    acc = ParamDef(kind="input", effect=E.READ, pattern=matrix())


register_typeinfer(LdMatrix)(infer)
register_typeinfer(StMatrix)(infer)
register_typeinfer(MatMul)(infer)
register_typeinfer(Mma8x8x8)(infer)


@register_access_relation(LdMatrix)
def load_access(call, ctx):
    shape = ctx.type_of(call.args[0]).shape
    dst = identity_access(3)
    src = AccessRelation(isl.map("{ [g,m,n] -> [g,n,m] }")) if call.target.transpose else dst
    return iterating(shape, (dst, src, dst))


@register_access_relation(StMatrix)
def store_access(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    src = identity_access(3)
    dst = AccessRelation(isl.map("{ [g,m,n] -> [g,n,m] }")) if call.target.transpose else src
    return iterating(shape, (dst, src, dst))


@register_access_relation(MatMul)
def product_access(call, ctx):
    lhs, rhs, dst = matmul_relations(ctx.type_of(call.args[1]).shape,
        ctx.type_of(call.args[2]).shape, (-2, -1, -1, -2))
    return dst, lhs, rhs, dst


@register_access_relation(Mma8x8x8)
def mma_access(call, ctx):
    lhs, rhs, dst = matmul_relations(ctx.type_of(call.args[1]).shape,
        ctx.type_of(call.args[2]).shape, (-2, -1, -1, -2))
    return dst, lhs, rhs, dst, dst


@register_schedule_eval(LdMatrix)
@register_schedule_eval(StMatrix)
def transfer_eval(ctx):
    _, src = cpu_args(ctx)
    result = src.transpose(-2, -1) if ctx.op.transpose else src
    return write_result(ctx, result.clone())


@register_schedule_eval(MatMul)
def product_eval(ctx):
    _, lhs, rhs = cpu_args(ctx)
    return write_result(ctx, lhs.to(torch.float64) @ rhs.to(torch.float64))


@register_schedule_eval(Mma8x8x8)
def mma_eval(ctx):
    _, lhs, rhs, acc = cpu_args(ctx)
    return write_result(ctx, lhs.to(torch.float64) @ rhs.to(torch.float64) + acc.to(torch.float64))
