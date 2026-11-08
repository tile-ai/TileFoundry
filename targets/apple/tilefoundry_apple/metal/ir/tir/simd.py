"""MLX-used native collectives and typed runtime lane selectors."""
from enum import Enum

import isl

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Constant, Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect as E
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op, register_type
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.pattern import (
    LayoutPattern,
    MeshPattern,
    OrPattern,
    RangePattern,
    ShardLayoutPattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.types import DType, Partial, Split
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.candidates import HirMatch

from .common import (
    INT,
    NUM,
    SIMD_MESH,
    attribute,
    cpu_args,
    dtype_attribute,
    infer,
    lanes,
    selector,
    vectors,
    write_result,
)


@register_type(dialect="T", namespace="metal")
class ReductionKind(Enum):
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AND = "and"
    OR = "or"
    XOR = "xor"
    ALL = "all"
    ANY = "any"

@register_type(dialect="T", namespace="metal")
class ShuffleKind(Enum):
    INDEX = "index"
    BROADCAST = "broadcast"
    XOR = "xor"
    UP = "up"
    DOWN = "down"
    ROTATE_UP = "rotate_up"
    ROTATE_DOWN = "rotate_down"

SLOTS = attribute(int, OrPattern(1, 2, 4), 1, name="slots")
WIDTH = WildcardPattern("width", predicates=(OrPattern(4, 32),))
GROUPS = WildcardPattern("groups", predicates=(RangePattern(lo=1, hi=256),))
COLLECTIVE_MESH = MeshPattern(("thread",), LayoutPattern(((GROUPS, WIDTH),), ((WIDTH, 1),),
    predicates=(GROUPS * WIDTH <= 1024,)))
QUAD = (WIDTH == 32) | (WildcardPattern("kind") == ReductionKind.SUM)
_REDUCTION_TYPES = {k: (DType.bool if k in (ReductionKind.ALL, ReductionKind.ANY)
    else OrPattern(*INT) if k in (ReductionKind.AND, ReductionKind.OR, ReductionKind.XOR)
    else OrPattern(*NUM)) for k in ReductionKind}

def _reduce_source(slots):
    native = dict(vectors(width=WIDTH, groups=GROUPS, predicates=(QUAD,)).branches)[slots]
    if slots != 1:
        return native
    partial = SwitchPattern("kind", {
        kind: ShardLayoutPattern(LayoutPattern((GROUPS, 1), (0, 0)),
            (Split(0), Partial(kind.value)), COLLECTIVE_MESH)
        for kind in (ReductionKind.SUM, ReductionKind.MIN, ReductionKind.MAX)})
    return TensorPattern(shape=(GROUPS, WildcardPattern("source_width", predicates=(OrPattern(1, WIDTH),))),
        dtype=native.dtype, storage=native.storage,
        layout=SwitchPattern("source_width", {1: partial, 4: native.layout, 32: native.layout}),
        predicates=(QUAD,))


@register_op(dialect="T", category="arith", namespace="metal", name="reduce")
class Reduce(Op):
    capability = OpCapability("metal_reduce")
    execution_mesh = COLLECTIVE_MESH
    hir_matches = tuple(HirMatch(Reshard, attributes={"slots": 1, "kind": kind},
        operands={"dst": "result", "src": "x"})
        for kind in (ReductionKind.SUM, ReductionKind.MIN, ReductionKind.MAX))
    kind = attribute(ReductionKind, OrPattern(*ReductionKind), ReductionKind.SUM, name="kind")
    dtype = attribute(DType, WildcardPattern("dtype", predicates=(SwitchPattern("kind", _REDUCTION_TYPES),)), DType.f32)
    slots = attribute(int, SwitchPattern("kind", {k: 1 if k in (ReductionKind.ALL,ReductionKind.ANY) else OrPattern(1,2,4) for k in ReductionKind}), 1, name="slots")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=vectors(broadcast=True, width=WIDTH, groups=GROUPS))
    src = ParamDef(kind="input", effect=E.READ, pattern=SwitchPattern("slots", {
        slots: _reduce_source(slots) for slots in (1, 2, 4)}))

@register_op(dialect="T", category="arith", namespace="metal", name="shuffle")
class Shuffle(Op):
    execution_mesh = SIMD_MESH
    capability = OpCapability("metal_shuffle")
    kind = attribute(ShuffleKind, OrPattern(*ShuffleKind), ShuffleKind.INDEX, name="kind")
    dtype = dtype_attribute((*NUM, DType.bf16), DType.f32)
    slots = SLOTS
    uniform = attribute(bool, SwitchPattern("kind", {k: OrPattern(True,False) if k is ShuffleKind.INDEX else True for k in ShuffleKind}), True, name="uniform")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=vectors())
    src = ParamDef(kind="input", effect=E.READ, pattern=vectors())
    index = ParamDef(kind="input", effect=E.READ, pattern=selector())

@register_op(dialect="T", category="arith", namespace="metal", name="shuffle_fill")
class ShuffleFill(Op):
    execution_mesh = SIMD_MESH
    capability = OpCapability("metal_shuffle_fill")
    kind = attribute(ShuffleKind, OrPattern(ShuffleKind.UP, ShuffleKind.DOWN), ShuffleKind.UP, name="kind")
    dtype = dtype_attribute((*NUM, DType.bf16), DType.f32)
    slots = SLOTS
    modulo = attribute(int, OrPattern(2, 4, 8, 16, 32), 32)
    uniform = attribute(bool, True, True, name="uniform")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=vectors())
    src = ParamDef(kind="input", effect=E.READ, pattern=vectors())
    fill = ParamDef(kind="input", effect=E.READ, pattern=vectors())
    delta = ParamDef(kind="input", effect=E.READ, pattern=selector())

@register_op(dialect="T", category="arith", namespace="metal", name="ballot")
class Ballot(Op):
    execution_mesh = SIMD_MESH
    capability = OpCapability("metal_ballot")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=lanes(DType.u32, broadcast=True))
    src = ParamDef(kind="input", effect=E.READ, pattern=lanes(DType.bool))

@register_op(dialect="T", category="arith", namespace="metal", name="prefix_sum")
class PrefixSum(Op):
    execution_mesh = SIMD_MESH
    dtype = dtype_attribute(NUM, DType.f32)
    slots = SLOTS
    inclusive = attribute(bool, OrPattern(False, True), False)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=vectors())
    src = ParamDef(kind="input", effect=E.READ, pattern=vectors())
    capability = OpCapability("metal_prefix_sum")

@register_op(dialect="T", category="arith", namespace="metal", name="prefix_product")
class PrefixProduct(Op):
    execution_mesh = SIMD_MESH
    dtype = dtype_attribute(NUM, DType.f32)
    slots = SLOTS
    inclusive = attribute(bool, OrPattern(False, True), False)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=vectors())
    src = ParamDef(kind="input", effect=E.READ, pattern=vectors())
    capability = OpCapability("metal_prefix_product")

K = ReductionKind
S = ShuffleKind

@register_schedule_eval(Reduce)
def eval_reduce(ctx):
    op = ctx.op
    if any(isinstance(attr, Partial) for attr in getattr(ctx.args[1].type.layout, "attrs", ())):
        return write_result(ctx, ctx.args[1].tensor.to_global().tensor)
    a = cpu_args(ctx)
    data=a[1].to(torch.float64) if a[1].is_floating_point() else a[1].to(torch.int64)
    if op.kind is K.SUM:
        result=data.sum(1,keepdim=True)
    elif op.kind is K.PRODUCT:
        result=data.prod(1,keepdim=True)
    elif op.kind is K.MIN:
        result=data.amin(1,keepdim=True)
    elif op.kind is K.MAX:
        result=data.amax(1,keepdim=True)
    elif op.kind is K.ALL:
        result=data.bool().all(1,keepdim=True)
    elif op.kind is K.ANY:
        result=data.bool().any(1,keepdim=True)
    else:
        result=data[:,:1].clone()
        for lane in range(1,data.shape[1]):
            if op.kind is K.AND:
                result &= data[:,lane:lane+1]
            elif op.kind is K.OR:
                result |= data[:,lane:lane+1]
            else:
                result ^= data[:,lane:lane+1]
    return write_result(ctx, result)

@register_schedule_eval(Shuffle)
def eval_shuffle(ctx):
    op = ctx.op
    a = cpu_args(ctx)
    data=a[1]
    groups=data.shape[0]
    amount=a[-1].to(torch.int64)
    if op.uniform:
        amount=amount.expand(groups,32)
    limit=32
    if bool(((amount<0)|(amount>=limit)).any()):
        raise EvalError("shuffle selector outside lane/vector range")
    lanes=torch.arange(32).expand(groups,32)
    if op.kind in (S.INDEX,S.BROADCAST):
        source=amount
    elif op.kind is S.XOR:
        source=lanes^amount
    elif op.kind in (S.UP,S.ROTATE_UP):
        source=lanes-amount
    else:
        source=lanes+amount
    if op.kind in (S.ROTATE_UP,S.ROTATE_DOWN):
        source%=32
    elif op.kind in (S.UP,S.DOWN):
        source=torch.where((source<0)|(source>=32),lanes,source)
    result=data[torch.arange(groups)[:,None],source]
    return write_result(ctx, result)

@register_schedule_eval(ShuffleFill)
def eval_shufflefill(ctx):
    op = ctx.op
    a = cpu_args(ctx)
    data=a[1]
    groups=data.shape[0]
    amount=a[-1].to(torch.int64)
    if op.uniform:
        amount=amount.expand(groups,32)
    limit=op.modulo+1
    if bool(((amount<0)|(amount>=limit)).any()):
        raise EvalError("shuffle selector outside lane/vector range")
    lanes=torch.arange(32).expand(groups,32)
    width=op.modulo
    relative=lanes%width
    if op.kind is S.UP:
        src=relative-amount
        fill_src=src+width
    else:
        src=relative+amount
        fill_src=src-width
    in_source=(src>=0)&(src<width)
    base=lanes-relative
    source=(base+src.clamp(0,width-1)).long()
    fill_index=(base+fill_src.clamp(0,width-1)).long()
    rows=torch.arange(groups)[:,None]
    result=data[rows,source]
    other=a[2][rows,fill_index]
    mask=in_source if data.ndim==2 else in_source[...,None]
    result=torch.where(mask,result,other)
    return write_result(ctx, result)

@register_schedule_eval(Ballot)
def eval_ballot(ctx):
    a = cpu_args(ctx)
    result=(a[1].to(torch.int64)*(1<<torch.arange(32))).sum(1,keepdim=True)
    return write_result(ctx, result)

@register_schedule_eval(PrefixSum)
def eval_prefixsum(ctx):
    op = ctx.op
    a = cpu_args(ctx)
    data=a[1].to(torch.float64) if a[1].is_floating_point() else a[1].to(torch.int64)
    result=data.cumsum(1)
    if not op.inclusive:
        identity=torch.zeros_like(result[:,:1])
        result=torch.cat((identity,result[:,:-1]),1)
    return write_result(ctx, result)

@register_schedule_eval(PrefixProduct)
def eval_prefixproduct(ctx):
    op = ctx.op
    a = cpu_args(ctx)
    data=a[1].to(torch.float64) if a[1].is_floating_point() else a[1].to(torch.int64)
    result=data.cumprod(1)
    if not op.inclusive:
        identity=torch.ones_like(result[:,:1])
        result=torch.cat((identity,result[:,:-1]),1)
    return write_result(ctx, result)

def reduction_access(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    if any(isinstance(attr, Partial) for attr in getattr(ctx.type_of(call.args[1]).layout, "attrs", ())):
        return iterating(shape, (identity_access(len(shape)),) * 3)
    vector = len(shape) == 3
    domain = 'g,l,s' if vector else 'g,l'
    result = 'g,0,s' if vector else 'g,0'
    dst = AccessRelation(isl.map(f'{{ [{domain}] -> [{result}] }}'))
    return iterating(shape, (dst, identity_access(len(shape)), dst))

def prefix_access(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    domain = 'g,l,s' if len(shape) == 3 else 'g,l'
    read = 'g,k,s' if len(shape) == 3 else 'g,k'
    bound = 'k <= l' if call.target.inclusive else 'k < l'
    dst = identity_access(len(shape))
    src = AccessRelation(isl.map(f'{{ [{domain}] -> [{read}] : 0 <= k < 32 and {bound} }}'))
    return iterating(shape, (dst, src, dst))

def shuffle_relations(call, ctx):
    shape = ctx.type_of(call.args[1]).shape
    domain = 'g,l,s' if len(shape) == 3 else 'g,l'
    read = 'g,k,s' if len(shape) == 3 else 'g,k'
    dst = identity_access(len(shape))
    src = AccessRelation(isl.map(f'{{ [{domain}] -> [{read}] : 0 <= k < 32 }}'), may_access=True)
    selector = '' if call.target.uniform else 'g,l'
    index = AccessRelation(isl.map(f'{{ [{domain}] -> [{selector}] }}'))
    return shape, dst, src, index

def shuffle_access(call, ctx):
    shape, dst, src, index = shuffle_relations(call, ctx)
    return iterating(shape, (dst, src, index, dst))

def fill_access(call, ctx):
    shape, dst, src, index = shuffle_relations(call, ctx)
    return iterating(shape, (dst, src, src, index, dst))

def verify_selector(call, ctx, limit):
    value = call.args[-1]
    if isinstance(value, Constant) and isinstance(value.value, int) and not 0 <= value.value < limit:
        ctx.error(call, 'shuffle selector is outside its legal lane/vector range')

@register_verify_stmt(Shuffle)
def verify_shuffle(call, ctx):
    verify_selector(call, ctx, 32)

@register_verify_stmt(ShuffleFill)
def verify_fill(call, ctx):
    verify_selector(call, ctx, call.target.modulo + 1)

register_typeinfer(Reduce)(infer)
register_access_relation(Reduce)(reduction_access)

register_typeinfer(Ballot)(infer)
register_access_relation(Ballot)(reduction_access)

register_typeinfer(Shuffle)(infer)
register_access_relation(Shuffle)(shuffle_access)

register_typeinfer(ShuffleFill)(infer)
register_access_relation(ShuffleFill)(fill_access)

register_typeinfer(PrefixSum)(infer)
register_access_relation(PrefixSum)(prefix_access)

register_typeinfer(PrefixProduct)(infer)
register_access_relation(PrefixProduct)(prefix_access)
