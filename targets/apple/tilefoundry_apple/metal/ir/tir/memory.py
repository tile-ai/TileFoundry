"""MLX special device atomics and Metal4 coherent(system) memory contracts.

Source inventory: MLX a21f892a6e8e83afb117cef1d1ceae3e82f48c32.
All request indices are runtime checked; no opaque atomic tensor type is used.
"""
from enum import Enum

import isl

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect as E
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op, register_type
from tilefoundry.ir.pattern import (
    LayoutPattern,
    MeshPattern,
    OrPattern,
    RangePattern,
    ShardLayoutPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern.predicates import variable
from tilefoundry.ir.pattern.utils import thread_execution_mesh
from tilefoundry.ir.types import DType, Split, StorageKind
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)

from .common import attribute, cpu_args, dtype_attribute, infer


@register_type(dialect="T", namespace="metal")
class AtomicKind(Enum):
    ADD = "add"
    AND = "and"
    OR = "or"
    MIN = "min"
    MAX = "max"


@register_type(dialect="T", namespace="metal")
class AtomicOrder(Enum):
    RELAXED = "relaxed"
    SEQ_CST = "seq_cst"


R = WildcardPattern("requests", predicates=(RangePattern(lo=1, hi=1024),))
DT = WildcardPattern("dtype")


def request(dtype=DT):
    return TensorPattern(shape=(R,), dtype=dtype, storage=StorageKind.RMEM,
        layout=ShardLayoutPattern(LayoutPattern((R,), (0,)), (Split(0),),
            MeshPattern(("thread",), LayoutPattern(((R,),), ((1,),)))))


def backing(dtype=DT):
    size = WildcardPattern("capacity", predicates=(RangePattern(lo=1),))
    return TensorPattern(shape=(size,), dtype=dtype, storage=StorageKind.GMEM,
        layout=LayoutPattern((size,), (1,)))


def index():
    return request(OrPattern(DType.i32, DType.u32, DType.i64))


@register_op(dialect="T", category="memory", namespace="metal", name="atom_load")
class AtomicLoad(Op):
    execution_mesh = thread_execution_mesh()
    dtype = dtype_attribute((DType.i32, DType.u32, DType.f32), DType.u32)
    order = attribute(AtomicOrder, AtomicOrder.RELAXED, AtomicOrder.RELAXED)

    capability = OpCapability("metal_atom_load")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=request())
    object = ParamDef(kind="input", effect=E.READ, pattern=backing())
    index = ParamDef(kind="input", effect=E.READ, pattern=index())


@register_op(dialect="T", category="memory", namespace="metal", name="atom_store")
class AtomicStore(Op):
    execution_mesh = thread_execution_mesh()
    dtype = dtype_attribute((DType.i32, DType.u32, DType.f32), DType.u32)
    order = attribute(AtomicOrder, AtomicOrder.RELAXED, AtomicOrder.RELAXED)

    capability = OpCapability("metal_atom_store")
    object = ParamDef(kind="input", effect=E.READ | E.WRITE, pattern=backing())
    value = ParamDef(kind="input", effect=E.READ, pattern=request())
    index = ParamDef(kind="input", effect=E.READ, pattern=index())


@register_op(dialect="T", category="memory", namespace="metal", name="atom")
class AtomicRmw(Op):
    execution_mesh = thread_execution_mesh()
    dtype = dtype_attribute((DType.i32, DType.u32, DType.f32), DType.u32)
    order = attribute(AtomicOrder, AtomicOrder.RELAXED, AtomicOrder.RELAXED)

    capability = OpCapability("metal_atom")
    kind = attribute(AtomicKind, WildcardPattern("kind", predicates=(
        OrPattern(*AtomicKind),
        (variable("dtype") != DType.f32) | (variable("kind") == AtomicKind.ADD),
    )), AtomicKind.ADD)
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=request())
    object = ParamDef(kind="input", effect=E.READ | E.WRITE, pattern=backing())
    value = ParamDef(kind="input", effect=E.READ, pattern=request())
    index = ParamDef(kind="input", effect=E.READ, pattern=index())


@register_op(dialect="T", category="memory", namespace="metal", name="atom_cas")
class AtomicCas(Op):
    execution_mesh = thread_execution_mesh()
    dtype = dtype_attribute((DType.i32, DType.u32, DType.f32), DType.u32)
    order = attribute(AtomicOrder, AtomicOrder.RELAXED, AtomicOrder.RELAXED)

    capability = OpCapability("metal_atom_cas")
    success = ParamDef(kind="input", effect=E.WRITE, pattern=request(DType.bool))
    object = ParamDef(kind="input", effect=E.READ | E.WRITE, pattern=backing())
    expected = ParamDef(kind="input", effect=E.READ | E.WRITE, pattern=request())
    desired = ParamDef(kind="input", effect=E.READ, pattern=request())
    index = ParamDef(kind="input", effect=E.READ, pattern=index())


@register_op(dialect="T", category="memory", namespace="metal", name="ldcoherent")
class CoherentLoad(Op):
    execution_mesh = thread_execution_mesh()
    dtype = dtype_attribute((DType.u32,), DType.u32)

    capability = OpCapability("metal_system_coherence")
    dst = ParamDef(kind="input", effect=E.WRITE, pattern=request())
    object = ParamDef(kind="input", effect=E.READ, pattern=backing())
    index = ParamDef(kind="input", effect=E.READ, pattern=index())


@register_op(dialect="T", category="memory", namespace="metal", name="stcoherent")
class CoherentStore(Op):
    execution_mesh = thread_execution_mesh()
    dtype = dtype_attribute((DType.u32,), DType.u32)

    capability = OpCapability("metal_system_coherence")
    object = ParamDef(kind="input", effect=E.READ | E.WRITE, pattern=backing())
    value = ParamDef(kind="input", effect=E.READ, pattern=request())
    index = ParamDef(kind="input", effect=E.READ, pattern=index())


def operand_access(call, ctx, object_pos):
    types = tuple(ctx.type_of(a) for a in call.args)
    capacity = int(types[object_pos].shape[0])
    requests = int(types[-1].shape[0])
    dynamic = AccessRelation(isl.map(f"{{ [r] -> [j] : 0 <= j < {capacity} }}"), may_access=True)
    relations = tuple(dynamic if i == object_pos else identity_access(1) for i in range(len(types)))
    return iterating((requests,), (*relations, relations[0]))

def load_access(call, ctx):
    return operand_access(call, ctx, 1)

def store_access(call, ctx):
    return operand_access(call, ctx, 0)

def requests(ctx, object_pos):
    a = cpu_args(ctx)
    obj = a[object_pos]
    indices = [int(x) for x in a[-1].tolist()]
    if any(i < 0 or i >= obj.numel() for i in indices):
        raise EvalError("Metal memory request index is outside backing capacity")
    return a, obj, indices

def _wrap(value, dtype):
    value = int(value) & 0xffffffff
    return value - 0x100000000 if dtype == DType.i32 and value >= 0x80000000 else value



@register_schedule_eval(AtomicLoad)
@register_schedule_eval(CoherentLoad)
def eval_atomicload(ctx):
    a, obj, indices = requests(ctx, 1)
    for r, i in enumerate(indices):
        old = obj[i].item()
        a[0][r] = old
    return ctx.args[0]

@register_schedule_eval(AtomicStore)
@register_schedule_eval(CoherentStore)
def eval_atomicstore(ctx):
    a, obj, indices = requests(ctx, 0)
    for r, i in enumerate(indices):
        obj[i] = a[1][r]
    return ctx.args[0]

@register_schedule_eval(AtomicCas)
def eval_atomiccas(ctx):
    op = ctx.op
    a, obj, indices = requests(ctx, 1)
    for r, i in enumerate(indices):
        old = obj[i].item()
        expected = a[2][r].item()
        equal = (obj.view(torch.int32)[i].item() == a[2].view(torch.int32)[r].item()
                 if op.dtype == DType.f32 else old == expected)
        a[0][r] = equal
        if equal:
            obj[i] = a[3][r]
        else:
            a[2][r] = obj[i]
    return ctx.args[0]

@register_schedule_eval(AtomicRmw)
def eval_atomicrmw(ctx):
    op = ctx.op
    a, obj, indices = requests(ctx, 1)
    for r, i in enumerate(indices):
        old = obj[i].item()
        v = a[2][r].item()
        if op.kind is AtomicKind.ADD:
            new = old + v
        elif op.kind is AtomicKind.AND:
            new = old & v
        elif op.kind is AtomicKind.OR:
            new = old | v
        elif op.kind is AtomicKind.MIN:
            new = min(old, v)
        elif op.kind is AtomicKind.MAX:
            new = max(old, v)
        else:
            raise EvalError("unsupported Metal atomic operation")
        a[0][r] = old
        obj[i] = new if op.dtype == DType.f32 else _wrap(new, op.dtype)
    return ctx.args[0]

register_typeinfer(AtomicLoad)(infer)
register_access_relation(AtomicLoad)(load_access)

register_typeinfer(AtomicStore)(infer)
register_access_relation(AtomicStore)(store_access)

register_typeinfer(AtomicRmw)(infer)
register_access_relation(AtomicRmw)(load_access)

register_typeinfer(AtomicCas)(infer)
register_access_relation(AtomicCas)(load_access)

register_typeinfer(CoherentLoad)(infer)
register_access_relation(CoherentLoad)(load_access)

register_typeinfer(CoherentStore)(infer)
register_access_relation(CoherentStore)(store_access)
