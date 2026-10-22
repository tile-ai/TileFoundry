"""SM90 tensor-map reduction from shared memory into an explicit destination."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import SameConstraint, SameModesConstraint, utils
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import (
    TmaBoxPattern,
    TmaGlobalPattern,
    _warp_scope,
)
from tilefoundry.ir.types import Mesh, UnitType
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.ir.types.utils import tensor_bytes
from tilefoundry.visitor_registry import register_cost_evaluator, register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)
from tilefoundry.visitor_registry.contexts import Cost, TrafficBytes


@register_op(dialect="T", category="async", name="copy_reduce_async_tensor")
class CopyReduceAsyncTensor(Op):
    """Add a logical partial sum to the global destination; hardware adds in L2."""

    capability = OpCapability("cp.async.bulk.tensor")
    resource = "tma_engine"
    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=utils.operand_tile(
            0, S.SMEM, TmaBoxPattern(utils.dtype_place(0), allow_partial=True)
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=utils.operand_tile(1, S.GMEM, TmaGlobalPattern(utils.dtype_place(1))),
    )
    kind = ParamDef(kind="attribute", pattern="add", default="add")
    execution_mesh = ParamDef(
        kind="attribute", annotation=Mesh, pattern=_warp_scope(), optional=True, default=None
    )
    between = (
        SameConstraint("shape", "src", "dst"),
        SameConstraint("dtype", "src", "dst"),
        SameModesConstraint("src", "dst", allow_partial=True),
    )


@register_typeinfer(CopyReduceAsyncTensor)
def _(call, ctx):
    return UnitType()


register_access_relation(CopyReduceAsyncTensor)(identity_relations)


@register_schedule_eval(CopyReduceAsyncTensor)
def _(ctx):
    return TensorValue(data=ctx.args[1].data + ctx.args[0].data, type=ctx.result_type)


@register_cost_evaluator(CopyReduceAsyncTensor)
def _(call, ctx):
    src, dst = (tensor_bytes(ctx.local_type_of(arg)) for arg in call.args)
    return Cost({}, (TrafficBytes(read=src), TrafficBytes(write=dst), TrafficBytes()))


__all__ = ["CopyReduceAsyncTensor"]
