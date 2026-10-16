"""Effect-form generation of one-dimensional register coordinates."""

from __future__ import annotations

from dataclasses import replace

import torch

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import OrPattern, utils
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op(dialect="T", category="arith", name="arange")
class Arange(Op):
    """Write start + i * step into each coordinate i of dst."""

    capability = OpCapability(None)
    execution_mesh = utils.thread_execution_mesh()
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=replace(
        utils.tensor_in(StorageKind.RMEM), shape=(None,), dtype=OrPattern(DType.i32, DType.i64),
    ))
    start = ParamDef(kind="attribute", annotation=int, default=0)
    step = ParamDef(kind="attribute", annotation=int, default=1)


@register_typeinfer(Arange)
def _(call, ctx) -> UnitType:
    return UnitType()


register_access_relation(Arange)(identity_relations)


@register_schedule_eval(Arange)
def _eval_arange(ctx):
    data = torch.arange(
        ctx.op.start, ctx.op.start + ctx.args[0].data.shape[0] * ctx.op.step, ctx.op.step,
        dtype=to_torch_dtype(ctx.result_type.dtype), device=ctx.device,
    )
    return TensorValue(data=data, type=ctx.result_type)
