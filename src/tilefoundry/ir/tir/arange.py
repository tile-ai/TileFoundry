"""Effect-form generation of one-dimensional register coordinates."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import core, utils
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect="T", category="arith", name="arange")
class Arange(expr.Op):
    """Write start + i * step into each coordinate i of dst."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=replace(
        pattern.tensor_in(expr.StorageKind.RMEM), shape=(None,), dtype=pattern.Or(expr.DType.i32, expr.DType.i64),
    ))
    start = expr.ParamDef(kind="attribute", annotation=int, default=0)
    step = expr.ParamDef(kind="attribute", annotation=int, default=1)


@typeinfer.register(Arange)
def _(call, ctx) -> types.UnitType:
    return types.UnitType()


access.register_relation(Arange)(access.identity_relations)


@evaluator.register(Arange)
def eval_arange(ctx):
    data = utils.torch.arange(
        ctx.op.start, ctx.op.start + ctx.args[0].data.shape[0] * ctx.op.step, ctx.op.step,
        dtype=evaluator.to_torch_dtype(ctx.result_type.dtype), device=ctx.device,
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, data))


__all__ = ["Arange"]
