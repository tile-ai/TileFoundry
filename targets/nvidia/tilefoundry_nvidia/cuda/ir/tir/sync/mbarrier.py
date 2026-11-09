"""Effect-form TIR Ops for the CUDA shared-memory barrier object (mbarrier).

[tir](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/tir.md#barrier-object-ops-tirsyncmbarrier_).
"""

from __future__ import annotations

from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, pattern, types

__all__ = [
    "MBARRIER_BYTES",
    "MBarrierInit",
    "MBarrierArriveExpectTx",
    "MBarrierWaitParity",
    "MBarrierInvalidate",
    "mbarrier_words",
]

MBARRIER_DTYPE = expr.DType.i64
MBARRIER_BYTES = MBARRIER_DTYPE.bit_width // 8


def mbarrier_words(count: int) -> types.TensorType:
    """The shared memory type of *count* mbarrier objects, one 64-bit word each."""
    return types.TensorType(
        (count,), MBARRIER_DTYPE, types.Layout((count,), (1,)), expr.StorageKind.SMEM
    )


@expr.register_op(dialect="T", category="sync", name="mbarrier_init")
class MBarrierInit(expr.Op):
    """Arm ``barrier`` so ``arrive_count`` arrivals complete a phase."""

    execution_mesh = pattern.thread_execution_mesh()

    barrier = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.WRITE, pattern=pattern.tensor_in(expr.StorageKind.SMEM)
    )
    arrive_count = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("arrive_count", predicates=(pattern.Wildcard("arrive_count") > 0,)),
    )


@typeinfer.register(MBarrierInit)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


@expr.register_op(dialect="T", category="sync", name="mbarrier_arrive_expect_tx")
class MBarrierArriveExpectTx(expr.Op):
    """Arrive on ``barrier`` and declare ``tx_bytes`` of asynchronous data."""

    capability = expr.OpCapability("mbarrier.expect_tx")
    execution_mesh = pattern.thread_execution_mesh()

    barrier = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.tensor_in(expr.StorageKind.SMEM),
    )
    tx_bytes = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("tx_bytes", predicates=(pattern.Wildcard("tx_bytes") > 0,)),
    )


@typeinfer.register(MBarrierArriveExpectTx)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


@expr.register_op(dialect="T", category="sync", name="mbarrier_wait_parity")
class MBarrierWaitParity(expr.Op):
    """Block until ``barrier``'s phase parity reaches ``phase``."""

    capability = expr.OpCapability("mbarrier.try_wait")
    execution_mesh = pattern.thread_execution_mesh()

    barrier = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.tensor_in(expr.StorageKind.SMEM)
    )
    phase = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.is_scalar_tensor())


@typeinfer.register(MBarrierWaitParity)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


@expr.register_op(dialect="T", category="sync", name="mbarrier_invalidate")
class MBarrierInvalidate(expr.Op):
    """Release ``barrier``'s shared-memory word for other use."""

    execution_mesh = pattern.thread_execution_mesh()

    barrier = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.WRITE, pattern=pattern.tensor_in(expr.StorageKind.SMEM)
    )


@typeinfer.register(MBarrierInvalidate)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


def barrier_access(call, ctx):
    inputs = [access.state(ctx.type_of(call.args[0]))]
    if len(call.args) > 1:
        inputs.append(access.identity(0))
    return access.iterating((), (*inputs, access.identity(0)))


access.register_relation(MBarrierInit)(barrier_access)
access.register_relation(MBarrierArriveExpectTx)(barrier_access)
access.register_relation(MBarrierWaitParity)(barrier_access)
access.register_relation(MBarrierInvalidate)(barrier_access)
