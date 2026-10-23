"""Effect-form TIR Ops for the CUDA shared-memory barrier object (mbarrier).

See [tir](targets/cuda/tilefoundry_cuda/docs/spec/tir.md#barrier-object-ops-tirsyncmbarrier_).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import WildcardPattern, is_scalar_tensor, utils
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_access,
    iterating,
    register_access_relation,
    state_access,
)

__all__ = [
    "MBarrierInit",
    "MBarrierArriveExpectTx",
    "MBarrierWaitParity",
    "MBarrierInvalidate",
]


@register_op(dialect="T", category="sync", name="mbarrier_init")
class MBarrierInit(Op):
    """Arm ``barrier`` so ``arrive_count`` arrivals complete a phase."""

    execution_mesh = utils.thread_execution_mesh()

    barrier = ParamDef(
        kind="input", effect=MemoryEffect.WRITE, pattern=utils.tensor_in(StorageKind.SMEM)
    )
    arrive_count = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("arrive_count", predicates=(WildcardPattern("arrive_count") > 0,)),
    )


@register_typeinfer(MBarrierInit)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_op(dialect="T", category="sync", name="mbarrier_arrive_expect_tx")
class MBarrierArriveExpectTx(Op):
    """Arrive on ``barrier`` and declare ``tx_bytes`` of asynchronous data."""

    capability = OpCapability("mbarrier.expect_tx")
    execution_mesh = utils.thread_execution_mesh()

    barrier = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=utils.tensor_in(StorageKind.SMEM),
    )
    tx_bytes = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("tx_bytes", predicates=(WildcardPattern("tx_bytes") > 0,)),
    )


@register_typeinfer(MBarrierArriveExpectTx)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_op(dialect="T", category="sync", name="mbarrier_wait_parity")
class MBarrierWaitParity(Op):
    """Block until ``barrier``'s phase parity reaches ``phase``."""

    capability = OpCapability("mbarrier.try_wait")
    execution_mesh = utils.thread_execution_mesh()

    barrier = ParamDef(
        kind="input", effect=MemoryEffect.READ, pattern=utils.tensor_in(StorageKind.SMEM)
    )
    phase = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=is_scalar_tensor())


@register_typeinfer(MBarrierWaitParity)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_op(dialect="T", category="sync", name="mbarrier_invalidate")
class MBarrierInvalidate(Op):
    """Release ``barrier``'s shared-memory word for other use."""

    execution_mesh = utils.thread_execution_mesh()

    barrier = ParamDef(
        kind="input", effect=MemoryEffect.WRITE, pattern=utils.tensor_in(StorageKind.SMEM)
    )


@register_typeinfer(MBarrierInvalidate)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


def _barrier_access(call, ctx):
    inputs = [state_access(ctx.type_of(call.args[0]))]
    if len(call.args) > 1:
        inputs.append(identity_access(0))
    return iterating((), (*inputs, identity_access(0)))


register_access_relation(MBarrierInit)(_barrier_access)
register_access_relation(MBarrierArriveExpectTx)(_barrier_access)
register_access_relation(MBarrierWaitParity)(_barrier_access)
register_access_relation(MBarrierInvalidate)(_barrier_access)
