"""Effect-ful TIR Op ``tir.memory.Copy``.

Copies ``src`` to ``dst``. Memory direction (gmem / smem
/ rmem) is inferred from ``.type.storage`` of each operand. Covers
Load / Store too. The Op is placed in Stmt position as
``Evaluate(Copy, ...)``; the invocation is unit-typed (no result value).
"""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import SameConstraint, utils
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Copy(Op):
    """Copies ``src`` into ``dst`` (in-place memory write)."""

    capability = OpCapability(None)

    execution_mesh = utils.thread_execution_mesh()

    @classmethod
    def execution_mesh_pattern(cls):
        return utils.memory_execution_mesh()

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=utils.operand_tile(0, execution_mesh=utils.memory_execution_mesh()),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=utils.operand_tile(1, execution_mesh=utils.memory_execution_mesh()),
    )

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_typeinfer(Copy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


register_access_relation(Copy)(identity_relations)


@register_schedule_eval(Copy)
def _eval_scheduled_copy(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)
