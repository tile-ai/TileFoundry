"""TIR abort effect operation."""

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op(dialect="T", category="control")
class Abort(Op):
    execution_mesh = utils.thread_execution_mesh()
    message = ParamDef(kind="attribute", annotation=str, default="")


@register_typeinfer(Abort)
def _(call, ctx) -> UnitType:
    return UnitType()


register_access_relation(Abort)(identity_relations)


@register_verify_stmt(Abort)
def _verify_abort(call, ctx) -> None:
    """Admit a control effect with no input contract.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    return None
