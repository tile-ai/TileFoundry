"""TIR abort effect operation."""
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op(dialect="T", category="control")
class Abort(Op):
    message = ParamDef(kind="attribute", annotation=str, default="")

@register_typeinfer(Abort)
def _(call, ctx) -> UnitType:
    return UnitType()


@register_verify_stmt(Abort)
def _(call, ctx) -> None:
    return None
