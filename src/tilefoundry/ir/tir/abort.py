"""TIR abort effect operation."""

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect="T", category="control")
class Abort(expr.Op):
    execution_mesh = pattern.thread_execution_mesh()
    message = expr.ParamDef(kind="attribute", annotation=str, default="")


@typeinfer.register(Abort)
def _(call, ctx) -> types.UnitType:
    return types.UnitType()


access.register_relation(Abort)(access.identity_relations)


@verify.register(Abort)
def verify_abort(call, ctx) -> None:
    """Admit a control effect with no input contract.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    return None


__all__ = ["Abort"]
