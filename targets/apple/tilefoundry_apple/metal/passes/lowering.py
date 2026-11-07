"""Issue Metal matrix instructions through the standard lowering boundary."""
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.passes.transforms.convert_hir_to_tir import (
    _BINDING,
    ConvertHIRToTIR,
    Lowering,
    LoweringError,
)
from tilefoundry.target import bind

from ..target import MetalTarget


class MetalLowering(Lowering):
    def _matrix(self, call, op, operands, ctx):
        plan, types, owner = ctx.issues.plan(call, op, ctx.enclosing_mesh())
        params = plan.params
        writes = [p for p in params if p.effect & MemoryEffect.WRITE]
        if len(writes) != 1:
            raise LoweringError("Metal matrix lowering requires one destination")
        destination = writes[0]
        scheduled = isinstance(call.target, ScheduleOp)
        supplied = params if scheduled else tuple(p for p in params if p.effect & MemoryEffect.READ)
        written = ctx.result(call)
        values = {p.name: value for p, value in zip(supplied, operands, strict=True)}
        if not destination.effect & MemoryEffect.READ:
            values[destination.name] = written
        with ctx.builder.region(mesh=owner, binding=Var(ctx.names.fresh("threads"), type=_BINDING)):
            issued = tuple(ctx.builder.value(ctx.storage.operand(values[p.name], t,
                stem=f"{p.name}_view"), ctx=ctx) for p, t in zip(params, types, strict=True))
            ctx.builder.add(Evaluate(op, issued))
        return values[destination.name]

    visit_T_MatMul = _matrix
    visit_T_Mma8x8x8 = _matrix


@bind(MetalTarget, ConvertHIRToTIR)
class MetalConvertHIRToTIR(ConvertHIRToTIR):
    visitor = MetalLowering
