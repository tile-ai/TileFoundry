"""Issue ANE instructions: the target half of HIR-to-TIR lowering."""

from tilefoundry import analysis, target
from tilefoundry.ir import core, expr, hir
from tilefoundry.passes import transforms
from tilefoundry_apple.ane import target as ane_target


class Lowering(transforms.Lowering):
    """The core lowering, issuing each ANE instruction once in the engine's region."""

    def default_visit_T(self, call: expr.Call, op, operands, ctx) -> expr.Expr:
        if not expr.get_registered_name(type(op)).startswith("T.ane."):
            return super().default_visit_T(call, op, operands, ctx)
        mesh = ctx.innermost_mesh()
        if mesh is None:
            raise transforms.LoweringError(
                f"{transforms.label(call)} instruction issue has no lexical mesh"
            )
        plan, type_list, owner = ctx.issues.plan(call, op, mesh)
        params = plan.params
        (destination,) = (param for param in params if param.effect & expr.MemoryEffect.WRITE)
        scheduled = isinstance(call.target, hir.schedule.ScheduleOp)
        supplied = (
            params if scheduled else tuple(p for p in params if p.effect & expr.MemoryEffect.READ)
        )
        values = {param.name: value for param, value in zip(supplied, operands, strict=True)}
        if destination.name not in values:
            values[destination.name] = ctx.result(call)
        with ctx.builder.region(
            mesh=owner, binding=expr.Var(ctx.names.fresh("engine"), type=transforms.BINDING)
        ):
            issued = tuple(
                ctx.builder.value(
                    ctx.storage.operand(values[param.name], type_, stem=f"{param.name}_view"),
                    ctx=ctx,
                )
                for param, type_ in zip(params, type_list, strict=True)
            )
            ctx.builder.add(core.Evaluate(op, issued))
        return values[destination.name]

    def visit_InsertSlice(self, call: expr.Call, operands, ctx) -> expr.Expr:
        """Return the destination a ``T.ane.store`` already wrote the update's window of.

        The engine copies nothing, so the result must share the destination's storage
        and the update must sit in the result's window.
        """
        result = core.SSAValue(call)
        placed = (
            (result, analysis.AliasKind.INSERT_SLICE, core.SSAValue(call.args[0])),
            (core.SSAValue(call.args[1]), analysis.AliasKind.WINDOW, result),
        )
        if not all(
            any(alias.kind is kind and alias.target == owner for alias in ctx.storage.aliases_of(value))
            for value, kind, owner in placed
        ):
            raise transforms.LoweringError(
                f"{transforms.label(call)} needs a copy; store the update into the window instead"
            )
        return ctx.result(call)


@target.bind(ane_target.Target, transforms.ConvertHIRToTIR)
class ConvertHIRToTIR(transforms.ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that issues ANE instructions."""

    visitor = Lowering


__all__ = ["ConvertHIRToTIR", "Lowering"]
