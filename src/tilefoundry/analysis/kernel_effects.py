"""Derive conservative external accesses from verified local-memory kernels."""

from tilefoundry.interfaces import execution
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.tir import memory, sync


class Collector(core.StmtVisitor):
    def __init__(self, function):
        self.params = function.params
        self.bindings = {id(param): frozenset({index}) for index, param in enumerate(function.params)
                         if isinstance(param.type, types.TensorType)}
        self.effects = {}

    def roots(self, value):
        if isinstance(value, expr.Var):
            if id(value) not in self.bindings:
                raise ValueError(f"unbound kernel operand: {value.name}")
            return self.bindings[id(value)]
        if isinstance(value, (expr.Constant, core.ShapeOf)):
            return frozenset()
        if isinstance(value, expr.Call):
            roots = frozenset().union(*(self.roots(arg) for arg in value.args))
            if roots and not isinstance(value.target, (memory.PtrOf, memory.TensorView)):
                raise ValueError(f"unmodeled kernel alias or expression: {type(value.target).__name__}")
            if roots and isinstance(value.target, memory.TensorView):
                ty = value.type
                layout = ty.layout.layout if isinstance(ty.layout, types.ShardLayout) else ty.layout
                if layout != types.Layout(ty.shape) or any(
                    ty.shape != self.params[index].type.shape or ty.dtype != self.params[index].type.dtype
                    for index in roots
                ):
                    raise ValueError("kernel effect summary requires full compact parameter views")
            return roots
        raise ValueError(f"unmodeled kernel operand: {type(value).__name__}")

    def visit_LetStmt(self, stmt):
        self.bindings[id(stmt.var)] = self.roots(stmt.value)

    def visit_Evaluate(self, stmt):
        if isinstance(stmt.callable, sync.Sync) and not stmt.args:
            return
        if not isinstance(stmt.callable, expr.Op) or not expr.capabilities_of(type(stmt.callable)):
            raise ValueError(f"unmodeled kernel effect: {type(stmt.callable).__name__}")
        params = tuple(param for param in expr.signature_of(type(stmt.callable)) if param.kind == "input")
        if len(params) != len(stmt.args):
            raise ValueError("kernel effect summary requires all instruction operands to be bound")
        for param, arg in zip(params, stmt.args, strict=True):
            if param.effect is None:
                raise ValueError(f"undeclared kernel effect: {type(stmt.callable).__name__}.{param.name}")
            for index in self.roots(arg):
                self.effects[index] = self.effects.get(index, expr.MemoryEffect(0)) | param.effect

    def visit_Return(self, stmt):
        raise ValueError("kernel effect summary does not support early returns")

    def visit_If(self, stmt):
        raise ValueError("kernel effect summary does not yet merge conditional aliases")

    def visit_While(self, stmt):
        raise ValueError("kernel effect summary requires static control flow")

    def visit_For(self, stmt):
        bounds = tuple(expr.static_dim_value(value) for value in (stmt.start, stmt.stop, stmt.step))
        if any(value is None for value in bounds) or bounds[2] == 0:
            raise ValueError("kernel effect summary requires static nonzero-step loops")
        self.bindings[id(stmt.induction_var)] = frozenset()
        if range(*bounds):
            self.visit(stmt.body)


def summarize(function: core.PrimFunction) -> execution.Summary:
    """Summarize a kernel whose declared local accesses finish at kernel completion.

    Kernel-internal synchronization remains its own contract. Remote operations
    require an explicit remote completion contract, not this local summary.
    """
    collector = Collector(function)
    collector.visit(function.body)
    return execution.Summary(tuple(sorted(collector.effects.items())))


__all__ = ["summarize"]
