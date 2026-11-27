"""Emit one lowered ANE entry: walk its statements into passes, then frame the descriptors."""

from tilefoundry import codegen
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import core, expr, types
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context, program

ENGINE_MESH = (("engine", 1),)
FP16_BYTES = 2


def view_binding(value: expr.Expr) -> tuple[expr.Expr, int]:
    """The source tensor and byte address of a bound view.

    A view over a slice names that slice's first element, so the DMA programs
    the displacement of the window instead of the tensor base.
    """
    if not isinstance(value, expr.Call):
        return value, 0
    pointer = value.args[0] if value.args else None
    if not isinstance(pointer, expr.Call) or len(pointer.args) != 1:
        return value, 0
    window = pointer.args[0]
    if isinstance(window, expr.Var):
        return window, 0
    if not isinstance(window, expr.Call) or len(window.args) != 2:
        return value, 0
    tensor, starts = window.args
    layout = getattr(getattr(tensor, "type", None), "layout", None)
    if not isinstance(layout, types.Layout):
        return value, 0
    total = 0
    for begin, step in zip(getattr(starts, "elements", ()), layout.strides, strict=False):
        index = begin.value if isinstance(begin, expr.Constant) else 0
        if not isinstance(index, int):
            return value, 0
        total += index * step
    return tensor, total * FP16_BYTES


class Walker(core.StmtVisitor[None]):
    """Visit the statements of one entry, emitting each ``T.ane.*`` call as a pass."""

    def __init__(self, ctx: context.Context) -> None:
        self.ctx = ctx
        self.calls: list[expr.Call] = []

    def generic_visit(self, stmt, label) -> None:
        raise ValueError(f"{label}: cannot emit {type(stmt).__name__} as an ANE task descriptor")

    def visit_Sequential(self, stmt: core.Sequential, label: str) -> None:
        super().generic_visit(stmt, label)

    def visit_LetStmt(self, stmt: core.LetStmt, label: str) -> None:
        self.ctx.bind(stmt.var, *view_binding(stmt.value))

    def visit_MeshScope(self, stmt: core.MeshScope, label: str) -> None:
        if tuple((item.name, item.size) for item in stmt.mesh.topologies) != ENGINE_MESH:
            raise ValueError(f"{label}: expected one ANE engine in the mesh")
        super().generic_visit(stmt, label)

    def visit_Evaluate(self, stmt: core.Evaluate, label: str) -> None:
        call = expr.Call(stmt.callable, stmt.args, type=types.TupleType.Void)
        self.calls.append(call)


@codegen.register(target.Target, codegen.Role.EMIT, core.PrimFunction)
def emit(fn: core.PrimFunction, ctx: context.Context) -> None:
    label = f"{tilefoundry_target.summary(fn.target)} {fn.name}"
    if not isinstance(fn.target, target.Target) or fn.target.architecture.name != "apple_h14":
        raise ValueError(f"{label}: expected an H14 ANE target")
    if fn.specializations or fn.variants:
        raise ValueError(f"{label}: descriptor emission does not support specialization")
    walker = Walker(ctx)
    walker.visit(fn.body, label)
    ctx.plan(walker.calls)
    for call in walker.calls:
        ctx.emit_call(call, label)
    if fn.output_count != 1 or len(fn.params) < 2:
        raise ValueError(f"{label}: expected one output and at least one input parameter")
    bound = tuple(ctx.addresses)
    sources = (*fn.params[:-1], *bound)
    for args in ctx.operands:
        if len(args) < 2:
            raise ValueError(f"{label}: every pass must name an output and a source")
        if args[0] not in (fn.params[-1], *bound):
            raise ValueError(f"{label}: the schedule must write the output parameter first")
        if any(arg not in sources for arg in args[1:]):
            raise ValueError(f"{label}: the schedule must read the function's inputs")


def build_context(fn: core.PrimFunction) -> context.Context:
    """Emit the passes and retain their symbolic operands."""
    ctx = context.Context()
    emit(fn, ctx)
    return ctx


def build_program(fn: core.PrimFunction) -> isa.Program:
    """The task descriptor stream of one lowered H14 entry."""
    return program.to_program(build_context(fn).passes)


__all__ = ["Walker", "build_context", "build_program", "view_binding"]
