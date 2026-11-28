"""Emit one lowered ANE entry: walk its statements into passes, then frame the descriptors."""

from tilefoundry import codegen
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.tir import memory
from tilefoundry_apple.ane import isa, target
from tilefoundry_apple.ane.codegen import context, program

ENGINE_MESH = (("engine", 1),)
FP16_BYTES = 2


def view_binding(value: expr.Expr) -> tuple[expr.Expr, int]:
    """The source tensor and byte address of a bound view.

    A view over a slice names that slice's first element, so the DMA programs
    the displacement of the window instead of the tensor base. An SMEM view at a
    constant offset is an allocation; the offset is its L2 byte base.
    """
    if isinstance(value, expr.Var):
        if not isinstance(value.type, types.TensorType):
            raise ValueError("ANE bound view source must be a tensor")
        return value, 0
    if not isinstance(value, expr.Call) or not isinstance(value.target, memory.TensorView):
        raise ValueError("ANE view expression is not decoded yet")
    if len(value.args) != 1:
        raise ValueError("TensorView must have exactly one pointer operand")
    pointer, = value.args
    if isinstance(pointer, expr.Constant):
        if (isinstance(value.type, types.TensorType)
                and value.type.storage is expr.StorageKind.SMEM
                and type(pointer.value) is int):
            return value, pointer.value
        raise ValueError("ANE integer TensorView allocation is not decoded yet")
    if not isinstance(pointer, expr.Call) or not isinstance(pointer.target, memory.PtrOf):
        raise ValueError("ANE view pointer expression is not decoded yet")
    if len(pointer.args) != 1:
        raise ValueError("PtrOf must have exactly one tensor operand")
    window = pointer.args[0]
    if isinstance(window, expr.Var):
        if not isinstance(window.type, types.TensorType):
            raise ValueError("PtrOf input must be a tensor")
        return window, 0
    if not isinstance(window, expr.Call) or not isinstance(window.target, tensor.Slice):
        raise ValueError("ANE view window expression is not decoded yet")
    if len(window.args) != 2:
        raise ValueError("Slice must have a tensor and a starts tuple")
    source, starts = window.args
    if not isinstance(source, expr.Var) or not isinstance(source.type, types.TensorType):
        raise ValueError("ANE slice source expression is not decoded yet")
    layout = source.type.layout
    if not isinstance(layout, types.Layout):
        raise ValueError("ANE slice layout is not decoded yet")
    if source.type.dtype is not expr.DType.f16:
        raise ValueError("ANE slice address requires an fp16 tensor")
    rank = len(source.type.shape)
    if (not isinstance(starts, expr.Tuple) or len(starts.elements) != rank
            or len(layout.strides) != rank
            or len(window.target.sizes) != rank or len(window.target.strides) != rank):
        raise ValueError("Slice starts, sizes, strides and layout axes must match the tensor rank")
    total = 0
    for axis, begin in enumerate(starts.elements):
        if not isinstance(begin, expr.Constant):
            raise ValueError("ANE nonconstant slice start is not decoded yet")
        if type(begin.value) is not int or begin.value < 0:
            raise ValueError("Slice starts must be non-negative integer constants")
        step = expr.static_dim_value(layout.strides[axis])
        extent = expr.static_dim_value(source.type.shape[axis])
        size = expr.static_dim_value(window.target.sizes[axis])
        stride = expr.static_dim_value(window.target.strides[axis])
        if None in (step, extent, size, stride):
            raise ValueError("ANE dynamic slice geometry is not decoded yet")
        if size < 0 or stride <= 0:
            raise ValueError("Slice sizes must be non-negative and strides must be positive")
        if begin.value >= extent or (size and begin.value + (size - 1) * stride >= extent):
            raise ValueError(f"Slice window leaves source axis {axis}")
        total += begin.value * step
    return source, total * FP16_BYTES


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
