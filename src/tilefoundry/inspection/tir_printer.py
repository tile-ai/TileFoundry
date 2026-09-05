"""Canonical Python DSL printer for TIR PrimFunction and Module values."""

from __future__ import annotations

import enum
import re

from tilefoundry.inspection._python_render import (
    mesh_str,
    pattern_ctor,
    tensor_annotation,
)
from tilefoundry.inspection.print_context import TirPrintContext
from tilefoundry.inspection.python_printer import PythonPrinter
from tilefoundry.ir.core import Call, Constant, Op, Tuple, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import (
    Evaluate,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Mesh
from tilefoundry.ir.visitor import StmtVisitor


class TirPrinter(PythonPrinter, StmtVisitor[list[str]]):
    """TIR statement printer with explicit statement-visitor entry points.

    The legacy helpers below remain as the formatting implementation while the
    visitor façade provides the stable dispatch surface used by callers.
    """

    def __init__(self, *, context: TirPrintContext | None = None, indent: str = "") -> None:
        PythonPrinter.__init__(self)
        self.context = context or TirPrintContext()
        self.indent = indent

    def visit(self, stmt, ctx=None):  # type: ignore[override]
        return StmtVisitor.visit(self, stmt)

    def visit_Sequential(self, stmt):
        return [line for child in stmt.body for line in self.visit(child)]

    def visit_LetStmt(self, stmt):
        return [f"{self.indent}{stmt.var.name} = {_expr(stmt.value, self.context_mesh_bindings())}"] + self.visit(stmt.body)

    def visit_Evaluate(self, stmt):
        return _emit_evaluate(stmt, self.indent, self.context_mesh_bindings())

    def visit_MeshScope(self, stmt):
        lines = [f"{self.indent}with {mesh_str(stmt.mesh)} as {stmt.binding.name}:"]
        self.context.push_mesh(stmt.mesh, stmt.binding.name)
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.body))
        self.context.pop_mesh()
        return lines

    def visit_For(self, stmt):
        lines = [f"{self.indent}for {stmt.induction_var.name} in range({_expr(stmt.start)}, {_expr(stmt.stop)}, {_expr(stmt.step)}):"]
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.body))
        return lines

    def visit_If(self, stmt):
        lines = [f"{self.indent}if {_expr(stmt.cond, self.context_mesh_bindings())}:"]
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.then_body))
        if stmt.else_body.body:
            lines.append(f"{self.indent}else:")
            lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.else_body))
        return lines

    def visit_While(self, stmt):
        return [f"{self.indent}while {_expr(stmt.cond, self.context_mesh_bindings())}:"] + TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.body)

    def visit_Return(self, stmt):
        return [f"{self.indent}return"]

    def visit_Abort(self, stmt):
        return [f"{self.indent}abort({stmt.message!r})"]

    def visit_DispatchCall(self, stmt):
        cases = []
        for patterns, call in zip(stmt.case_patterns, stmt.case_calls):
            pats = ", ".join(pattern_ctor(pattern) for pattern in patterns)
            args = _join_args(call.args, self.context_mesh_bindings())
            cases.append(f"(({pats},), {_binding_name(call.callable.name)!r}, ({args},))")
        lines = [f"{self.indent}with dispatch_call({stmt.callee_name!r}, subjects=({_join_args(stmt.subjects, self.context_mesh_bindings())},), cases=({', '.join(cases)},)):"]
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.fallback))
        return lines

    def context_mesh_bindings(self) -> dict[int, str]:
        aliases: dict[int, str] = {}
        for frame in self.context._mesh_aliases:
            aliases.update(frame)
        return aliases


_STMT_PRINTERS: dict[type, object] = {}


def register_tir_printer(node_type: type):
    """Register the source emitter for one TIR callable/statement type."""
    def decorate(fn):
        _STMT_PRINTERS[node_type] = fn
        return fn
    return decorate


def _binding_name(name: str) -> str:
    return re.sub(r"\W", "_", name)


def _python_value(value: object, mesh_bindings: dict[int, str] | None = None) -> str:
    """Render descriptor attributes through their canonical PythonExpr form."""
    mesh_bindings = mesh_bindings or {}
    if isinstance(value, Mesh) and id(value) in mesh_bindings:
        return mesh_bindings[id(value)]
    to_python = getattr(value, "to_python", None)
    if callable(to_python):
        return to_python().text
    if isinstance(value, TensorType):
        return tensor_annotation(value)
    if isinstance(value, DType):
        return repr(value.name)
    if isinstance(value, MmaAtom):
        return f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (str, int, float, bool, tuple)):
        return repr(value)
    raise NotImplementedError(
        f"TIR printer has no canonical attribute form for {type(value).__name__}"
    )


def _expr(expr: object, mesh_bindings: dict[int, str] | None = None) -> str:
    mesh_bindings = {} if mesh_bindings is None else mesh_bindings
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Constant):
        return repr(expr.value)
    if isinstance(expr, SymbolRef):
        return _binding_name(expr.name)
    if isinstance(expr, ShapeOf):
        return f"shape_of({expr.param.name}, axis={expr.axis})"
    if isinstance(expr, Op):
        name = getattr(getattr(expr, "_op_schema", None), "name", type(expr).__name__.lower())
        return f"T.{name}"
    if isinstance(expr, Tuple):
        values = ", ".join(_expr(x, mesh_bindings) for x in expr.elements)
        if len(expr.elements) == 1:
            values += ","
        return f"({values})"
    if isinstance(expr, Call):
        target = expr.target
        scalar_binary = {
            BinaryKind.EQ: "==",
            BinaryKind.NE: "!=",
            BinaryKind.LT: "<",
            BinaryKind.LE: "<=",
            BinaryKind.GT: ">",
            BinaryKind.GE: ">=",
            BinaryKind.AND: "and",
        }
        kind = getattr(target, "kind", None)
        if kind in scalar_binary and len(expr.args) == 2 and expr.type.dtype is DType.bool:
            return f"{_expr(expr.args[0], mesh_bindings)} {scalar_binary[kind]} {_expr(expr.args[1], mesh_bindings)}"
        op_name = getattr(getattr(target, "_op_schema", None), "name", None)
        if op_name is None:
            op_name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(target).__name__).lower()
        args = [_expr(x, mesh_bindings) for x in expr.args]
        for p in type(target).params():
            if p.kind != "attribute":
                continue
            value = getattr(target, p.name, None)
            if value is None:
                continue
            value = _python_value(value, mesh_bindings)
            args.append(f"{p.name}={value}")
        return f"T.{op_name}({', '.join(args)})"
    raise NotImplementedError(f"TIR printer has no canonical expression form for {type(expr).__name__}")


@register_tir_printer(Launch)
def _print_launch(stmt: Evaluate, indent: str, mesh_bindings: dict[int, str]) -> list[str]:
    callee, grid = stmt.args[0], stmt.args[1:4]
    block = stmt.args[4:7]
    forwarded = stmt.args[7:]
    return [
        f"{indent}launch({_expr(callee, mesh_bindings)}, {_join_args(forwarded, mesh_bindings)}, "
        f"grid={_expr(Tuple(type=grid[0].type, elements=tuple(grid)))}, "
        f"block={_expr(Tuple(type=block[0].type, elements=tuple(block)))})  # noqa: F821"
    ]


@register_tir_printer(Op)
def _print_op_evaluate(stmt: Evaluate, indent: str, mesh_bindings: dict[int, str]) -> list[str]:
    target = stmt.callable
    args = list(stmt.args)
    attrs = []
    op_name = getattr(getattr(target, "_op_schema", None), "name", None)
    for p in type(target).params():
        if p.kind != "attribute":
            continue
        value = getattr(target, p.name, None)
        if value is None:
            continue
        rendered = _python_value(value, mesh_bindings)
        attrs.append(rendered if op_name == "sync" and p.name == "mesh" else f"{p.name}={rendered}")
    rendered_args = [_expr(arg, mesh_bindings) for arg in args]
    return [f"{indent}{_expr(target, mesh_bindings)}({', '.join(rendered_args + attrs)})"]


def _emit_evaluate(stmt: Evaluate, indent: str, mesh_bindings: dict[int, str]) -> list[str]:
    handler = _STMT_PRINTERS.get(type(stmt.callable))
    if handler is None and isinstance(stmt.callable, Op):
        handler = _STMT_PRINTERS[Op]
    if handler is None:
        raise NotImplementedError(f"TIR printer has no emitter for {type(stmt.callable).__name__}")
    return handler(stmt, indent, mesh_bindings)


def _emit_stmt(stmt, indent: str, lines: list[str], mesh_bindings=None) -> None:
    lines.extend(TirPrinter(context=TirPrintContext(), indent=indent).visit(stmt))


def _join_args(args, mesh_bindings=None) -> str:
    return ", ".join(_expr(arg, mesh_bindings) for arg in args)


def _function_block(fn: PrimFunction) -> list[str]:
    target = fn.target.to_python()
    lines = ["@prim_func(target=" + target.text + ")"]
    params = ", ".join(
        f"{p.name}: {tensor_annotation(p.type) if isinstance(p.type, TensorType) else repr(p.type)}"
        for p in fn.params
    )
    lines.append(f"def {_binding_name(fn.name)}({params}):")
    body = TirPrinter(indent="    ").visit(fn.body)
    lines.extend(body or ["    pass"])
    return lines


def _imports(text: str, targets: set[str], *, module: bool) -> list[str]:
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
    names = ["module"] if module else []
    names.append("prim_func")
    lines = [f"from tilefoundry import {', '.join(names)}"]
    dsl_names = [name for name in ("T", "Tensor") if name in identifiers]
    if dsl_names:
        lines.append(f"from tilefoundry.dsl import {', '.join(dsl_names)}")
    kind_names = sorted(name for name in identifiers if name.endswith("Kind"))
    if kind_names:
        lines.append(f"from tilefoundry.ir.core.kinds import {', '.join(kind_names)}")
    shard_names = [
        name
        for name in ("B", "ComposedLayout", "Layout", "Mesh", "P", "S", "ShardLayout", "Topology")
        if name in identifiers
    ]
    if shard_names:
        lines.append(f"from tilefoundry.ir.types.shard import {', '.join(shard_names)}")
    if targets:
        lines.append(f"from tilefoundry.target import {', '.join(sorted(targets))}")
    return lines


def tir_function_to_python(fn: PrimFunction, *, options=None) -> str:
    lines = _function_block(fn)
    text = "\n".join(lines)
    targets = {type(fn.target).__name__}
    lines = ["from __future__ import annotations", "", *_imports(text, targets, module=False), "", "", *lines]
    return "\n".join(lines) + "\n"


def tir_module_to_python(mod: Module, module_name: str | None = None, *, options=None) -> str:
    name = module_name or mod.name
    lines: list[str] = []
    kwargs = []
    if mod.entry is not None:
        kwargs.append(f'entry="{_binding_name(mod.entry)}"')
    if mod.target is not None:
        kwargs.append(f"target={mod.target.to_python().text}")
    if mod.topologies is not None:
        rendered = ", ".join(f'Topology("{t.name}", {t.size!r})' for t in mod.topologies)
        kwargs.append(f"topologies=({rendered},)" if rendered else "topologies=()")
    lines.append(f"@module({', '.join(kwargs)})")
    lines.append(f"class {name}:")
    blocks: list[list[str]] = []
    for child in mod.modules:
        child_source = tir_module_to_python(child, child.name, options=options).splitlines()
        class_at = next(i for i, line in enumerate(child_source) if line.startswith("@module"))
        blocks.append(child_source[class_at:])
    for fn in mod.functions:
        if not isinstance(fn, PrimFunction):
            if isinstance(fn, HirFunction):
                raise NotImplementedError("mixed HIR/TIR module printing is not yet supported")
            raise TypeError(f"TIR printer cannot serialize {type(fn).__name__}")
        blocks.append(_function_block(fn))
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend("    " + line if line else line for line in block)
    text = "\n".join(lines)
    targets = {type(fn.target).__name__ for fn in mod.functions if isinstance(fn, PrimFunction)}
    if mod.target is not None:
        targets.add(type(mod.target).__name__)
    header = ["from __future__ import annotations", "", *_imports(text, targets, module=True), "", ""]
    return "\n".join(header + lines) + "\n"
