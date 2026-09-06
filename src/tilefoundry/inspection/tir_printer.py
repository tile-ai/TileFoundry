"""Canonical Python DSL printer for TIR PrimFunction and Module values."""

from __future__ import annotations

import re

from tilefoundry.inspection.print_context import TirPrintContext
from tilefoundry.inspection.printer_base import PythonPrinter
from tilefoundry.ir.core import Call, Constant, Op, Tuple, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import (
    Evaluate,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.utils.python_source import PythonExpr, _merge_imports


class _RenderedLines(list):
    def __init__(self, lines, imports):
        super().__init__(lines)
        self.imports = imports


class TirPrinter(PythonPrinter, StmtVisitor[list[str]]):
    """TIR statement printer with explicit statement-visitor entry points.

    The legacy helpers below remain as the formatting implementation while the
    visitor façade provides the stable dispatch surface used by callers.
    """

    def __init__(self, *, context: TirPrintContext | None = None, indent: str = "") -> None:
        PythonPrinter.__init__(self)
        self.context = context or TirPrintContext()
        self.indent = indent

    def dim_entry(self, value, ctx=None) -> str:
        return f"_{value.name}" if hasattr(value, "name") else str(value)

    def visit(self, stmt, ctx=None):  # type: ignore[override]
        return StmtVisitor.visit(self, stmt)

    def visit_Sequential(self, stmt):
        return [line for child in stmt.body for line in self.visit(child)]

    def visit_LetStmt(self, stmt):
        return [f"{self.indent}{stmt.var.name} = {self._expr(stmt.value)}"] + self.visit(stmt.body)

    def visit_Evaluate(self, stmt):
        return self._emit_evaluate(stmt)

    def visit_MeshScope(self, stmt):
        lines = [f"{self.indent}with {self._render_mesh_compact(stmt.mesh, self.context)} as {stmt.binding.name}:"]
        self.context.push_mesh(stmt.mesh, stmt.binding.name)
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.body))
        self.context.pop_mesh()
        return lines

    def visit_For(self, stmt):
        lines = [f"{self.indent}for {stmt.induction_var.name} in range({self._expr(stmt.start)}, {self._expr(stmt.stop)}, {self._expr(stmt.step)}):"]
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.body))
        return lines

    def visit_If(self, stmt):
        lines = [f"{self.indent}if {self._expr(stmt.cond)}:"]
        lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.then_body))
        if stmt.else_body.body:
            lines.append(f"{self.indent}else:")
            lines.extend(TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.else_body))
        return lines

    def visit_While(self, stmt):
        return [f"{self.indent}while {self._expr(stmt.cond)}:"] + TirPrinter(context=self.context, indent=self.indent + "    ").visit(stmt.body)

    def visit_Return(self, stmt):
        return [f"{self.indent}return"]

    def _expr(self, expr):
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
            self.context.use(PythonExpr(("from tilefoundry.dsl import T",), "T"))
            return f"T.{name}"
        if isinstance(expr, Tuple):
            vals = ", ".join(self._expr(x) for x in expr.elements)
            return f"({vals}{',' if len(expr.elements)==1 else ''})"
        if isinstance(expr, Call):
            target = expr.target
            scalar_binary = {BinaryKind.EQ:"==", BinaryKind.NE:"!=", BinaryKind.LT:"<", BinaryKind.LE:"<=", BinaryKind.GT:">", BinaryKind.GE:">=", BinaryKind.AND:"and"}
            kind = getattr(target, "kind", None)
            if kind in scalar_binary and len(expr.args)==2 and expr.type.dtype is DType.bool:
                return f"{self._expr(expr.args[0])} {scalar_binary[kind]} {self._expr(expr.args[1])}"
            name = getattr(getattr(target, "_op_schema", None), "name", None) or re.sub(r"(?<!^)(?=[A-Z])", "_", type(target).__name__).lower()
            args = [self._expr(x) for x in expr.args]
            for p in type(target).params():
                if p.kind == "attribute":
                    value = getattr(target, p.name, None)
                    if value is not None:
                        args.append(f"{p.name}={self.render_value(value, self.context, self.indent + '    ')}")
            self.context.use(PythonExpr(("from tilefoundry.dsl import T",), "T"))
            return f"T.{name}({', '.join(args)})"
        return self.render_value(expr, self.context)

    def _join_args(self, args): return ", ".join(self._expr(arg) for arg in args)

    def _emit_evaluate(self, stmt):
        handler = _STMT_PRINTERS.get(type(stmt.callable)) or (_STMT_PRINTERS.get(Op) if isinstance(stmt.callable, Op) else None)
        if handler is None:
            raise NotImplementedError(f"TIR printer has no emitter for {type(stmt.callable).__name__}")
        return handler(stmt, self)


_STMT_PRINTERS: dict[type, object] = {}


def register_tir_printer(node_type: type):
    """Register the source emitter for one TIR callable/statement type."""
    def decorate(fn):
        _STMT_PRINTERS[node_type] = fn
        return fn
    return decorate


def _binding_name(name: str) -> str:
    return re.sub(r"\W", "_", name)


@register_tir_printer(Launch)
def _print_launch(stmt: Evaluate, printer: TirPrinter) -> list[str]:
    indent = printer.indent
    callee, grid = stmt.args[0], stmt.args[1:4]
    block = stmt.args[4:7]
    forwarded = stmt.args[7:]
    return [f"{indent}launch({printer._expr(callee)}, {printer._join_args(forwarded)}, grid={printer._expr(Tuple(type=grid[0].type, elements=tuple(grid)))}, block={printer._expr(Tuple(type=block[0].type, elements=tuple(block)))})  # noqa: F821"]


@register_tir_printer(Op)
def _print_op_evaluate(stmt: Evaluate, printer: TirPrinter) -> list[str]:
    indent = printer.indent
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
        rendered = printer.render_value(value, printer.context, printer.indent + "    ")
        attrs.append(rendered if op_name == "sync" and p.name == "mesh" else f"{p.name}={rendered}")
    rendered_args = [printer._expr(arg) for arg in args]
    return [f"{indent}{printer._expr(target)}({', '.join(rendered_args + attrs)})"]


def _function_block(fn: PrimFunction) -> list[str]:
    ctx = TirPrintContext()
    target = ctx.use(fn.target.to_python())
    ctx.use(PythonExpr(("from tilefoundry import prim_func",), "prim_func"))
    ctx.use(PythonExpr(("from tilefoundry.dsl import Tensor",), "Tensor"))
    dim_vars = {d.name: d for p in fn.params if isinstance(p.type, TensorType) for d in p.type.shape if hasattr(d, "name")}
    if dim_vars:
        ctx.use(PythonExpr(("from tilefoundry.dsl import DimVar",), "DimVar"))
    lines = [f'_{d.name} = DimVar("{d.name}", {d.lo}, {d.hi})' for d in dim_vars.values()]
    lines.append("@prim_func(target=" + target + ")")
    params = ", ".join(
        f"{p.name}: {TirPrinter(context=ctx).render_value(p.type, ctx) if isinstance(p.type, TensorType) else repr(p.type)}"
        for p in fn.params
    )
    lines.append(f"def {_binding_name(fn.name)}({params}):")
    body = TirPrinter(context=ctx, indent="    ").visit(fn.body)
    lines.extend(body or ["    pass"])
    if fn.variants:
        ctx.use(PythonExpr(("from tilefoundry.ir.core.pattern import DimVarRangePat",), "DimVarRangePat"))
    for variant in fn.variants:
        pat = variant.specializations[0]
        lines.append("")
        lines.append(f"@{_binding_name(fn.name)}.specialize({TirPrinter(context=ctx).render_pattern(pat, ctx)})")
        lines.append(f"def {_binding_name(getattr(variant, '_display_name', variant.name))}({params}):")
        vbody = TirPrinter(context=ctx, indent="    ").visit(variant.body)
        lines.extend(vbody or ["    pass"])
    return _RenderedLines(lines, ctx.imports)


def _imports_from(lines) -> list[str]:
    return list(_merge_imports(tuple(getattr(lines, "imports", ()))))


def tir_function_to_python(fn: PrimFunction, *, options=None) -> str:
    lines = _function_block(fn)
    lines = ["from __future__ import annotations", "", *_imports_from(lines), "", "", *lines]
    return "\n".join(lines) + "\n"


def tir_module_to_python(mod: Module, module_name: str | None = None, *, options=None) -> str:
    name = module_name or mod.name
    lines: list[str] = []
    imports = {"from tilefoundry import module"}
    kwargs = []
    if mod.entry is not None:
        kwargs.append(f'entry="{_binding_name(mod.entry)}"')
    if mod.target is not None:
        target = mod.target.to_python()
        imports.update(target.imports)
        kwargs.append(f"target={target.text}")
    if mod.topologies is not None:
        imports.add("from tilefoundry.ir.types.shard import Topology")
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
        block = _function_block(fn)
        imports.update(block.imports)
        blocks.append(block)
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend(("    " + line) if line else "" for line in block if " = DimVar(" not in line)
    declarations = list(dict.fromkeys(line for block in blocks for line in block if " = DimVar(" in line))
    if declarations:
        remaining = [line for line in lines if line not in declarations]
        while remaining and not remaining[0]:
            remaining.pop(0)
        lines = declarations + ["", ""] + remaining
    header = ["from __future__ import annotations", "", *_merge_imports(tuple(imports)), ""]
    if not declarations:
        header.append("")
    return "\n".join(header + lines) + "\n"
