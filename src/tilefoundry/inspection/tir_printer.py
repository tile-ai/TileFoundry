"""Canonical Python DSL printer for TIR PrimFunction and Module values."""

from __future__ import annotations

from tilefoundry.inspection.print_context import TirPrintContext
from tilefoundry.inspection.printer_base import PythonPrinter
from tilefoundry.ir.core import Call, Constant, Op, Tuple, Var, get_metadata
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    LetStmt,
    MeshScope,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import DType, LayoutBase, PointerType, TensorType, UnitType
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor

_LINE_LENGTH = 100


def _split_top_level(text: str) -> list[str]:
    """Split comma-separated Python fragments without cutting nested forms."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


class _RenderedLines(list):
    def __init__(self, lines, imports):
        super().__init__(lines)
        self.imports = imports


class TirPrinter(PythonPrinter, StmtVisitor[list[str]], CallFrameVisitor[str]):
    """TIR statement printer with explicit statement-visitor entry points.

    The legacy helpers below remain as the formatting implementation while the
    visitor façade provides the stable dispatch surface used by callers.
    """

    def __init__(self, *, context: TirPrintContext | None = None, indent: str = "") -> None:
        PythonPrinter.__init__(self)
        self.context = context or TirPrintContext()
        self.indent = indent

    def visit_DimVar(self, value, ctx=None) -> str:
        return f"_{value.name}"

    def dim_entry(self, value, ctx=None, *, nested: bool = False) -> str:
        return super().dim_entry(value, ctx, nested=nested)

    def visit(self, node, ctx=None):  # type: ignore[override]
        """Dispatch statements, expressions, and types by their concrete name."""
        return PythonPrinter.visit(self, node, ctx)

    def visit_Var(self, expr: Var, ctx=None) -> str:
        return self.context.binding_name(expr)

    def visit_Constant(self, expr: Constant, ctx=None) -> str:
        return self.print(expr.value, ctx)

    def visit_Tuple(self, expr: Tuple, ctx=None) -> str:
        if not expr.elements:
            return "()"
        child = TirPrinter(context=self.context, indent=self.indent + "    ")
        elements = "\n".join(f"{child.indent}{child.visit(item, ctx)}," for item in expr.elements)
        return f"(\n{elements}\n{self.indent})"

    def visit_SymbolRef(self, expr: SymbolRef, ctx=None) -> str:
        return ".".join(_binding_name(part) for part in (*expr.nested, expr.name))

    def visit_ShapeOf(self, expr: ShapeOf, ctx=None) -> str:
        return f"shape_of({self.visit(expr.param, ctx)}, axis={expr.axis})"

    def visit_Op(self, expr: Op, ctx=None) -> str:
        name = getattr(getattr(expr, "_op_schema", None), "name", type(expr).__name__.lower())
        return f"T.{name}"

    def visit_program_call(self, expr: Call, ctx=None) -> str:
        target = expr.target
        if isinstance(target, Slice):
            return self._window_subscript(expr, ctx)
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
            return f"{self.visit(expr.args[0])} {scalar_binary[kind]} {self.visit(expr.args[1])}"
        name = (
            getattr(getattr(target, "_op_schema", None), "name", None)
            or re.sub(r"(?<!^)(?=[A-Z])", "_", type(target).__name__).lower()
        )
        args = [self.visit(item) for item in expr.args]
        for param in type(target).params():
            if param.kind == "attribute":
                value = getattr(target, param.name, None)
                if value is not None:
                    with self.type_surface(indent=self.indent + "    "):
                        rendered = (
                            self.layout_surface(value, self.context)
                            if isinstance(value, LayoutBase)
                            else self.print(value, self.context, self.indent + "    ")
                        )
                    args.append(f"{param.name}={rendered}")
        return f"T.{name}({', '.join(args)})"

    def _window_subscript(self, expr: Call, ctx=None) -> str:
        """A window as the subscript it was authored as, which reads back.

        The generic call form names an Op the parser does not bind, so a window
        is written as the slice it is: each axis runs from its start to that
        start plus what the window reaches, with a step where it has one.
        """
        spans = []
        starts = expr.args[1].elements
        for start, size, stride in zip(starts, expr.target.sizes, expr.target.strides):
            low = self.dim_entry(start, ctx) if is_dim_op_call(start) else self.visit(start, ctx)
            high = f"{low} + {size * stride}"
            spans.append(f"{low}:{high}" if stride == 1 else f"{low}:{high}:{stride}")
        return f"{self.visit(expr.args[0], ctx)}[{', '.join(spans)}]"

    def visit_Sequential(self, stmt, ctx=None):
        return [line for child in stmt.body for line in self.visit(child)]

    def visit_LetStmt(self, stmt, ctx=None):
        rendered = self.visit(stmt.value)
        line = f"{self.indent}{self.visit(stmt.var)} = {rendered}"
        if len(line) + 4 <= _LINE_LENGTH or not isinstance(stmt.value, Call):
            return [line] + self.visit(stmt.body)
        if isinstance(stmt.value.target, AllocTensor):
            lines = self._wrapped_alloc(stmt)
        else:
            lines = self._wrapped_call(f"{self.visit(stmt.var)} = ", rendered)
        return lines + self.visit(stmt.body)

    def _wrapped_call(self, prefix: str, rendered: str) -> list[str]:
        head, separator, arguments = rendered.partition("(")
        if not separator or not arguments.endswith(")"):
            return [f"{self.indent}{prefix}{rendered}"]
        content = arguments[:-1]
        continuation = f"{self.indent}    {content}"
        if len(continuation) + 4 <= _LINE_LENGTH:
            middle = [continuation]
        else:
            middle = [f"{self.indent}    {part}," for part in _split_top_level(content)]
        return [f"{self.indent}{prefix}{head}(", *middle, f"{self.indent})"]

    def _wrapped_alloc(self, stmt) -> list[str]:
        target = stmt.value.target
        tensor = self.print(target.tensor_type, self.context, self.indent + "    ")
        argument = f"tensor_type={tensor}"
        continuation = f"{self.indent}    {argument}"
        if len(continuation) + 4 <= _LINE_LENGTH:
            middle = [continuation]
        else:
            assert tensor.startswith("Tensor[") and tensor.endswith("]")
            fields = _split_top_level(tensor[len("Tensor[") : -1])
            packed = ", ".join(fields)
            field_indent = f"{self.indent}        "
            if len(field_indent + packed) + 4 <= _LINE_LENGTH:
                field_lines = [field_indent + packed]
            else:
                field_lines = [field_indent + field + "," for field in fields]
            middle = [
                f"{self.indent}    tensor_type=Tensor[",
                *field_lines,
                f"{self.indent}    ]",
            ]
        return [
            f"{self.indent}{self.visit(stmt.var)} = T.alloc_tensor(",
            *middle,
            f"{self.indent})",
        ]

    def visit_Evaluate(self, stmt, ctx=None):
        return self._emit_evaluate(stmt)

    def _body(self, body):
        indent = self.indent + "    "
        lines = TirPrinter(context=self.context, indent=indent).visit(body)
        return lines or [f"{indent}pass"]

    def visit_MeshScope(self, stmt, ctx=None):
        rendered = self.mesh_context(stmt.mesh, self.context)
        line = f"{self.indent}with {rendered} as {self.visit(stmt.binding)}:"
        if len(line) + 4 <= _LINE_LENGTH:
            lines = [line]
        else:
            content = rendered.removeprefix("Mesh(").removesuffix(")")
            lines = [
                f"{self.indent}with Mesh(",
                f"{self.indent}    {content}",
                f"{self.indent}) as {self.visit(stmt.binding)}:",
            ]
        self.context.push_mesh(stmt.mesh, self.visit(stmt.binding))
        lines.extend(self._body(stmt.body))
        self.context.pop_mesh()
        return lines

    def visit_For(self, stmt, ctx=None):
        """A loop, its bounds read through the scopes the loop stands in.

        A bound may name a coordinate of an enclosing mesh -- a CTA grid deals
        each CTA its own tiles -- and which scope binds that mesh is what the
        print context holds, so the bounds are rendered through it.
        """
        bounds = ", ".join(
            self.visit(bound, self.context) for bound in (stmt.start, stmt.stop, stmt.step)
        )
        lines = [f"{self.indent}for {self.visit(stmt.induction_var)} in range({bounds}):"]
        lines.extend(self._body(stmt.body))
        return lines

    def visit_If(self, stmt, ctx=None):
        lines = [f"{self.indent}if {self.visit(stmt.cond)}:"]
        lines.extend(self._body(stmt.then_body))
        if stmt.else_body.body:
            lines.append(f"{self.indent}else:")
            lines.extend(self._body(stmt.else_body))
        return lines

    def visit_While(self, stmt, ctx=None):
        return [f"{self.indent}while {self.visit(stmt.cond)}:"] + self._body(stmt.body)

    def visit_Return(self, stmt, ctx=None):
        return [f"{self.indent}return"]

    def _join_args(self, args):
        return ", ".join(self.visit(arg) for arg in args)

    def _emit_evaluate(self, stmt):
        target = stmt.callable
        if isinstance(target, (PrimFunction, ModuleList, Call)):
            frame = getattr(self.context, "frame", None)
            frame = self.callee_frame(Call(target, stmt.args, type=UnitType()), frame) if frame is not None else None
            callee = target if frame is None else frame.function
            head = callee.name if isinstance(target, PrimFunction) else (
                target.key if isinstance(target, ModuleList) else target.args[0].key
            )
            if isinstance(target, Call):
                head += f"[{self.visit(target.args[1])}]"
            return [f"{self.indent}{head}({', '.join(self.visit(arg) for arg in stmt.args)})"]
        handler = _STMT_PRINTERS.get(type(stmt.callable)) or (
            _STMT_PRINTERS.get(Op) if isinstance(stmt.callable, Op) else None
        )
        if handler is None:
            raise NotImplementedError(
                f"TIR printer has no emitter for {type(stmt.callable).__name__}"
            )
        return handler(stmt, self)


_STMT_PRINTERS: dict[type, object] = {}


def register_tir_printer(node_type: type):
    """Register the source emitter for one TIR callable/statement type."""

    def decorate(fn):
        _STMT_PRINTERS[node_type] = fn
        return fn

    return decorate


def _binding_name(name: str) -> str:
    return TirPrintContext.identifier(name)


@register_tir_printer(Launch)
def _print_launch(stmt: Evaluate, printer: TirPrinter) -> list[str]:
    indent = printer.indent
    callee, grid = stmt.args[0], stmt.args[1:4]
    block = stmt.args[4:7]
    forwarded = stmt.args[7:]
    return [
        f"{indent}launch({printer.visit(callee)}, {printer._join_args(forwarded)}, grid={printer.visit(Tuple(type=grid[0].type, elements=tuple(grid)))}, block={printer.visit(Tuple(type=block[0].type, elements=tuple(block)))})  # noqa: F821"
    ]


@register_tir_printer(SymbolRef)
def _print_symbol_call(stmt: Evaluate, printer: TirPrinter) -> list[str]:
    rendered = f"{printer.visit(stmt.callable)}({printer._join_args(stmt.args)})"
    if len(printer.indent + rendered) + 4 <= _LINE_LENGTH:
        return [printer.indent + rendered]
    return printer._wrapped_call("", rendered)


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
        rendered = printer.print(value, printer.context, printer.indent + "    ")
        attrs.append(rendered if op_name == "sync" and p.name == "mesh" else f"{p.name}={rendered}")
    rendered_args = [printer.visit(arg) for arg in args]
    head = printer.visit_Op(target)
    arguments = rendered_args + attrs
    line = f"{indent}{head}({', '.join(arguments)})"
    if len(line) + 4 <= _LINE_LENGTH:
        return [line]
    return [
        f"{indent}{head}(",
        *(f"{indent}    {argument}," for argument in arguments),
        f"{indent})",
    ]


class _BindingCollector(StmtVisitor):
    def __init__(self, params):
        self.bindings = list(params)

    def generic_visit(self, stmt):
        binding = (
            stmt.var if isinstance(stmt, LetStmt) else
            stmt.binding if isinstance(stmt, MeshScope) else
            stmt.induction_var if isinstance(stmt, For) else None
        )
        if binding is not None:
            self.bindings.append(binding)
        super().generic_visit(stmt)


def _definition_lines(fn: PrimFunction, name: str, ctx: TirPrintContext) -> list[str]:
    collector = _BindingCollector(fn.params)
    collector.visit(fn.body)
    with ctx.definition(collector.bindings):
        printer = TirPrinter(context=ctx)
        rendered = []
        for param in fn.params:
            type_ = printer.visit(param.type, ctx) if isinstance(param.type, (TensorType, PointerType)) else repr(param.type)
            if param.is_const and isinstance(param.type, TensorType):
                type_ = type_.replace("Tensor[", "ConstTensor[", 1)
            rendered.append(f"{printer.visit(param)}: {type_}")
        params = ", ".join(rendered)
        definition = f"def {name}({params}):"
        lines = (
            [definition] if len(definition) + 4 <= _LINE_LENGTH else
            [f"def {name}(", f"    {params}", "):"]
        )
        body = TirPrinter(context=ctx, indent="    ").visit(fn.body)
        lines.extend(body or ["    pass"])
        return lines


def _function_block(fn: PrimFunction, context: TirPrintContext | None = None, *, frame=None) -> list[str]:
    ctx = context or TirPrintContext()
    ctx.frame = frame
    target = ctx.use(fn.target.to_python())
    dim_vars = {
        d.name: d
        for p in fn.params
        if isinstance(p.type, TensorType)
        for d in p.type.shape
        if hasattr(d, "name")
    }
    lines = [f'_{d.name} = DimVar("{d.name}", {d.lo}, {d.hi})' for d in dim_vars.values()]
    attrs = [f"target={target}"]
    if fn.output_count != 1:
        attrs.append(f"output_count={fn.output_count}")
    lines.append(f"@prim_func({', '.join(attrs)})")
    lines.extend(_definition_lines(fn, _binding_name(fn.name), ctx))
    for variant in fn.variants:
        pat = variant.specializations[0]
        lines.append("")
        lines.append(
            f"@{_binding_name(fn.name)}.specialize({TirPrinter(context=ctx).render_pattern(pat, ctx)})"
        )
        name = _binding_name(getattr(variant, "_display_name", variant.name))
        lines.extend(_definition_lines(variant, name, ctx))
    return _RenderedLines(lines, ctx.imports)


def tir_function_to_python(fn: PrimFunction, *, options=None) -> str:
    lines = _function_block(fn)
    ctx = TirPrintContext()
    ctx.imports.update(lines.imports)
    lines = [*ctx.header(), "", *lines]
    comments = _function_comments(fn, options)
    if comments:
        lines = [*comments, "", *lines]
    return "\n".join(lines) + "\n"


def _function_comments(fn: PrimFunction, options) -> list[str]:
    """Render requested function metadata using the analysis report machinery.

    Import after initialization: analysis_report imports HirPrinter, whose
    module imports this TIR printer.
    """
    if options is None:
        return []
    from tilefoundry.analysis.metadata import RegionMemoryMetadata  # noqa: PLC0415
    from tilefoundry.inspection.analysis_report import (  # noqa: PLC0415
        memory_summary,
        render_summary,
    )

    records = []
    for metadata_type in options.comment_metadata_types:
        record = get_metadata(fn, metadata_type)
        if record is not None:
            records.extend(
                memory_summary(record) if isinstance(record, RegionMemoryMetadata) else (record,)
            )
    return render_summary(records, opt_in=options.comment_opt_in).splitlines()


def _single_module_to_python(mod: Module, module_name: str, names, *, options=None) -> str:
    name = module_name or mod.name
    lines: list[str] = []
    ctx = TirPrintContext()
    kwargs = []
    if mod.entry is not None:
        kwargs.append(f'entry="{_binding_name(mod.entry)}"')
    if mod.target is not None:
        target = mod.target.to_python()
        ctx.imports.update(target.imports)
        kwargs.append(f"target={target.text}")
    if mod.topologies is not None:
        rendered = ", ".join(f'Topology("{t.name}", {t.size!r})' for t in mod.topologies)
        kwargs.append(f"topologies=({rendered},)" if rendered else "topologies=()")
    decorator = f"@module({', '.join(kwargs)})"
    if len(decorator) > _LINE_LENGTH:
        packed = f"    {', '.join(kwargs)}"
        if len(packed) <= _LINE_LENGTH:
            lines.extend(("@module(", packed, ")"))
        else:
            lines.extend(("@module(", *(f"    {item}," for item in kwargs), ")"))
    else:
        lines.append(decorator)
    lines.append(f"class {name}:")
    blocks: list[list[str]] = []
    from .python_printer import _slot_binding  # noqa: PLC0415 -- printer cycle

    bindings = [_slot_binding(key, slot, names) for key, slot in mod.children.items()]
    if bindings:
        blocks.append(bindings)
    for fn in mod.functions:
        if not isinstance(fn, PrimFunction):
            if isinstance(fn, HirFunction):
                raise NotImplementedError("mixed HIR/TIR module printing is not yet supported")
            raise TypeError(f"TIR printer cannot serialize {type(fn).__name__}")
        block = _function_block(fn, frame=CallFrame(mod, fn))
        ctx.imports.update(block.imports)
        blocks.append([*_function_comments(fn, options), *block])
    for index, block in enumerate(blocks):
        if index:
            lines.append("")
        lines.extend(("    " + line) if line else "" for line in block if " = DimVar(" not in line)
    declarations = list(
        dict.fromkeys(line for block in blocks for line in block if " = DimVar(" in line)
    )
    if declarations:
        remaining = [line for line in lines if line not in declarations]
        while remaining and not remaining[0]:
            remaining.pop(0)
        lines = declarations + ["", ""] + remaining
    header = ctx.header()
    if not declarations:
        header.append("")
    return "\n".join(header + lines) + "\n"


def tir_module_to_python(mod: Module, module_name: str | None = None, *, options=None) -> str:
    """Print shared TIR definitions once, with slots referring to those definitions."""
    from .python_printer import (  # noqa: PLC0415 -- printer cycle
        _definition_names,
        _module_definition_order,
    )

    names = _definition_names(mod, module_name or mod.name)
    ctx = TirPrintContext()
    declarations = []
    lines = []
    for node in _module_definition_order(mod):
        source = _single_module_to_python(
            node, names[id(node)], names, options=options
        ).splitlines()
        start = next(index for index, line in enumerate(source) if line.startswith("@module"))
        for line in source[:start]:
            if line.startswith(("from ", "import ")) and line not in (
                "from __future__ import annotations",
                "from tilefoundry.dsl import *",
            ):
                ctx.imports.add(line)
            elif " = DimVar(" in line and line not in declarations:
                declarations.append(line)
        if lines:
            lines.append("")
        lines.extend(source[start:])
        if node is not mod and names[id(node)] != node.name:
            lines.append(f"{names[id(node)]}.name = {node.name!r}")
    if mod.children:
        lines.extend(
            (
                "",
                f"del {', '.join(names[id(node)] for node in _module_definition_order(mod) if node is not mod)}",
            )
        )
    header = ctx.header()
    if declarations:
        header.extend([*declarations, ""])
    header.append("")
    return "\n".join([*header, *lines]) + "\n"
