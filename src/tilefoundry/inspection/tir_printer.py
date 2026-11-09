"""Canonical Python DSL printer for TIR PrimFunction and Module values."""

from __future__ import annotations

from tilefoundry import dispatch, services
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.ir.expr import op_registry
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.tir import memory

from .module_names import definition_names, module_definition_order, slot_binding
from .print_context import TirPrintContext, identifier
from .printer_base import PythonPrinter

LINE_LENGTH = 100


def split_top_level(text: str) -> list[str]:
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


class RenderedLines(list):
    def __init__(self, lines, imports):
        super().__init__(lines)
        self.imports = imports


class TirPrinter(PythonPrinter, core.StmtVisitor[list[str]], core.ExprVisitor[str]):
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

    def visit(self, node, ctx=None):
        """Dispatch statements, expressions, and types by their concrete name."""
        return PythonPrinter.visit(self, node, ctx)

    def visit_Var(self, node: expr.Var, ctx=None) -> str:
        return self.context.binding_name(node)

    def visit_Constant(self, node: expr.Constant, ctx=None) -> str:
        return self.print(node.value, ctx if ctx is not None else self.context)

    def visit_Tuple(self, node: expr.Tuple, ctx=None) -> str:
        if not node.elements:
            return "()"
        child = TirPrinter(context=self.context, indent=self.indent + "    ")
        elements = "\n".join(f"{child.indent}{child.visit(item, ctx)}," for item in node.elements)
        return f"(\n{elements}\n{self.indent})"

    def visit_SymbolRef(self, node: core.SymbolRef, ctx=None) -> str:
        return ".".join(identifier(part) for part in (*node.nested, node.name))

    def visit_ShapeOf(self, node: core.ShapeOf, ctx=None) -> str:
        return f"shape_of({self.visit(node.param, ctx)}, axis={node.axis})"

    def visit_Op(self, node: expr.Op, ctx=None) -> str:
        return expr.get_registered_name(node)

    def visit_program_call(self, node: expr.Call, ctx=None) -> str:
        target = node.target
        if isinstance(target, tensor.Slice):
            return self.window_subscript(node, ctx)
        scalar_binary = {
            expr.BinaryKind.EQ: "==",
            expr.BinaryKind.NE: "!=",
            expr.BinaryKind.LT: "<",
            expr.BinaryKind.LE: "<=",
            expr.BinaryKind.GT: ">",
            expr.BinaryKind.GE: ">=",
            expr.BinaryKind.AND: "and",
        }
        kind = getattr(target, "kind", None)
        if kind in scalar_binary and len(node.args) == 2 and node.type.dtype is expr.DType.bool:
            return f"{self.visit(node.args[0])} {scalar_binary[kind]} {self.visit(node.args[1])}"
        name = expr.get_registered_name(target)
        dialect, *path, leaf = name.split(".")
        selected = op_registry.lookup(dialect, tuple(path), leaf)
        args = [self.visit(item) for item in node.args]
        for param in expr.parameters_of(type(target)):
            if param.kind == "attribute" and param.name not in selected.fixed_attrs:
                value = getattr(target, param.name, None)
                if value is not None:
                    with self.type_surface(indent=self.indent + "    "):
                        rendered = (
                            self.layout_surface(
                                value, self.context, storage=getattr(node.type, "storage", None)
                            )
                            if isinstance(value, types.LayoutBase)
                            else self.print(value, self.context, self.indent + "    ")
                        )
                    args.append(f"{param.name}={rendered}")
        return f"{name}({', '.join(args)})"

    def window_subscript(self, node: expr.Call, ctx=None) -> str:
        """A window as the subscript it was authored as, which reads back.

        The generic call form names an Op the parser does not bind, so a window
        is written as the slice it is: each axis runs from its start to that
        start plus what the window reaches, with a step where it has one.
        """
        spans = []
        starts = node.args[1].elements
        for start, size, stride in zip(starts, node.target.sizes, node.target.strides):
            low = self.dim_entry(start, ctx) if expr.is_dim_op_call(start) else self.visit(start, ctx)
            high = f"{low} + {size * stride}"
            spans.append(f"{low}:{high}" if stride == 1 else f"{low}:{high}:{stride}")
        return f"{self.visit(node.args[0], ctx)}[{', '.join(spans)}]"

    def visit_Sequential(self, stmt, ctx=None):
        return [line for child in stmt.body for line in self.visit(child)]

    def visit_LetStmt(self, stmt, ctx=None):
        rendered = self.visit(stmt.value)
        line = f"{self.indent}{self.visit(stmt.var)} = {rendered}"
        if len(line) + 4 <= LINE_LENGTH or not isinstance(stmt.value, expr.Call):
            return [line]
        if isinstance(stmt.value.target, memory.AllocTensor):
            lines = self.wrapped_alloc(stmt)
        else:
            lines = self.wrapped_call(f"{self.visit(stmt.var)} = ", rendered)
        return lines

    def wrapped_call(self, prefix: str, rendered: str) -> list[str]:
        head, separator, arguments = rendered.partition("(")
        if not separator or not arguments.endswith(")"):
            return [f"{self.indent}{prefix}{rendered}"]
        content = arguments[:-1]
        continuation = f"{self.indent}    {content}"
        if len(continuation) + 4 <= LINE_LENGTH:
            middle = [continuation]
        else:
            middle = [f"{self.indent}    {part}," for part in split_top_level(content)]
        return [f"{self.indent}{prefix}{head}(", *middle, f"{self.indent})"]

    def wrapped_alloc(self, stmt) -> list[str]:
        target = stmt.value.target
        tensor_value = self.print(target.tensor_type, self.context, self.indent + "    ")
        argument = f"tensor_type={tensor_value}"
        continuation = f"{self.indent}    {argument}"
        if len(continuation) + 4 <= LINE_LENGTH:
            middle = [continuation]
        else:
            assert tensor_value.startswith("Tensor[") and tensor_value.endswith("]")
            fields = split_top_level(tensor_value[len("Tensor[") : -1])
            packed = ", ".join(fields)
            field_indent = f"{self.indent}        "
            if len(field_indent + packed) + 4 <= LINE_LENGTH:
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
        return self.emit_evaluate(stmt)

    def block_body(self, body):
        indent = self.indent + "    "
        lines = TirPrinter(context=self.context, indent=indent).visit(body)
        return lines or [f"{indent}pass"]

    def visit_MeshScope(self, stmt, ctx=None):
        rendered = self.mesh_context(stmt.mesh, self.context)
        line = f"{self.indent}with {rendered} as {self.visit(stmt.binding)}:"
        if len(line) + 4 <= LINE_LENGTH:
            lines = [line]
        else:
            content = rendered.removeprefix("Mesh(").removesuffix(")")
            lines = [
                f"{self.indent}with Mesh(",
                f"{self.indent}    {content}",
                f"{self.indent}) as {self.visit(stmt.binding)}:",
            ]
        self.context.push_mesh(stmt.mesh, self.visit(stmt.binding))
        lines.extend(self.block_body(stmt.body))
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
        lines.extend(self.block_body(stmt.body))
        return lines

    def visit_If(self, stmt, ctx=None):
        lines = [f"{self.indent}if {self.visit(stmt.cond)}:"]
        lines.extend(self.block_body(stmt.then_body))
        if stmt.else_body.body:
            lines.append(f"{self.indent}else:")
            lines.extend(self.block_body(stmt.else_body))
        return lines

    def visit_While(self, stmt, ctx=None):
        return [f"{self.indent}while {self.visit(stmt.cond)}:"] + self.block_body(stmt.body)

    def visit_Return(self, stmt, ctx=None):
        return [f"{self.indent}return"]

    def join_args(self, args):
        return ", ".join(self.visit(arg) for arg in args)

    def emit_evaluate(self, stmt):
        target = stmt.callable
        if isinstance(target, (core.PrimFunction, core.ModuleList, expr.Call)):
            frame = self.context.valuation
            frame = (
                frame.callee_context(expr.Call(target, stmt.args, type=types.UnitType()))
                if frame is not None
                else None
            )
            callee = target if frame is None else frame.function
            head = (
                callee.name
                if isinstance(target, core.PrimFunction)
                else (target.key if isinstance(target, core.ModuleList) else target.args[0].key)
            )
            if isinstance(target, expr.Call):
                head += f"[{self.visit(target.args[1])}]"
            return [f"{self.indent}{head}({', '.join(self.visit(arg) for arg in stmt.args)})"]
        handler = registry.get(type(stmt.callable), None) or (
            registry.get(expr.Op, None) if isinstance(stmt.callable, expr.Op) else None
        )
        if handler is None:
            raise NotImplementedError(
                f"TIR printer has no emitter for {type(stmt.callable).__name__}"
            )
        return handler(stmt, self)


registry: dispatch.Registry[type, object] = dispatch.Registry("tir_printer")
register_tir_printer = registry.register_for


@register_tir_printer(tir.launch.Launch)
def print_launch(stmt: core.Evaluate, printer: TirPrinter) -> list[str]:
    indent = printer.indent
    callee, grid = stmt.args[0], stmt.args[1:4]
    block = stmt.args[4:7]
    forwarded = stmt.args[7:]
    options = "".join(
        f", {param.name}={printer.print(getattr(stmt.callable, param.name), printer.context)}"
        for param in expr.collect_param_defs(tir.launch.Launch)
        if getattr(stmt.callable, param.name) != param.default
    )
    return [
        f"{indent}launch({printer.visit(callee)}, {printer.join_args(forwarded)}, grid={printer.visit(expr.Tuple(type=grid[0].type, elements=tuple(grid)))}, block={printer.visit(expr.Tuple(type=block[0].type, elements=tuple(block)))}{options})  # noqa: F821"
    ]


@register_tir_printer(core.SymbolRef)
def print_symbol_call(stmt: core.Evaluate, printer: TirPrinter) -> list[str]:
    rendered = f"{printer.visit(stmt.callable)}({printer.join_args(stmt.args)})"
    if len(printer.indent + rendered) + 4 <= LINE_LENGTH:
        return [printer.indent + rendered]
    return printer.wrapped_call("", rendered)


@register_tir_printer(expr.Op)
def print_op_evaluate(stmt: core.Evaluate, printer: TirPrinter) -> list[str]:
    indent = printer.indent
    target = stmt.callable
    args = list(stmt.args)
    attrs = []
    name = expr.get_registered_name(target)
    dialect, *path, leaf = name.split(".")
    selected = op_registry.lookup(dialect, tuple(path), leaf)
    for p in expr.parameters_of(type(target)):
        if p.kind != "attribute" or p.name in selected.fixed_attrs:
            continue
        value = getattr(target, p.name, None)
        if value is None:
            continue
        rendered = printer.print(value, printer.context, printer.indent + "    ")
        attrs.append(rendered if name == "T.sync" and p.name == "mesh" else f"{p.name}={rendered}")
    rendered_args = [printer.visit(arg) for arg in args]
    head = printer.visit_Op(target)
    arguments = rendered_args + attrs
    line = f"{indent}{head}({', '.join(arguments)})"
    if len(line) + 4 <= LINE_LENGTH:
        return [line]
    return [
        f"{indent}{head}(",
        *(f"{indent}    {argument}," for argument in arguments),
        f"{indent})",
    ]


class BindingCollector(core.StmtVisitor):
    def __init__(self, params):
        self.bindings = list(params)

    def generic_visit(self, stmt):
        binding = (
            stmt.var
            if isinstance(stmt, core.LetStmt)
            else stmt.binding
            if isinstance(stmt, core.MeshScope)
            else stmt.induction_var
            if isinstance(stmt, core.For)
            else None
        )
        if binding is not None:
            self.bindings.append(binding)
        super().generic_visit(stmt)


def definition_lines(fn: core.PrimFunction, name: str, ctx: TirPrintContext) -> list[str]:
    collector = BindingCollector(fn.params)
    collector.visit(fn.body)
    with ctx.definition(collector.bindings):
        printer = TirPrinter(context=ctx)
        rendered = []
        for param in fn.params:
            type_ = (
                printer.visit(param.annotation, ctx)
                if isinstance(param.annotation, (types.TensorType, types.PointerType))
                else repr(param.annotation)
            )
            if param.is_const and isinstance(param.type, types.TensorType):
                type_ = type_.replace("Tensor[", "ConstTensor[", 1)
            rendered.append(f"{printer.visit(param)}: {type_}")
        params = ", ".join(rendered)
        definition = f"def {name}({params}):"
        lines = (
            [definition]
            if len(definition) + 4 <= LINE_LENGTH
            else [f"def {name}(", f"    {params}", "):"]
        )
        body = TirPrinter(context=ctx, indent="    ").visit(fn.body)
        lines.extend(body or ["    pass"])
        return lines


def function_block(
    fn: core.PrimFunction, context: TirPrintContext | None = None, *, frame=None
) -> list[str]:
    ctx = context or TirPrintContext()
    ctx.bind_function(frame, fn.target)
    target = ctx.use(fn.target.to_python())
    dim_vars = {
        d.name: d
        for p in fn.params
        if isinstance(p.type, types.TensorType)
        for d in p.type.shape
        if hasattr(d, "name")
    }
    lines = [f'_{d.name} = DimVar("{d.name}", {d.lo}, {d.hi})' for d in dim_vars.values()]
    attrs = [f"target={target}"]
    if fn.output_count != 1:
        attrs.append(f"output_count={fn.output_count}")
    lines.append(f"@prim_func({', '.join(attrs)})")
    lines.extend(definition_lines(fn, identifier(fn.name), ctx))
    for variant in fn.variants:
        pat = variant.specializations[0]
        lines.append("")
        lines.append(
            f"@{identifier(fn.name)}.specialize({TirPrinter(context=ctx).render_pattern(pat, ctx)})"
        )
        name = identifier(getattr(variant, "display_name", variant.name))
        lines.extend(definition_lines(variant, name, ctx))
    return RenderedLines(lines, ctx.imports)


def tir_function_to_python(fn: core.PrimFunction) -> str:
    lines = function_block(fn)
    ctx = TirPrintContext()
    ctx.imports.update(lines.imports)
    lines = [*ctx.header(), "", *lines]
    return "\n".join(lines) + "\n"


def single_module_to_python(mod: core.Module, module_name: str, names) -> str:
    name = module_name or mod.name
    lines: list[str] = []
    ctx = TirPrintContext()
    kwargs = []
    if mod.entry is not None:
        kwargs.append(f'entry="{identifier(mod.entry)}"')
    if mod.target is not None:
        target = mod.target.to_python()
        ctx.imports.update(target.imports)
        kwargs.append(f"target={target.text}")
    if mod.topologies is not None:
        rendered = ", ".join(f'Topology("{t.name}", {t.size!r})' for t in mod.topologies)
        kwargs.append(f"topologies=({rendered},)" if rendered else "topologies=()")
    decorator = f"@module({', '.join(kwargs)})"
    if len(decorator) > LINE_LENGTH:
        packed = f"    {', '.join(kwargs)}"
        if len(packed) <= LINE_LENGTH:
            lines.extend(("@module(", packed, ")"))
        else:
            lines.extend(("@module(", *(f"    {item}," for item in kwargs), ")"))
    else:
        lines.append(decorator)
    lines.append(f"class {name}:")
    blocks: list[list[str]] = []
    bindings = [slot_binding(key, slot, names) for key, slot in mod.children.items()]
    if bindings:
        blocks.append(bindings)
    for fn in mod.functions:
        if not isinstance(fn, core.PrimFunction):
            if isinstance(fn, core.Function):
                raise NotImplementedError("mixed HIR/TIR module printing is not yet supported")
            raise TypeError(f"TIR printer cannot serialize {type(fn).__name__}")
        block = function_block(fn, frame=valuation.Context(mod, fn))
        ctx.imports.update(block.imports)
        blocks.append(block)
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


def tir_module_to_python(mod: core.Module, module_name: str | None = None) -> str:
    """Print shared TIR definitions once, with slots referring to those definitions."""
    names = definition_names(mod, module_name or mod.name)
    ctx = TirPrintContext()
    declarations = []
    lines = []
    for node in module_definition_order(mod):
        source = single_module_to_python(node, names[id(node)], names).splitlines()
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
                f"del {', '.join(names[id(node)] for node in module_definition_order(mod) if node is not mod)}",
            )
        )
    header = ctx.header()
    if declarations:
        header.extend([*declarations, ""])
    header.append("")
    return "\n".join([*header, *lines]) + "\n"


def render(node: object) -> str:
    """An IR expression as TIR text."""
    return TirPrinter().visit(node)


services.registry.register(services.render, render)

__all__ = [
    "TirPrinter",
    "function_block",
    "register_tir_printer",
    "registry",
    "render",
    "tir_function_to_python",
    "tir_module_to_python",
]
