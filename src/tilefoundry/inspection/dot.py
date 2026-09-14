"""Serialize SSA HIR functions as Graphviz DOT."""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Var, binding_name
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.visitor import ExprWalker

from .print_context import HirPrintContext
from .printer_base import PythonPrinter


def _op_display_name(target) -> str:
    cls = type(target).__name__
    for suffix in ("Op", "Expr", "Stmt"):
        if cls.endswith(suffix) and cls != suffix:
            cls = cls[: -len(suffix)]
    return cls


def _type_lines(ty, printer: PythonPrinter, ctx: HirPrintContext) -> list[str]:
    if not isinstance(ty, TensorType):
        return [str(ty)]
    with printer.type_surface():
        return printer.visit(ty, ctx).split("\n")


def _escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def hir_function_to_dot(fn: HirFunction) -> str:
    """Convert a HIR function to a DOT digraph using canonical type text."""
    printer = PythonPrinter()
    ctx = HirPrintContext()
    root_scope = fn.body if isinstance(fn.body, MeshRegion) else None
    if root_scope is not None:
        ctx.push_mesh(root_scope.mesh, "mesh")

    lines = [
        f"digraph {fn.name} {{",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fillcolor="#f0f0f0"];',
        '  edge [fontsize=10, fontcolor="#555555"];',
        "",
    ]
    counter = [0]
    ids: dict[int, str] = {}

    def node_id(node):
        key = id(node)
        if key not in ids:
            ids[key] = f"n{counter[0]}"
            counter[0] += 1
        return ids[key]

    def emit_node(nid, label_lines, fill="#f0f0f0"):
        label = "\\n".join(_escape_dot(line) for line in label_lines)
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}"];')

    def emit_edge(src_id, dst_id, label=""):
        suffix = f' [label="{label}"]' if label else ""
        lines.append(f"  {src_id} -> {dst_id}{suffix};")

    var_fill, const_fill = "#d4e6f1", "#f9e79f"
    call_fill, shard_fill = "#d5f5e3", "#e8daef"

    class DotWalker(ExprWalker[None]):
        def _emit_leaf(self, expr, label_lines, fill):
            emit_node(node_id(expr), label_lines, fill=fill)

        def visit_Var(self, expr: Var, current=None) -> None:
            self._emit_leaf(expr, [f"Var: {expr.name}", *_type_lines(expr.type, printer, ctx)], var_fill)

        def visit_Constant(self, expr: Constant, current=None) -> None:
            value = f"{expr.value:.6g}" if isinstance(expr.value, float) else str(expr.value)
            self._emit_leaf(expr, [f"Const: {value}", *_type_lines(expr.type, printer, ctx)], const_fill)

        def visit_Call(self, expr: Call, current=None) -> None:
            nid = node_id(expr)
            target = expr.target
            if isinstance(target, Reshard):
                name = binding_name(expr)
                header = f"{name}\\nReshard" if name else "Reshard"
                emit_node(nid, [header, *_type_lines(expr.type, printer, ctx)], fill=shard_fill)
            else:
                name = binding_name(expr)
                op_label = _op_display_name(target)
                header = f"{name}\\n{op_label}" if name else op_label
                emit_node(nid, [header, *_type_lines(expr.type, printer, ctx)], fill=call_fill)
            for index, arg in enumerate(expr.args):
                self.visit(arg, current)
                emit_edge(node_id(arg), nid, f"arg[{index}]" if len(expr.args) > 1 else "")

        def visit_MeshRegion(self, expr: MeshRegion, current=None) -> None:
            alias = ctx.scope_name(expr.mesh)
            ctx.push_mesh(expr.mesh, alias)
            try:
                self.visit(expr.body, current)
                for arg in expr.args:
                    self.visit(arg, current)
            finally:
                ctx.pop_mesh()

        def visit_Tuple(self, expr, current=None) -> None:
            self._emit_leaf(expr, [type(expr).__name__], "#ffffff")

        def visit_LoopRegion(self, expr, current=None) -> None:
            self._emit_leaf(expr, [type(expr).__name__], "#ffffff")

        def visit_ShapeOf(self, expr, current=None) -> None:
            emit_node(node_id(expr), ["ShapeOf"], fill="#ffffff")

        def default_visit(self, expr, current=None) -> None:
            emit_node(node_id(expr), [type(expr).__name__], fill="#ffffff")

    walker = DotWalker()
    walker.visit(fn.body)
    for param in fn.params:
        walker.visit(param)
    if root_scope is not None:
        ctx.pop_mesh()

    lines.extend([
        "",
        '  subgraph cluster_legend {',
        '    label="Legend";',
        "    style=dashed;",
        '    fontsize=11;',
        '    l_var [label="Var/Param", fillcolor="#d4e6f1", shape=box, style=filled];',
        '    l_const [label="Constant", fillcolor="#f9e79f", shape=box, style=filled];',
        '    l_call [label="Op", fillcolor="#d5f5e3", shape=box, style=filled];',
        '    l_shard [label="Reshard", fillcolor="#e8daef", shape=box, style=filled];',
        "  }",
        "}",
    ])
    return "\n".join(lines) + "\n"


def module_entry_to_dot(module: Module) -> str:
    """Convert a Module's entry function to DOT."""
    return hir_function_to_dot(module.entry_function())
