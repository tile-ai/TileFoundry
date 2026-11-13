"""Serialize SSA HIR functions as Graphviz DOT."""

from __future__ import annotations

from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import sharding

from .print_context import HirPrintContext
from .printer_base import PythonPrinter


def type_lines(ty, printer: PythonPrinter, ctx: HirPrintContext) -> list[str]:
    if not isinstance(ty, types.TensorType):
        return [str(ty)]
    return printer.print(ty, ctx).split("\n")


def escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def hir_function_to_dot(fn: core.Function) -> str:
    """Convert a HIR function to a DOT digraph using canonical type text."""
    printer = PythonPrinter()
    ctx = HirPrintContext()
    root_scope = fn.body if isinstance(fn.body, core.MeshRegion) else None
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
        label = "\\n".join(escape_dot(line) for line in label_lines)
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}"];')

    def emit_edge(src_id, dst_id, label=""):
        suffix = f' [label="{label}"]' if label else ""
        lines.append(f"  {src_id} -> {dst_id}{suffix};")

    var_fill, const_fill = "#d4e6f1", "#f9e79f"
    call_fill, shard_fill = "#d5f5e3", "#e8daef"

    class DotWalker(core.ExprWalker[None]):
        def emit_leaf(self, node, label_lines, fill):
            emit_node(node_id(node), label_lines, fill=fill)

        def visit_Var(self, node: expr.Var, current=None) -> None:
            self.emit_leaf(node, [f"Var: {node.name}", *type_lines(node.type, printer, ctx)], var_fill)

        def visit_Constant(self, node: expr.Constant, current=None) -> None:
            value = f"{node.value:.6g}" if isinstance(node.value, float) else str(node.value)
            self.emit_leaf(node, [f"Const: {value}", *type_lines(node.type, printer, ctx)], const_fill)

        def visit_Call(self, node: expr.Call, current=None) -> None:
            nid = node_id(node)
            target = node.target
            op_label = target.qualified_name if isinstance(target, core.FunctionRef) else expr.get_registered_name(target)
            name = expr.binding_name(node)
            header = f"{name}\\n{op_label}" if name else op_label
            emit_node(nid, [header, *type_lines(node.type, printer, ctx)],
                      fill=shard_fill if isinstance(target, sharding.Reshard) else call_fill)
            for index, arg in enumerate(node.args):
                self.visit(arg, current)
                emit_edge(node_id(arg), nid, f"arg[{index}]" if len(node.args) > 1 else "")

        def visit_MeshRegion(self, node: core.MeshRegion, current=None) -> None:
            alias = ctx.scope_name(node.mesh)
            ctx.push_mesh(node.mesh, alias)
            try:
                self.visit(node.body, current)
                for arg in node.args:
                    self.visit(arg, current)
            finally:
                ctx.pop_mesh()

        def visit_Tuple(self, node, current=None) -> None:
            self.emit_leaf(node, [type(node).__name__], "#ffffff")

        def visit_LoopRegion(self, node, current=None) -> None:
            self.emit_leaf(node, [type(node).__name__], "#ffffff")

        def visit_ShapeOf(self, node, current=None) -> None:
            emit_node(node_id(node), ["ShapeOf"], fill="#ffffff")

        def default_visit(self, node, current=None) -> None:
            emit_node(node_id(node), [type(node).__name__], fill="#ffffff")

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


def module_entry_to_dot(module: core.Module) -> str:
    """Convert a Module's entry function to DOT."""
    return hir_function_to_dot(module.entry_function())


__all__ = ["hir_function_to_dot", "module_entry_to_dot"]
