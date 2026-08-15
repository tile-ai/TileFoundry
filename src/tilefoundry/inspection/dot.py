"""Serialize SSA HIR functions as Graphviz DOT.

Variables, calls, and constants receive numbered nodes. Type and shard-layout
labels reuse canonical printer renderers so DOT, Python output, and the viewer
remain consistent.
See [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms).
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Var, binding_name
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.visitor import ExprWalker

from .python_printer import _collect_meshes, _mesh_name_map, _op_display_name, _tensor_annotation


def _type_lines(ty, mesh_name_map: dict[int, str]) -> list[str]:
    """Return canonical type-annotation lines for a node label.

    Split multiline shard-layout fallback text into separate label lines.
    See [inspection §2.3](docs/spec/inspection.md#23-dsl-text-forms).
    """
    text = _tensor_annotation(ty, mesh_name_map=mesh_name_map) if isinstance(ty, TensorType) else str(ty)
    return text.split("\n")


def _escape_dot(s: str) -> str:
    """Escape a string for safe inclusion in a DOT label."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def hir_function_to_dot(fn: HirFunction) -> str:
    """Convert a hir.Function to a DOT digraph string.

    Args:
        fn: The HIR function to visualize.

    Returns:
        A Graphviz DOT format string.
    """
    mesh_map = _mesh_name_map(_collect_meshes(fn, include_node_types=True))
    lines = [f"digraph {fn.name} {{", '  rankdir=TB;',
             '  node [shape=box, style=filled, fillcolor="#f0f0f0"];',
             '  edge [fontsize=10, fontcolor="#555555"];', '']
    _counter = [0]
    _ids = {}
    def _id(node):
        key = id(node)
        if key not in _ids:
            _ids[key] = f"n{_counter[0]}"
            _counter[0] += 1
        return _ids[key]

    def _emit_node(nid, label_lines, fill="#f0f0f0"):
        escaped = [_escape_dot(ln) for ln in label_lines]
        label = "\\n".join(escaped)
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}"];')

    def _emit_edge(src_id, dst_id, label=""):
        if label:
            lines.append(f'  {src_id} -> {dst_id} [label="{label}"];')
        else:
            lines.append(f'  {src_id} -> {dst_id};')

    VAR_FILL = "#d4e6f1"
    CONST_FILL = "#f9e79f"
    CALL_FILL = "#d5f5e3"
    SHARDING_FILL = "#e8daef"

    class _DotWalker(ExprWalker[None]):
        def _emit_leaf(self, expr, label_lines, fill) -> None:
            _emit_node(_id(expr), label_lines, fill=fill)

        def visit_Var(self, expr: Var) -> None:
            self._emit_leaf(
                expr,
                [f"Var: {expr.name}", *_type_lines(expr.type, mesh_map)],
                VAR_FILL,
            )

        def visit_Constant(self, expr: Constant) -> None:
            value = f"{expr.value:.6g}" if isinstance(expr.value, float) else str(expr.value)
            self._emit_leaf(
                expr,
                [f"Const: {value}", *_type_lines(expr.type, mesh_map)],
                CONST_FILL,
            )

        def visit_Call(self, expr: Call) -> None:
            nid = _id(expr)
            target = expr.target
            if isinstance(target, Reshard):
                name = binding_name(expr)
                header = f"{name}\\nReshard" if name else "Reshard"
                _emit_node(nid, [header, *_type_lines(expr.type, mesh_map)], fill=SHARDING_FILL)
                for arg in expr.args:
                    self.visit(arg)
                    _emit_edge(_id(arg), nid)
                return

            op_label = _op_display_name(target)
            name = binding_name(expr)
            header = f"{name}\\n{op_label}" if name else op_label
            _emit_node(nid, [header, *_type_lines(expr.type, mesh_map)], fill=CALL_FILL)
            for i, arg in enumerate(expr.args):
                self.visit(arg)
                edge_label = f"arg[{i}]" if len(expr.args) > 1 else ""
                _emit_edge(_id(arg), nid, edge_label)

        def visit_Tuple(self, expr) -> None:
            self._emit_leaf(expr, [type(expr).__name__], "#ffffff")

        def visit_GridRegionExpr(self, expr) -> None:
            self._emit_leaf(expr, [type(expr).__name__], "#ffffff")

        def visit_ShapeOf(self, expr) -> None:
            _emit_node(_id(expr), ["ShapeOf"], fill="#ffffff")

        def default_visit(self, expr) -> None:
            _emit_node(_id(expr), [type(expr).__name__], fill="#ffffff")

    walker = _DotWalker()
    walker.visit(fn.body)
    for param in fn.params:
        walker.visit(param)


    lines.append("")
    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend";')
    lines.append('    style=dashed;')
    lines.append('    fontsize=11;')
    lines.append('    l_var [label="Var/Param", fillcolor="#d4e6f1", shape=box, style=filled];')
    lines.append('    l_const [label="Constant", fillcolor="#f9e79f", shape=box, style=filled];')
    lines.append('    l_call [label="Op", fillcolor="#d5f5e3", shape=box, style=filled];')
    lines.append('    l_shard [label="Reshard", fillcolor="#e8daef", shape=box, style=filled];')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def module_entry_to_dot(module: Module) -> str:
    """Convert a Module's entry function to DOT."""
    fn = module.entry_function()
    return hir_function_to_dot(fn)
