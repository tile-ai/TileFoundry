"""Common expression-printer base classes.

Lazy imports avoid a cycle between the shared base and the legacy HIR module.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

from tilefoundry.ir.visitor import ExprFunctor


class PythonPrinter(ExprFunctor[str]):
    """Shared expression/value visitor base for HIR and TIR printers."""

    def render_layout(self, layout, ctx=None) -> str:
        from ._python_render import layout_str
        return layout_str(layout)

    def render_mesh(self, mesh, ctx=None) -> str:
        from ._python_render import mesh_str
        return mesh_str(mesh)

    def render_shard_layout(self, layout, ctx=None) -> str:
        from ._python_render import shard_layout_str
        return shard_layout_str(layout)

    def render_tensor_type(self, ty, ctx=None) -> str:
        from ._python_render import tensor_annotation
        return tensor_annotation(ty)
