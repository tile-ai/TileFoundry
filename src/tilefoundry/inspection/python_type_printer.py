"""Canonical Python rendering for immutable IR type values.

Expression printers own traversal and statement/function syntax.  This module
owns the value-language shared by those printers so HIR and TIR cannot grow
independent type-formatting implementations.
"""

from __future__ import annotations

from typing import Any

from tilefoundry.ir.types import DType, TensorType, TupleType, UnitType
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout, LayoutBase
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import TypeFunctor
from tilefoundry.utils.python_source import PythonExpr


class PythonTypePrinter(TypeFunctor[str]):
    """Render supported IR types/layouts through one Python value surface."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self._indent = ""

    def render(self, value: Any, ctx=None, indent: str = "") -> str:
        """Render a value while carrying the caller's multiline indentation."""
        previous = self._indent
        self._indent = indent
        try:
            return self.visit(value, ctx)
        finally:
            self._indent = previous

    def visit_TensorType(self, value: TensorType, ctx=None) -> str:
        return self.render_tensor_type(value, ctx, self._indent)

    def visit_TupleType(self, value: TupleType, ctx=None) -> str:
        fields = ", ".join(self.visit(field, ctx) for field in value.fields)
        return f"Tuple[{fields}]"

    def visit_UnitType(self, value: UnitType, ctx=None) -> str:
        return "None"

    def visit_DType(self, value: DType, ctx=None) -> str:
        return self.owner.dtype_str(value, ctx)

    def visit_Mesh(self, value: Mesh, ctx=None) -> str:
        alias = ctx.mesh_alias(value) if ctx is not None else None
        if alias is not None:
            return alias
        return self.render_mesh(value, ctx)

    def visit_Layout(self, value: Layout, ctx=None) -> str:
        return self.render_layout(value, ctx)

    def visit_ComposedLayout(self, value: ComposedLayout, ctx=None) -> str:
        return self.render_layout(value, ctx)

    def visit_ShardLayout(self, value: ShardLayout, ctx=None) -> str:
        return self.render_shard_layout(value, ctx)

    def visit_Broadcast(self, value: Broadcast, ctx=None) -> str:
        return self._shard_attr_str(value, ctx)

    def visit_Split(self, value: Split, ctx=None) -> str:
        return self._shard_attr_str(value, ctx)

    def visit_Partial(self, value: Partial, ctx=None) -> str:
        return self._shard_attr_str(value, ctx)

    def render_tensor_type(
        self, ty: TensorType, ctx=None, indent: str = "", is_const: bool = False
    ) -> str:
        head = "ConstTensor" if is_const else "Tensor"
        result = f'{head}[{self.owner.shape_tuple(ty.shape, ctx)}, "{self.owner.dtype_str(ty.dtype, ctx)}"'
        if isinstance(ty.layout, ShardLayout):
            surface = self.owner.shard_surface(ty.layout, ctx)
            if surface is not None:
                result += f", {surface}"
            else:
                result += f",\n{indent}    {self.render_shard_layout(ty.layout, ctx, indent + '    ')}"
        if ty.storage is not StorageKind.GMEM:
            result += f', "{ty.storage.name.lower()}"'
        return result + "]"

    def _shard_attr_str(self, attr, ctx=None) -> str:
        if isinstance(attr, Broadcast):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import B",), "B"))
            return "B()"
        if isinstance(attr, Split):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import S",), "S"))
            return f"S({attr.axis})"
        if isinstance(attr, Partial):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import P",), "P"))
            return f'P("{attr.reduction}")'
        raise TypeError(f"unsupported shard attribute: {type(attr).__name__}")

    def render_layout(self, layout: LayoutBase | None, ctx=None, indent: str = "") -> str:
        if layout is None:
            return "None"
        if isinstance(layout, Layout):
            strides = (
                self.owner.shape_tuple(layout.strides, ctx)
                if layout.strides is not None
                else "None"
            )
            return f"Layout({self.owner.shape_tuple(layout.shape, ctx)}, {strides})"
        if isinstance(layout, ShardLayout):
            return self.render_shard_layout(layout, ctx, indent)
        if isinstance(layout, ComposedLayout):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ComposedLayout",), ""))
            child = indent + "    "
            return (
                "ComposedLayout(\n"
                f"{child}inner={self.render_layout(layout.inner, ctx, child)},\n"
                f"{child}offset={self.owner.dim_entry(layout.offset, ctx)},\n"
                f"{child}outer={self.render_layout(layout.outer, ctx, child)},\n"
                f"{indent})"
            )
        raise TypeError(f"unsupported layout type: {type(layout).__name__}")

    def render_mesh(self, mesh: Mesh, ctx=None, indent: str = "") -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Layout, Mesh, Topology",), ""))
        values = ", ".join(
            f'Topology("{topology.name}", {self.owner.dim_entry(topology.size, ctx)})'
            for topology in mesh.topologies
        )
        topologies = f"({values}{',' if len(mesh.topologies) == 1 else ''})"
        result = f"Mesh({topologies}, {self.render_layout(mesh.layout, ctx, indent)}"
        if mesh.names:
            result += f", names={tuple(mesh.names)!r}"
        return result + ")"

    def render_shard_layout(
        self, layout: ShardLayout, ctx=None, indent: str = "", *, mesh_ref=None
    ) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ShardLayout",), ""))
        child = indent + "    "
        attrs = ", ".join(self._shard_attr_str(attr, ctx) for attr in layout.attrs)
        if len(layout.attrs) == 1:
            attrs += ","
        mesh_text = mesh_ref if mesh_ref is not None else self.render_mesh(layout.mesh, ctx, child)
        layout_text = self.render_layout(layout.layout, ctx, child)
        return (
            "ShardLayout(\n"
            f"{child}layout={layout_text},\n"
            f"{child}attrs=({attrs}),\n"
            f"{child}mesh={mesh_text},\n"
            f"{indent})"
        )

    def _render_layout_positional(self, layout, ctx=None):
        if isinstance(layout, Layout):
            strides = self.owner.shape_tuple(layout.strides, ctx) if layout.strides is not None else "None"
            return f"Layout({self.owner.shape_tuple(layout.shape, ctx)}, {strides})"
        return self.render_layout(layout, ctx)

    def _render_mesh_dataclass(self, mesh, ctx=None):
        if ctx is not None:
            alias = ctx.mesh_alias(mesh)
            if alias is not None:
                return alias
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Mesh, Topology",), ""))
        values = ", ".join(
            f'Topology(name="{topology.name}", size={self.owner.dim_entry(topology.size, ctx)})'
            for topology in mesh.topologies
        )
        if len(mesh.topologies) == 1:
            values += ","
        names = ", ".join(f'"{name}"' for name in mesh.names)
        if len(mesh.names) == 1:
            names += ","
        layout = self.render_layout(mesh.layout, ctx)
        return f"Mesh(topologies=({values}), layout={layout}, names=({names}))"

    def _render_mesh_compact(self, mesh, ctx=None):
        if ctx is not None:
            alias = ctx.mesh_alias(mesh)
            if alias is not None:
                return alias
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Mesh, Topology",), ""))
        values = ", ".join(
            f'Topology("{topology.name}", {self.owner.dim_entry(topology.size, ctx)})'
            for topology in mesh.topologies
        )
        topologies = f"({values}{',' if len(mesh.topologies) == 1 else ''})"
        names = f", names={tuple(mesh.names)!r}" if mesh.names else ""
        return f"Mesh({topologies}, {self._render_layout_positional(mesh.layout, ctx)}{names})"
