"""Canonical Python rendering for immutable IR type values.

Expression printers own traversal and statement/function syntax.  This module
owns the value-language shared by those printers so HIR and TIR cannot grow
independent type-formatting implementations.  Each type has exactly one
``visit_<Type>`` implementation and the visitors recurse into each other, so an
expression printer renders a type by entering the same ``visit`` its children
use rather than through a parallel ``render_*`` facade.
"""

from __future__ import annotations

from contextlib import contextmanager

from tilefoundry.ir.types import DType, TensorType, TupleType, UnitType
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
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

    def __init__(self) -> None:
        self._indent = ""
        self._tensor_head = "Tensor"

    @contextmanager
    def type_surface(self, *, indent: str | None = None, const: bool = False):
        """Carry the caller's block indentation and const-ness into ``visit``.

        Neither belongs to a type value: indentation is the statement the type
        is printed inside and const-ness is a parameter's property, so they
        travel as printer state instead of widening every visitor signature.
        """
        previous = (self._indent, self._tensor_head)
        if indent is not None:
            self._indent = indent
        self._tensor_head = "ConstTensor" if const else "Tensor"
        try:
            yield
        finally:
            self._indent, self._tensor_head = previous

    def dim_entry(self, value, ctx=None) -> str:
        return str(value)

    def dtype_str(self, dtype: DType, ctx=None) -> str:
        return dtype.name

    def shape_tuple(self, shape: tuple, ctx=None) -> str:
        values = tuple(self.dim_entry(entry, ctx) for entry in shape)
        return f"({values[0]},)" if len(values) == 1 else "(" + ", ".join(values) + ")"

    def shard_surface(self, value: ShardLayout, ctx=None) -> str | None:
        """Parser sugar for a shard layout, or ``None`` when it has none."""
        from .python_printer import _shard_layout_surface_str  # noqa: PLC0415

        mesh_name = ctx.mesh_alias(value.mesh) if ctx is not None else None
        if mesh_name is None or not value.mesh.names:
            return None
        count = ctx.mesh_count() if ctx is not None and hasattr(ctx, "mesh_count") else 1
        return _shard_layout_surface_str(value, mesh_name=mesh_name, mesh_unique=count == 1)

    def visit_TensorType(self, value: TensorType, ctx=None) -> str:
        result = (
            f"{self._tensor_head}["
            f'{self.shape_tuple(value.shape, ctx)}, "{self.dtype_str(value.dtype, ctx)}"'
        )
        if isinstance(value.layout, ShardLayout):
            surface = self.shard_surface(value.layout, ctx)
            if surface is not None:
                result += f", {surface}"
            else:
                with self.type_surface(indent=self._indent + "    "):
                    result += f",\n{self._indent}{self.visit(value.layout, ctx)}"
        if value.storage is not StorageKind.GMEM:
            result += f', "{value.storage.name.lower()}"'
        return result + "]"

    def visit_TupleType(self, value: TupleType, ctx=None) -> str:
        fields = ", ".join(self.visit(field, ctx) for field in value.fields)
        return f"Tuple[{fields}]"

    def visit_UnitType(self, value: UnitType, ctx=None) -> str:
        return "None"

    def visit_DType(self, value: DType, ctx=None) -> str:
        return self.dtype_str(value, ctx)

    def visit_Mesh(self, value: Mesh, ctx=None) -> str:
        alias = ctx.mesh_alias(value) if ctx is not None else None
        if alias is not None:
            return alias
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Layout, Mesh, Topology",), ""))
        values = ", ".join(
            f'Topology("{topology.name}", {self.dim_entry(topology.size, ctx)})'
            for topology in value.topologies
        )
        topologies = f"({values}{',' if len(value.topologies) == 1 else ''})"
        result = f"Mesh({topologies}, {self.visit(value.layout, ctx)}"
        if value.names:
            result += f", names={tuple(value.names)!r}"
        return result + ")"

    def visit_NoneType(self, value: None, ctx=None) -> str:
        """An absent layout is part of the type language, not a missing case."""
        return "None"

    def visit_Layout(self, value: Layout, ctx=None) -> str:
        strides = (
            self.shape_tuple(value.strides, ctx) if value.strides is not None else "None"
        )
        return f"Layout({self.shape_tuple(value.shape, ctx)}, {strides})"

    def visit_ComposedLayout(self, value: ComposedLayout, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ComposedLayout",), ""))
        outer, child = self._indent, self._indent + "    "
        with self.type_surface(indent=child):
            inner_text = self.visit(value.inner, ctx)
            outer_text = self.visit(value.outer, ctx)
        return (
            "ComposedLayout(\n"
            f"{child}inner={inner_text},\n"
            f"{child}offset={self.dim_entry(value.offset, ctx)},\n"
            f"{child}outer={outer_text},\n"
            f"{outer})"
        )

    def visit_ShardLayout(self, value: ShardLayout, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ShardLayout",), ""))
        outer, child = self._indent, self._indent + "    "
        attrs = ", ".join(self.visit(attr, ctx) for attr in value.attrs)
        if len(value.attrs) == 1:
            attrs += ","
        with self.type_surface(indent=child):
            mesh_text = self.visit(value.mesh, ctx)
            layout_text = self.visit(value.layout, ctx)
        return (
            "ShardLayout(\n"
            f"{child}layout={layout_text},\n"
            f"{child}attrs=({attrs}),\n"
            f"{child}mesh={mesh_text},\n"
            f"{outer})"
        )

    def visit_Broadcast(self, value: Broadcast, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import B",), "B"))
        return "B()"

    def visit_Split(self, value: Split, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import S",), "S"))
        return f"S({value.axis})"

    def visit_Partial(self, value: Partial, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import P",), "P"))
        return f'P("{value.reduction}")'
