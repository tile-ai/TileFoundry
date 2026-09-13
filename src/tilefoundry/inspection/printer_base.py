"""Common expression-printer base classes.

Lazy imports avoid a cycle between the shared base and the legacy HIR module.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import enum

from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import LayoutBase
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.ir.visitor import ExprFunctor
from tilefoundry.target import Target
from tilefoundry.utils.python_source import PythonExpr

from .python_type_printer import PythonTypePrinter


class PythonPrinter(ExprFunctor[str]):
    """Shared expression/value visitor base for HIR and TIR printers."""

    def __init__(self) -> None:
        super().__init__()
        self.type_printer = PythonTypePrinter(self)

    def dim_entry(self, value, ctx=None) -> str:
        return str(value)

    def shard_surface(self, value, ctx=None):
        """Use the HIR sugar classifier through this shared HIR/TIR hook."""
        from .python_printer import _shard_layout_surface_str  # noqa: PLC0415

        mesh_name = ctx.mesh_alias(value.mesh) if ctx is not None else None
        if mesh_name is None or not value.mesh.names:
            return None
        count = ctx.mesh_count() if ctx is not None and hasattr(ctx, "mesh_count") else 1
        return _shard_layout_surface_str(value, mesh_name=mesh_name, mesh_unique=count == 1)

    def atom_reference(self, value, ctx=None) -> str:
        return f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"

    def render_value(self, value, ctx=None, indent: str = "") -> str:
        """Render a DSL value and register every import needed by it."""
        if isinstance(value, (TensorType, Mesh, LayoutBase, DType)):
            return self.type_printer.render(value, ctx, indent)
        if isinstance(value, MmaAtom):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.dsl import T",), "T"))
            return self.atom_reference(value, ctx)
        if isinstance(value, enum.Enum):
            if ctx is not None:
                ctx.use(PythonExpr((f"from {type(value).__module__} import {type(value).__name__}",), ""))
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, Target):
            expr = value.to_python()
            return ctx.use(expr) if ctx is not None else expr.text
        if isinstance(value, DType):
            return self.dtype_str(value, ctx)
        if isinstance(value, (str, int, float, bool, tuple, type(None))):
            if isinstance(value, tuple):
                vals = ", ".join(self.render_value(v, ctx, indent) for v in value)
                return f"({vals}{',' if len(value)==1 else ''})"
            return repr(value)
        raise NotImplementedError(f"no canonical Python form for {type(value).__name__}")

    def dtype_str(self, dtype: DType, ctx=None) -> str:
        return dtype.name

    def shape_tuple(self, shape: tuple, ctx=None) -> str:
        values = tuple(self.dim_entry(entry, ctx) for entry in shape)
        return f"({values[0]},)" if len(values) == 1 else "(" + ", ".join(values) + ")"

    def _shard_attr_str(self, attr, ctx=None) -> str:
        return self.type_printer._shard_attr_str(attr, ctx)

    def render_layout(self, layout: LayoutBase | None, ctx=None, indent: str = "") -> str:
        return self.type_printer.render_layout(layout, ctx, indent)

    def render_mesh(self, mesh: Mesh, ctx=None, indent: str = "") -> str:
        return self.type_printer.render_mesh(mesh, ctx, indent)

    def render_shard_layout(self, layout: ShardLayout, ctx=None, indent: str = "", *, mesh_ref=None) -> str:
        return self.type_printer.render_shard_layout(layout, ctx, indent, mesh_ref=mesh_ref)

    def render_tensor_type(self, ty: TensorType, ctx=None, indent: str = "", is_const=False) -> str:
        return self.type_printer.render_tensor_type(ty, ctx, indent, is_const)

    def _render_layout_positional(self, layout, ctx=None):
        return self.type_printer._render_layout_positional(layout, ctx)

    def _render_mesh_dataclass(self, mesh, ctx=None):
        return self.type_printer._render_mesh_dataclass(mesh, ctx)

    def _render_mesh_compact(self, mesh, ctx=None):
        return self.type_printer._render_mesh_compact(mesh, ctx)

    def render_pattern(self, pattern: Pattern, ctx=None) -> str:
        if isinstance(pattern, DimVarRangePat):
            return f'DimVarRangePat("{pattern.dim_var}", {pattern.lo}, {pattern.hi})'
        return repr(pattern)

    def mesh_name_map(self, meshes: dict[int, Mesh]) -> dict[int, str]:
        used: set[str] = set()
        result: dict[int, str] = {}
        for identity, mesh in meshes.items():
            base = mesh.topologies[0].name if mesh.topologies else "mesh"
            name, suffix = base, 2
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name)
            result[identity] = name
        return result
