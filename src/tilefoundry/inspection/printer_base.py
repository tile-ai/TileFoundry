"""Common expression-printer base classes.

Lazy imports avoid a cycle between the shared base and the legacy HIR module.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import enum

from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout, LayoutBase
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprFunctor
from tilefoundry.utils.python_source import PythonExpr


class PythonPrinter(ExprFunctor[str]):
    """Shared expression/value visitor base for HIR and TIR printers."""

    def dim_entry(self, value, ctx=None) -> str:
        return str(value)

    def shard_surface(self, value, ctx=None):
        return None

    def atom_reference(self, value, ctx=None) -> str:
        return f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"

    def render_value(self, value, ctx=None, indent: str = "") -> str:
        """Render a DSL value and register every import needed by it."""
        if isinstance(value, Mesh) and ctx is not None:
            alias = getattr(ctx, "mesh_alias", lambda _m: None)(value)
            if alias is not None:
                return alias
        if isinstance(value, TensorType):
            return self.render_tensor_type(value, ctx, indent)
        if isinstance(value, Mesh):
            if ctx is not None and hasattr(ctx, "_mesh_aliases"):
                return ctx.use(value.to_python())
            return self.render_mesh(value, ctx, indent)
        if isinstance(value, LayoutBase):
            if ctx is not None and hasattr(ctx, "_mesh_aliases"):
                return ctx.use(value.to_python())
            return self.render_layout(value, ctx, indent)
        if isinstance(value, MmaAtom):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.dsl import T",), "T"))
            return self.atom_reference(value, ctx)
        if isinstance(value, enum.Enum):
            if ctx is not None:
                ctx.use(PythonExpr((f"from {type(value).__module__} import {type(value).__name__}",), ""))
            return f"{type(value).__name__}.{value.name}"
        rendered = getattr(value, "to_python", None)
        if callable(rendered):
            expr = rendered()
            return ctx.use(expr) if ctx is not None else expr.text
        if isinstance(value, DType):
            return repr(value.name)
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
            strides = self.shape_tuple(layout.strides, ctx) if layout.strides is not None else "None"
            return f"Layout({self.shape_tuple(layout.shape, ctx)}, {strides})"
        if isinstance(layout, ShardLayout):
            return self.render_shard_layout(layout, ctx, indent)
        if isinstance(layout, ComposedLayout):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ComposedLayout",), ""))
            child = indent + "    "
            return (
                "ComposedLayout(\n"
                f"{child}inner={self.render_layout(layout.inner, ctx, child)},\n"
                f"{child}offset={self.dim_entry(layout.offset, ctx)},\n"
                f"{child}outer={self.render_layout(layout.outer, ctx, child)},\n"
                f"{indent})"
            )
        raise TypeError(f"unsupported layout type: {type(layout).__name__}")

    def render_mesh(self, mesh: Mesh, ctx=None, indent: str = "") -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Layout, Mesh, Topology",), ""))
        values = ", ".join(
            f'Topology("{topology.name}", {self.dim_entry(topology.size, ctx)})'
            for topology in mesh.topologies
        )
        topologies = f"({values}{',' if len(mesh.topologies) == 1 else ''})"
        result = f"Mesh({topologies}, {self.render_layout(mesh.layout, ctx, indent)}"
        if mesh.names:
            result += f", names={tuple(mesh.names)!r}"
        return result + ")"

    def render_shard_layout(self, layout: ShardLayout, ctx=None, indent: str = "") -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ShardLayout",), ""))
        child = indent + "    "
        attrs = ", ".join(self._shard_attr_str(attr, ctx) for attr in layout.attrs)
        if len(layout.attrs) == 1:
            attrs += ","
        return (
            "ShardLayout(\n"
            f"{child}layout={self.render_layout(layout.layout, ctx, child)},\n"
            f"{child}attrs=({attrs}),\n"
            f"{child}mesh={self.render_value(layout.mesh, ctx, child) if ctx is not None and hasattr(ctx, '_mesh_aliases') else self.render_mesh(layout.mesh, ctx, child)},\n"
            f"{indent})"
        )

    def render_tensor_type(self, ty: TensorType, ctx=None, indent: str = "", is_const=False) -> str:
        head = "ConstTensor" if is_const else "Tensor"
        result = f'{head}[{self.shape_tuple(ty.shape, ctx)}, "{self.dtype_str(ty.dtype, ctx)}"'
        if isinstance(ty.layout, ShardLayout):
            surface = self.shard_surface(ty.layout, ctx)
            if surface is not None:
                result += f", {surface}"
            else:
                result += f",\n{indent}    {self.render_shard_layout(ty.layout, ctx, indent + '    ')}"
        if ty.storage is not StorageKind.GMEM:
            result += f', "{ty.storage.name.lower()}"'
        return result + "]"

    def render_pattern(self, pattern: Pattern, ctx=None) -> str:
        if isinstance(pattern, DimVarRangePat):
            return f'DimVarRangePat("{pattern.dim_var}", {pattern.lo}, {pattern.hi})'
        return repr(pattern)

    def mesh_name_map(self, meshes: dict[int, Mesh]) -> dict[int, str]:
        used: set[str] = set()
        result: dict[int, str] = {}
        signatures: dict[str, str] = {}
        for identity, mesh in meshes.items():
            signature = self.render_mesh(mesh)
            if signature in signatures:
                result[identity] = signatures[signature]
                continue
            base = mesh.topologies[0].name if mesh.topologies else "mesh"
            name, suffix = base, 2
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name)
            result[identity] = name
            signatures[signature] = name
        return result
