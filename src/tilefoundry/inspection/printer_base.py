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
from tilefoundry.ir.visitor import ExprFunctor
from tilefoundry.target import Target
from tilefoundry.utils.python_source import PythonExpr

from .python_type_printer import PythonTypePrinter


class PythonPrinter(PythonTypePrinter, ExprFunctor[str]):
    """Shared expression/value visitor base for HIR and TIR printers.

    Inheriting the type printer rather than owning one keeps a single
    ``visit``: an expression printer emits ``expr.type`` through the same
    dispatch that emits the type's own children.
    """

    def __init__(self) -> None:
        PythonTypePrinter.__init__(self)
        ExprFunctor.__init__(self)

    def atom_reference(self, value, ctx=None) -> str:
        return f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"

    def render_value(self, value, ctx=None, indent: str = "") -> str:
        """Render a non-type DSL attribute value and register its imports.

        Type values are not routed here: a printer that holds an ``expr.type``
        calls ``self.visit`` directly, so this stays the surface for the
        op-attribute literals that are not part of the type language.
        """
        if isinstance(value, (TensorType, Mesh, LayoutBase, DType)):
            with self.type_surface(indent=indent):
                return self.visit(value, ctx)
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
        if isinstance(value, (str, int, float, bool, tuple, type(None))):
            if isinstance(value, tuple):
                vals = ", ".join(self.render_value(v, ctx, indent) for v in value)
                return f"({vals}{',' if len(value)==1 else ''})"
            return repr(value)
        raise NotImplementedError(f"no canonical Python form for {type(value).__name__}")

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
