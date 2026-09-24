"""Canonical Python visitor shared by the HIR and TIR printers."""

from __future__ import annotations

import enum
from contextlib import contextmanager

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.types import DType, TensorType, TupleType, UnitType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimConst,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout, LayoutBase, Swizzle
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprFunctor, TypeFunctor
from tilefoundry.target import Target
from tilefoundry.utils.python_source import PythonExpr

_DIM_INFIX_OPS: dict[type, str] = {
    DimAdd: "+",
    DimSub: "-",
    DimMul: "*",
    DimFloorDiv: "//",
    DimMod: "%",
}

_DIM_FUNC_OPS: dict[type, str] = {
    DimMin: "min",
    DimMax: "max",
}


class PythonPrinter(ExprFunctor[str], TypeFunctor[str]):
    """Render the Python DSL with one dispatch root for expressions and types."""

    def __init__(self) -> None:
        ExprFunctor.__init__(self)
        self._indent = ""
        self._tensor_head = "Tensor"
        self._nested_dim = False

    def visit(self, value, ctx=None):  # type: ignore[override]
        """Dispatch every implemented functor family by the concrete node name."""
        method = getattr(self, f"visit_{type(value).__name__}", None)
        if method is not None:
            return method(value, ctx)
        return self.default_visit(value, ctx)

    def default_visit(self, value, ctx=None) -> str:
        if isinstance(value, bool):
            return repr(value)
        if isinstance(value, (int, float)):
            return str(value)
        raise NotImplementedError(f"no Python visit routine for {type(value).__name__}")

    @contextmanager
    def type_surface(self, *, indent: str | None = None, const: bool = False):
        """Carry statement indentation and parameter const-ness through type visits."""
        previous = (self._indent, self._tensor_head)
        if indent is not None:
            self._indent = indent
        self._tensor_head = "ConstTensor" if const else "Tensor"
        try:
            yield
        finally:
            self._indent, self._tensor_head = previous

    @contextmanager
    def nested_dim(self, nested: bool):
        previous = self._nested_dim
        self._nested_dim = nested
        try:
            yield
        finally:
            self._nested_dim = previous

    def dim_entry(self, value, ctx=None, *, nested: bool = False) -> str:
        """One entry of a shape or stride tuple, which may itself be a group.

        A layout groups the modes of one tensor axis by writing them as a
        tuple in that axis's place, so an entry is read as the shape tuple it
        is rather than as a single dimension.
        """
        if isinstance(value, tuple):
            entries = ", ".join(self.dim_entry(item, ctx) for item in value)
            return f"({entries}{',' if len(value) == 1 else ''})"
        with self.nested_dim(nested):
            return self.visit(value, ctx)

    def shape_tuple(self, shape: tuple, ctx=None) -> str:
        values = tuple(self.dim_entry(entry, ctx) for entry in shape)
        return f"({values[0]},)" if len(values) == 1 else "(" + ", ".join(values) + ")"

    def dtype_str(self, dtype: DType, ctx=None) -> str:
        return dtype.name

    def visit_DimVar(self, value: DimVar, ctx=None) -> str:
        if ctx is not None:
            ctx.declare_dim(value.name, value)
        return value.name

    def visit_Var(self, value: Var, ctx=None) -> str:
        return value.name

    def visit_Constant(self, value: Constant, ctx=None) -> str:
        return repr(value.value)

    def visit_Tuple(self, value: Tuple, ctx=None) -> str:
        rendered = ", ".join(self.visit(item, ctx) for item in value.elements)
        return f"({rendered}{',' if len(value.elements) == 1 else ''})"

    def visit_Call(self, value: Call, ctx=None) -> str:
        ceildiv_args = self._ceildiv_args(value)
        if ceildiv_args is not None:
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.types.dim import ceildiv",), "ceildiv"))
            left, right = ceildiv_args
            return f"ceildiv({self.dim_entry(left, ctx)}, {self.dim_entry(right, ctx)})"
        target = value.target
        if isinstance(target, MeshCoord):
            return self._mesh_coordinate_text(value, target, ctx)
        if isinstance(target, DimConst):
            return str(target.value)
        for op_type, symbol in _DIM_INFIX_OPS.items():
            if isinstance(target, op_type):
                left, right = value.args
                rendered = (
                    f"{self.dim_entry(left, ctx, nested=True)} {symbol} "
                    f"{self.dim_entry(right, ctx, nested=True)}"
                )
                return f"({rendered})" if self._nested_dim else rendered
        for op_type, name in _DIM_FUNC_OPS.items():
            if isinstance(target, op_type):
                args = ", ".join(self.dim_entry(arg, ctx) for arg in value.args)
                return f"{name}({args})"
        return self.visit_program_call(value, ctx)

    def _mesh_coordinate_text(self, value: Call, target: MeshCoord, ctx) -> str:
        """Render one coordinate through the active binding of its mesh."""
        axis = static_dim_value(value.args[0]) if value.args else None
        if axis is None or axis < 0 or axis >= len(target.mesh.layout.shape):
            raise ValueError("MeshCoord requires a literal in-range axis to print")
        if ctx is None:
            raise ValueError("MeshCoord requires an active mesh binding to print")
        ref = ctx.mesh_axis_alias(target.mesh, axis)
        if ref is not None:
            return ref
        alias = ctx.mesh_alias(target.mesh)
        if alias is None:
            raise ValueError("MeshCoord mesh has no active binding to print")
        if axis < len(target.mesh.names):
            axis_name = target.mesh.names[axis]
        elif axis < 3:
            axis_name = ("x", "y", "z")[axis]
        else:
            raise ValueError("unnamed MeshCoord axes above z cannot be printed")
        return f"{alias}.{axis_name}"

    def visit_program_call(self, value: Call, ctx=None) -> str:
        raise NotImplementedError(f"{type(self).__name__} cannot render program calls")

    @staticmethod
    def _ceildiv_args(value: Call) -> tuple[object, object] | None:
        """Recover the public constructor from ceildiv's canonical arithmetic tree."""
        if not isinstance(value.target, DimFloorDiv) or len(value.args) != 2:
            return None
        numerator, divisor = value.args
        if not (
            isinstance(numerator, Call)
            and isinstance(numerator.target, DimSub)
            and len(numerator.args) == 2
            and isinstance(numerator.args[1], Constant)
            and numerator.args[1].value == 1
        ):
            return None
        added = numerator.args[0]
        if not (
            isinstance(added, Call)
            and isinstance(added.target, DimAdd)
            and len(added.args) == 2
            and added.args[1] == divisor
        ):
            return None
        return added.args[0], divisor

    def shard_surface(self, value: ShardLayout, ctx=None) -> str | None:
        """Render placement sugar only when every mesh axis has a scope binding.

        The sugar states one extent per tensor axis, each a dimension
        expression, so it declines a layout whose modes are grouped by tile
        axis: writing the groups in would emit a line the parser refuses.
        """
        layout = value.layout
        names = value.mesh.names
        if (
            not isinstance(layout, Layout)
            or not names
            or len(value.attrs) != len(names)
            or ctx is None
        ):
            return None
        if any(
            isinstance(entry, tuple)
            for entry in (*layout.shape, *(layout.strides or ()))
        ):
            return None
        refs = tuple(ctx.mesh_axis_alias(value.mesh, index) for index in range(len(names)))
        if any(ref is None for ref in refs):
            return None

        splits: dict[int, list[str]] = {}
        partials: list[str] = []
        broadcasts: list[tuple[str, str, Broadcast]] = []
        named_bindings: set[str] = set()
        for index, (attr, ref) in enumerate(zip(value.attrs, refs, strict=True)):
            assert ref is not None
            binding = ref.partition(".")[0]
            if isinstance(attr, Split):
                if attr.axis >= len(layout.shape):
                    return None
                splits.setdefault(attr.axis, []).append(ref)
                named_bindings.add(binding)
            elif isinstance(attr, Partial):
                partials.append(f'{ref} @ {self.visit(attr, ctx)}')
                named_bindings.add(binding)
            elif isinstance(attr, Broadcast):
                broadcasts.append((binding, ref, attr))
            else:
                return None

        states = list(partials)
        for binding, ref, attr in broadcasts:
            if binding not in named_bindings:
                states.append(f"{ref} @ {self.visit(attr, ctx)}")
                named_bindings.add(binding)
        if not splits and not states:
            states = [f"{ref} @ {self.visit(attr, ctx)}" for _binding, ref, attr in broadcasts]
        if not splits and not states:
            return None

        explicit = layout.strides is not None
        if explicit and any(
            axis in splits
            and self.dim_entry(dim, ctx, nested=True) != self.dim_entry(dim, ctx)
            for axis, dim in enumerate(layout.shape)
        ):
            return None

        dims = [
            (
                f"{self.dim_entry(dim, ctx, nested=True)} "
                + " ".join(f"@ {ref}" for ref in splits[axis])
            )
            if axis in splits
            else self.dim_entry(dim, ctx)
            for axis, dim in enumerate(layout.shape)
        ]
        dims_text = ", ".join(dims) + ("," if len(dims) == 1 else "")
        parts = [f"({dims_text})"]
        if explicit:
            parts.append(self.shape_tuple(layout.strides, ctx))
        if states:
            parts.append("{" + ", ".join(states) + "}")
        return parts[0] if len(parts) == 1 else "(" + ", ".join(parts) + ")"

    def visit_TensorType(self, value: TensorType, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr((f"from tilefoundry.dsl import {self._tensor_head}",), self._tensor_head))
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
        elif value.layout is not None:
            result += f", {self.visit(value.layout, ctx)}"
        if value.storage is not StorageKind.GMEM:
            result += f', "{value.storage.name.lower()}"'
        return result + "]"

    def visit_TupleType(self, value: TupleType, ctx=None) -> str:
        return f"Tuple[{', '.join(self.visit(field, ctx) for field in value.fields)}]"

    def visit_UnitType(self, value: UnitType, ctx=None) -> str:
        return "None"

    def visit_DType(self, value: DType, ctx=None) -> str:
        return self.dtype_str(value, ctx)

    def visit_Mesh(self, value: Mesh, ctx=None) -> str:
        if ctx is not None:
            alias = ctx.mesh_alias(value)
            if alias is not None:
                return alias
            sliced = ctx.mesh_slice(value)
            if sliced is not None:
                return sliced
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Mesh, Topology",), "Mesh"))
        topologies = ", ".join(
            f'Topology("{topology.name}", {self.dim_entry(topology.size, ctx)})'
            for topology in value.topologies
        )
        topologies = f"({topologies}{',' if len(value.topologies) == 1 else ''})"
        result = f"Mesh({topologies}, {self.visit(value.layout, ctx)}"
        if value.names:
            result += f", names={tuple(value.names)!r}"
        return result + ")"

    def visit_NoneType(self, value: None, ctx=None) -> str:
        return "None"

    def visit_Layout(self, value: Layout, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Layout",), "Layout"))
        strides = self.shape_tuple(value.strides, ctx) if value.strides is not None else "None"
        return f"Layout({self.shape_tuple(value.shape, ctx)}, {strides})"

    def visit_Swizzle(self, value: Swizzle, ctx=None) -> str:
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import Swizzle",), ""))
        return f"Swizzle({value.bits}, {value.base}, {value.shift})"

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
        surface = self.shard_surface(value, ctx)
        if surface is not None:
            return surface
        if ctx is not None:
            ctx.use(PythonExpr(("from tilefoundry.ir.types.shard import ShardLayout",), ""))
        outer, child = self._indent, self._indent + "    "
        attrs = ", ".join(self.visit(attr, ctx) for attr in value.attrs)
        if len(value.attrs) == 1:
            attrs += ","
        with self.type_surface(indent=child):
            layout_text = self.visit(value.layout, ctx)
            mesh_text = self.visit(value.mesh, ctx)
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

    def atom_reference(self, value: MmaAtom, ctx=None) -> str:
        return f"T.cuda.mma.atom(op=T.cuda.mma.{value.op.name})"

    def render_value(self, value, ctx=None, indent: str = "") -> str:
        """Render a non-expression attribute through the same visitor when possible."""
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
            rendered = value.to_python()
            return ctx.use(rendered) if ctx is not None else rendered.text
        if isinstance(value, tuple):
            rendered = ", ".join(self.render_value(item, ctx, indent) for item in value)
            return f"({rendered}{',' if len(value) == 1 else ''})"
        if value is None or isinstance(value, (str, int, float, bool)):
            return repr(value)
        raise NotImplementedError(f"no canonical Python form for {type(value).__name__}")

    def render_pattern(self, pattern: Pattern, ctx=None) -> str:
        if isinstance(pattern, DimVarRangePat):
            if ctx is not None:
                ctx.use(PythonExpr(("from tilefoundry.ir.core.pattern import DimVarRangePat",), ""))
            return f'DimVarRangePat("{pattern.dim_var}", {pattern.lo}, {pattern.hi})'
        return repr(pattern)


__all__ = ["PythonPrinter"]
