"""Dispatch, read-only walks and rebuilding walks over type values.

Types are immutable value objects rather than expressions. ``TypeVisitor`` and
``TypeMutator`` walk the eight type classes by their stated fields only; every
shape, stride, offset and size entry goes to ``visit_dim``. A dtype, a storage,
a shard attr, a name, a ``Swizzle``, a topology written as a string and ``None``
are leaves. See [visitor-mutator](docs/spec/visitor-mutator.md).
"""

from __future__ import annotations

from typing import Any

from .layout import ComposedLayout, Layout
from .mesh import Mesh
from .pointer import PointerType
from .shard_layout import ShardLayout
from .tensor_type import TensorType, TupleType
from .topology import Topology

__all__ = ["TypeFunctor", "TypeMutator", "TypeVisitor"]

TYPE_VALUES = (
    TensorType,
    TupleType,
    PointerType,
    Layout,
    ComposedLayout,
    ShardLayout,
    Mesh,
    Topology,
)


class TypeFunctor[T]:
    """Dispatch read-only operations over IR type and layout values.

    Types are immutable value objects rather than expressions, so they need a
    separate dispatch surface.  Keeping this visitor independent from the
    expression visitor prevents printers and analysis consumers from growing
    parallel ``isinstance`` ladders for the same type family.
    """

    type_values = TYPE_VALUES

    def visit(self, value: Any, ctx: Any = None) -> T:
        return self.dispatch_visit(value, ctx)

    def dispatch_visit(self, value: Any, ctx: Any) -> T:
        method = getattr(self, f"visit_{type(value).__name__}", None)
        if method is not None:
            return method(value, ctx)
        return self.default_visit(value, ctx)

    def default_visit(self, value: Any, ctx: Any) -> T:
        raise NotImplementedError(f"no type visit routine for {type(value).__name__}")

    def visit_part(self, value: Any, ctx: Any) -> Any:
        """Visit *value* when it is one of the type classes; a leaf comes back as it is."""
        return self.visit(value, ctx) if isinstance(value, self.type_values) else value


class TypeVisitor[T](TypeFunctor[T]):
    """Read-only walk over a type and everything it holds; ``visit_dim`` sees each entry.

    The order is fixed: a tensor's shape before its layout, tuple fields left to
    right, a shard layout's layout before its mesh, a mesh's topologies before
    its layout, a composed layout's outer, inner, then offset, and a layout's
    shape before its strides, nested entries in order.
    """

    def visit_dim(self, entry: Any, ctx: Any) -> None:
        return None

    def visit_entries(self, entries: Any, ctx: Any) -> None:
        for entry in entries:
            if isinstance(entry, tuple):
                self.visit_entries(entry, ctx)
            elif entry is not None:
                self.visit_dim(entry, ctx)

    def visit_TensorType(self, value: TensorType, ctx: Any) -> None:
        self.visit_entries(value.shape, ctx)
        self.visit_part(value.layout, ctx)

    def visit_TupleType(self, value: TupleType, ctx: Any) -> None:
        for field in value.fields:
            self.visit_part(field, ctx)

    def visit_PointerType(self, value: PointerType, ctx: Any) -> None:
        return None

    def visit_Topology(self, value: Topology, ctx: Any) -> None:
        self.visit_dim(value.size, ctx)

    def visit_Mesh(self, value: Mesh, ctx: Any) -> None:
        for topology in value.topologies:
            self.visit_part(topology, ctx)
        self.visit_part(value.layout, ctx)

    def visit_ShardLayout(self, value: ShardLayout, ctx: Any) -> None:
        self.visit_part(value.layout, ctx)
        self.visit_part(value.mesh, ctx)

    def visit_ComposedLayout(self, value: ComposedLayout, ctx: Any) -> None:
        self.visit_part(value.outer, ctx)
        self.visit_part(value.inner, ctx)
        self.visit_dim(value.offset, ctx)

    def visit_Layout(self, value: Layout, ctx: Any) -> None:
        self.visit_entries(value.shape, ctx)
        self.visit_entries(value.strides, ctx)


class TypeMutator(TypeFunctor[Any]):
    """The same walk, rebuilding; a value none of whose parts changed comes back as itself.

    A mesh is built once, after its topologies and its layout are both rebuilt.
    """

    def visit_dim(self, entry: Any, ctx: Any) -> Any:
        return entry

    def visit_entries(self, entries: tuple, ctx: Any) -> tuple:
        return tuple(
            self.visit_entries(entry, ctx)
            if isinstance(entry, tuple)
            else (None if entry is None else self.visit_dim(entry, ctx))
            for entry in entries
        )

    def visit_TensorType(self, value: TensorType, ctx: Any) -> TensorType:
        shape = tuple(self.visit_dim(entry, ctx) for entry in value.shape)
        layout = self.visit_part(value.layout, ctx)
        if shape == value.shape and layout is value.layout:
            return value
        return TensorType(shape=shape, dtype=value.dtype, layout=layout, storage=value.storage)

    def visit_TupleType(self, value: TupleType, ctx: Any) -> TupleType:
        fields = tuple(self.visit_part(field, ctx) for field in value.fields)
        return value if fields == value.fields else TupleType(fields=fields)

    def visit_PointerType(self, value: PointerType, ctx: Any) -> PointerType:
        return value

    def visit_Topology(self, value: Topology, ctx: Any) -> Topology:
        size = self.visit_dim(value.size, ctx)
        return value if size == value.size else Topology(value.name, size)

    def visit_Mesh(self, value: Mesh, ctx: Any) -> Mesh:
        topologies = tuple(self.visit_part(topology, ctx) for topology in value.topologies)
        layout = self.visit_part(value.layout, ctx)
        if topologies == value.topologies and layout == value.layout:
            return value
        return Mesh(topologies=topologies, layout=layout, names=value.names)

    def visit_ShardLayout(self, value: ShardLayout, ctx: Any) -> ShardLayout:
        layout = self.visit_part(value.layout, ctx)
        mesh = self.visit_part(value.mesh, ctx)
        if layout is value.layout and mesh is value.mesh:
            return value
        return ShardLayout(layout=layout, attrs=value.attrs, mesh=mesh)

    def visit_ComposedLayout(self, value: ComposedLayout, ctx: Any) -> ComposedLayout:
        outer = self.visit_part(value.outer, ctx)
        inner = self.visit_part(value.inner, ctx)
        offset = self.visit_dim(value.offset, ctx)
        if outer is value.outer and inner is value.inner and offset == value.offset:
            return value
        return ComposedLayout(inner=inner, offset=offset, outer=outer)

    def visit_Layout(self, value: Layout, ctx: Any) -> Layout:
        shape = self.visit_entries(value.shape, ctx)
        strides = self.visit_entries(value.strides, ctx)
        if shape == value.shape and strides == value.strides:
            return value
        return Layout(shape=shape, strides=strides)
