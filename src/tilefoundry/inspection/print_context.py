"""Shared state accumulated while canonical Python source is rendered."""

from __future__ import annotations

from contextlib import contextmanager
from math import prod

from tilefoundry.ir.types.int_tuple import repeat_like
from tilefoundry.ir.types.layout import ComposedLayout, Layout, flatten
from tilefoundry.ir.types.mesh import Mesh
from tilefoundry.utils.python_source import PythonExpr, _merge_imports


class PrintContext:
    """Imports, symbolic declarations, and lexical mesh bindings for one file."""

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self._dim_declarations: dict[str, tuple[object, str]] = {}
        self._mesh_bindings: list[tuple[Mesh, str]] = []
        self._used_scope_names: set[str] = set()
        self._type_annotation_surface = False

    def use(self, rendered: PythonExpr | str) -> str:
        if isinstance(rendered, PythonExpr):
            self.imports.update(rendered.imports)
            return rendered.text
        return rendered

    def declare_dim(
        self,
        name: str,
        var,
        *,
        import_statement: str = "from tilefoundry.ir.types.dim import DimVar",
    ) -> None:
        self.imports.add(import_statement)
        self._dim_declarations.setdefault(name, (var, "DimVar"))

    def header(self) -> list[str]:
        """Render only imports and declarations reached while rendering the body."""
        imports = list(_merge_imports(tuple(self.imports)))
        imports = [
            f"{line}  # noqa: F401, F403" if line.endswith(" import *") else line
            for line in imports
        ]
        lines = ["from __future__ import annotations", "", *imports, ""]
        if self._dim_declarations:
            lines.extend(
                f'{name} = {constructor}("{var.name}", {var.lo}, {var.hi})'
                for name, (var, constructor) in self._dim_declarations.items()
            )
            lines.append("")
        return lines

    def scope_name(self, mesh: Mesh, preferred: str | None = None) -> str:
        base = preferred or (mesh.topologies[0].name if mesh.topologies else "mesh")
        base = base if base.isidentifier() else "mesh"
        name = base
        suffix = 2
        while name in self._used_scope_names:
            name = f"{base}_{suffix}"
            suffix += 1
        self._used_scope_names.add(name)
        return name

    def push_mesh(self, mesh: Mesh, name: str) -> None:
        self._mesh_bindings.append((mesh, name))
        self._used_scope_names.add(name)

    def pop_mesh(self) -> None:
        self._mesh_bindings.pop()

    @contextmanager
    def type_annotation_surface(self):
        """Prefer a parent binding when a type refers to a sliced mesh."""
        previous = self._type_annotation_surface
        self._type_annotation_surface = True
        try:
            yield
        finally:
            self._type_annotation_surface = previous

    def mesh_alias(self, mesh: Mesh) -> str | None:
        for bound, name in reversed(self._mesh_bindings):
            if bound is mesh:
                return name
        return None

    @staticmethod
    def _axis_levels(mesh: Mesh) -> tuple[str, ...]:
        """The level each of the mesh's axes stands in, one name per axis."""
        stated = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
        return flatten(
            tuple(
                repeat_like(mode, topology.name)
                for mode, topology in zip(stated.shape, mesh.topologies, strict=True)
            )
        )

    def mesh_axis_alias(self, mesh: Mesh, axis: int) -> str | None:
        """Name one mesh axis through an active scope binding, if one dominates it."""
        names = mesh.names
        if axis >= len(names):
            return None
        target_name = names[axis]
        target_levels = self._axis_levels(mesh)
        target_level = target_levels[axis]
        target_topology = next(
            (topology for topology in mesh.topologies if topology.name == target_level), None
        )
        for binding_index in range(len(self._mesh_bindings) - 1, -1, -1):
            bound, alias = self._mesh_bindings[binding_index]
            if (
                self._type_annotation_surface
                and bound is mesh
                and (mesh.layout.offset if isinstance(mesh.layout, ComposedLayout) else 0) != 0
            ):
                continue
            if not bound.names or target_name not in bound.names:
                continue
            bound_levels = self._axis_levels(bound)
            for bound_axis, bound_name in enumerate(bound.names):
                if bound_name != target_name or bound_levels[bound_axis] != target_level:
                    continue
                bound_topology = next(
                    topology for topology in bound.topologies if topology.name == target_level
                )
                if target_topology == bound_topology:
                    return f"{alias}.{bound_name}"
        return None

    def mesh_slice(self, mesh: Mesh) -> str | None:
        """Recover ``binding[start:stop]`` for a sliced active mesh."""
        if not isinstance(mesh.layout, ComposedLayout):
            return None
        for parent, alias in reversed(self._mesh_bindings):
            text = self._slice_from_parent(parent, mesh, alias)
            if text is not None:
                return text
        return None

    @staticmethod
    def _slice_from_parent(parent: Mesh, child: Mesh, alias: str) -> str | None:
        if not (
            isinstance(parent.layout, Layout)
            and isinstance(child.layout, ComposedLayout)
            and child.layout.inner is None
            and isinstance(child.layout.outer, Layout)
        ):
            return None
        if parent.topologies != child.topologies or parent.names != child.names:
            return None
        parent_shape = flatten(parent.layout.shape)
        parent_strides = flatten(parent.layout.strides)
        child_shape = flatten(child.layout.outer.shape)
        child_strides = flatten(child.layout.outer.strides)
        if (
            len(parent_shape) != len(child_shape)
            or parent_strides != child_strides
            or any(not isinstance(item, int) for item in (*parent_shape, *child_shape, *parent_strides))
            or any(size < 1 or size > extent for size, extent in zip(child_shape, parent_shape))
        ):
            return None

        remaining = child.layout.offset
        starts = [0] * len(parent_shape)
        for axis in sorted(range(len(parent_shape)), key=lambda item: parent_strides[item], reverse=True):
            stride = parent_strides[axis]
            maximum = parent_shape[axis] - child_shape[axis]
            start = min(maximum, remaining // stride) if stride else 0
            starts[axis] = start
            remaining -= start * stride
        if remaining != 0:
            return None
        if sum(start * stride for start, stride in zip(starts, parent_strides)) != child.layout.offset:
            return None
        if prod(child_shape) > prod(parent_shape):
            return None

        pieces: list[str] = []
        for start, size, extent in zip(starts, child_shape, parent_shape):
            if start == 0 and size == extent:
                pieces.append(":")
                continue
            stop = start + size
            pieces.append(f"{'' if start == 0 else start}:{'' if stop == extent else stop}")
        while len(pieces) > 1 and pieces[-1] == ":":
            pieces.pop()
        return f"{alias}[{', '.join(pieces)}]"


class HirPrintContext(PrintContext):
    pass


class TirPrintContext(PrintContext):
    pass


__all__ = ["PrintContext", "HirPrintContext", "TirPrintContext"]
