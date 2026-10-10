"""Shared state accumulated while canonical Python source is rendered."""

from __future__ import annotations

from contextlib import contextmanager

from tilefoundry.ir.types.layout import ComposedLayout, Layout, flatten
from tilefoundry.ir.types.mesh import Mesh, axis_keys, make_mesh
from tilefoundry.utils.python_source import PythonExpr, _merge_imports

from .mesh_utils import selection_layout, sub_box


class PrintContext:
    """Imports, symbolic declarations, and lexical mesh bindings for one file."""

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self._dim_declarations: dict[str, tuple[object, str]] = {}
        self._mesh_bindings: list[tuple[Mesh, str]] = []
        self._used_scope_names: set[str] = set()
        self._type_annotation_surface = False
        self.topologies = ()

    def use(self, rendered: PythonExpr | str) -> str:
        if isinstance(rendered, PythonExpr):
            self.imports.update(rendered.imports)
            return rendered.text
        return rendered

    def declare_dim(self, name: str, var) -> None:
        self._dim_declarations.setdefault(name, (var, "DimVar"))

    def header(self) -> list[str]:
        """Render file imports and declarations reached while rendering the body."""
        lines = [
            "from __future__ import annotations",
            "",
            "from tilefoundry.dsl import *",
            *_merge_imports(tuple(self.imports)),
            "",
        ]
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

    @property
    def current_mesh(self) -> Mesh | None:
        return make_mesh(*(mesh for mesh, _name in self._mesh_bindings)) if self._mesh_bindings else None

    def mesh_alias(self, mesh: Mesh) -> str | None:
        for bound, name in reversed(self._mesh_bindings):
            if bound == mesh:
                return name
        return None

    def mesh_axis_alias(self, mesh: Mesh, axis: int) -> str | None:
        """Name one mesh axis through an active scope binding, if one dominates it."""
        names = mesh.names
        if axis >= len(names):
            return None
        target_level, target_name = axis_keys(mesh)[axis]
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
            for bound_level, bound_name in axis_keys(bound):
                if bound_name != target_name or bound_level != target_level:
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

    def mesh_selection(self, mesh: Mesh) -> tuple[str, Layout] | None:
        """Recover a lexical selection and its local composition layout."""
        for parent, alias in reversed(self._mesh_bindings):
            layout = selection_layout(mesh, parent)
            if layout is not None:
                return alias, layout
            box = sub_box(parent, mesh)
            if box is None:
                continue
            selection = parent[box]
            layout = selection_layout(mesh, selection)
            if layout is not None:
                return self._box_text(parent, box, alias), layout
        return None

    @staticmethod
    def _slice_from_parent(parent: Mesh, child: Mesh, alias: str) -> str | None:
        if parent.names != child.names:
            return None
        box = sub_box(parent, child)
        if box is None or parent[box] != child:
            return None
        return PrintContext._box_text(parent, box, alias)

    @staticmethod
    def _box_text(parent: Mesh, box: tuple[slice, ...], alias: str) -> str:
        pieces: list[str] = []
        for part, extent in zip(box, flatten(parent.layout).shape):
            start, stop = part.start, part.stop
            if start == 0 and stop == extent:
                pieces.append(":")
            else:
                pieces.append(f"{'' if start == 0 else start}:{'' if stop == extent else stop}")
        while len(pieces) > 1 and pieces[-1] == ":":
            pieces.pop()
        return f"{alias}[{', '.join(pieces)}]"


class HirPrintContext(PrintContext):
    pass


class TirPrintContext(PrintContext):
    pass


__all__ = ["PrintContext", "HirPrintContext", "TirPrintContext"]
