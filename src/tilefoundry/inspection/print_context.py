"""Shared state accumulated while canonical Python source is rendered."""

from __future__ import annotations

import keyword
import re
from contextlib import contextmanager

from tilefoundry import utils
from tilefoundry.ir import types

from .mesh_utils import selection_layout, sub_box


def box_text(parent: types.Mesh, box: tuple[slice, ...], alias: str) -> str:
    pieces: list[str] = []
    for part, extent in zip(box, types.flatten(parent.layout).shape):
        start, stop = part.start, part.stop
        if start == 0 and stop == extent:
            pieces.append(":")
        else:
            pieces.append(f"{'' if start == 0 else start}:{'' if stop == extent else stop}")
    while len(pieces) > 1 and pieces[-1] == ":":
        pieces.pop()
    return f"{alias}[{', '.join(pieces)}]"


def slice_from_parent(parent: types.Mesh, child: types.Mesh, alias: str) -> str | None:
    if parent.names != child.names:
        return None
    box = sub_box(parent, child)
    if box is None or parent[box] != child:
        return None
    return box_text(parent, box, alias)


def identifier(name: str) -> str:
    name = re.sub(r"\W", "_", name)
    if not name or name[0].isdigit() or keyword.iskeyword(name):
        name = "_" + name
    return name


class PrintContext:
    """Imports, symbolic declarations, and lexical mesh bindings for one file."""

    def __init__(self, *, target=None, value_printer=None) -> None:
        self.imports: set[str] = set()
        self.dim_declarations: dict[str, tuple[object, str]] = {}
        self.mesh_bindings: list[tuple[types.Mesh, str]] = []
        self.used_scope_names: set[str] = set()
        self.topologies = ()
        self.target = target
        self.value_printer = value_printer

    def enter_module(self, target, topologies) -> tuple:
        """Make a module's target and topologies current; return the ones it replaces."""
        previous = (self.target, self.topologies)
        self.target = target or self.target
        self.topologies = topologies if topologies is not None else self.topologies
        return previous

    def leave_module(self, previous: tuple) -> None:
        """Restore the target and topologies ``enter_module`` returned."""
        self.target, self.topologies = previous

    def use(self, rendered: utils.PythonExpr | str) -> str:
        if isinstance(rendered, utils.PythonExpr):
            self.imports.update(rendered.imports)
            return rendered.text
        return rendered

    def declare_dim(self, name: str, var) -> None:
        self.dim_declarations.setdefault(name, (var, "DimVar"))

    def header(self) -> list[str]:
        """Render file imports and declarations reached while rendering the body."""
        lines = [
            "from __future__ import annotations",
            "",
            "from tilefoundry.dsl import *",
            *utils.merge_imports(tuple(self.imports)),
            "",
        ]
        if self.dim_declarations:
            lines.extend(
                f'{name} = {constructor}("{var.name}", {var.lo}, {var.hi})'
                for name, (var, constructor) in self.dim_declarations.items()
            )
            lines.append("")
        return lines

    def scope_name(self, mesh: types.Mesh, preferred: str | None = None) -> str:
        base = preferred or (mesh.topologies[0].name if mesh.topologies else "mesh")
        base = base if base.isidentifier() else "mesh"
        name = base
        suffix = 2
        while name in self.used_scope_names:
            name = f"{base}_{suffix}"
            suffix += 1
        self.used_scope_names.add(name)
        return name

    def push_mesh(self, mesh: types.Mesh, name: str) -> None:
        self.mesh_bindings.append((mesh, name))
        self.used_scope_names.add(name)

    def pop_mesh(self) -> None:
        self.mesh_bindings.pop()

    @property
    def current_mesh(self) -> types.Mesh | None:
        return types.Mesh.from_meshes(*(mesh for mesh, _ in self.mesh_bindings)) if self.mesh_bindings else None

    def mesh_alias(self, mesh: types.Mesh) -> str | None:
        for bound, name in reversed(self.mesh_bindings):
            if bound == mesh:
                return name
        return None

    def mesh_axis_alias(self, mesh: types.Mesh, axis: int) -> str | None:
        """Name one mesh axis through an active scope binding, if one dominates it."""
        names = mesh.names
        if axis >= len(names):
            return None
        target_level, target_name = types.axis_keys(mesh)[axis]
        target_topology = next(
            (topology for topology in mesh.topologies if topology.name == target_level), None
        )
        for binding_index in range(len(self.mesh_bindings) - 1, -1, -1):
            bound, alias = self.mesh_bindings[binding_index]
            if not bound.names or target_name not in bound.names:
                continue
            for bound_level, bound_name in types.axis_keys(bound):
                if bound_name != target_name or bound_level != target_level:
                    continue
                bound_topology = next(
                    topology for topology in bound.topologies if topology.name == target_level
                )
                target_geometry = next(
                    part for part in types.separate(mesh) if part.topologies[0].name == target_level
                )
                bound_geometry = next(
                    part for part in types.separate(bound) if part.topologies[0].name == target_level
                )
                if (
                    target_topology == bound_topology
                    and target_geometry.layout == bound_geometry.layout
                ):
                    return f"{alias}.{bound_name}"
        return None

    def mesh_slice(self, mesh: types.Mesh) -> str | None:
        """Recover ``binding[start:stop]`` for a sliced active mesh."""
        if not isinstance(mesh.layout, types.ComposedLayout):
            return None
        for parent, alias in reversed(self.mesh_bindings):
            text = slice_from_parent(parent, mesh, alias)
            if text is not None:
                return text
        return None

    def mesh_selection(self, mesh: types.Mesh) -> tuple[str, types.Layout] | None:
        """Recover a lexical selection and its local composition layout."""
        for parent, alias in reversed(self.mesh_bindings):
            layout = selection_layout(mesh, parent)
            if layout is not None:
                return alias, layout
            box = sub_box(parent, mesh)
            if box is None:
                continue
            selection = parent[box]
            layout = selection_layout(mesh, selection)
            if layout is not None:
                return box_text(parent, box, alias), layout
        return None





class HirPrintContext(PrintContext):
    pass


class TirPrintContext(PrintContext):
    def __init__(self) -> None:
        super().__init__()
        self.binding_names = {}
        self.reserved_names = set()
        self.valuation = None

    def bind_function(self, frame, target) -> None:
        """Take the call-path frame and target of the PrimFunction about to print."""
        self.valuation = frame
        self.target = target

    @contextmanager
    def definition(self, bindings):
        """Reserve all authored identifiers before naming one independent def.

        Imports stay file-wide; mesh bindings and allocated names belong to this
        definition. A generated suffix cannot consume a later authored name.
        """
        previous = self.mesh_bindings, self.used_scope_names, self.binding_names, self.reserved_names
        self.mesh_bindings, self.used_scope_names = [], set()
        self.binding_names = {}
        self.reserved_names = {
            binding.name for binding in bindings
            if binding.name.isidentifier() and not keyword.iskeyword(binding.name)
        }
        try:
            for binding in bindings:
                self.binding_name(binding)
            yield
        finally:
            self.mesh_bindings, self.used_scope_names, self.binding_names, self.reserved_names = previous

    def binding_name(self, binding) -> str:
        known = self.binding_names.get(id(binding))
        if known is not None:
            return known
        base = identifier(binding.name)
        name, suffix = base, 2
        while name in self.used_scope_names or (
            name in self.reserved_names and name != binding.name
        ):
            name = f"{base}_{suffix}"
            suffix += 1
        self.used_scope_names.add(name)
        self.binding_names[id(binding)] = name
        return name


__all__ = ["PrintContext", "HirPrintContext", "TirPrintContext", "identifier"]
