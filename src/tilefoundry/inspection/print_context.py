"""Shared printer state and policy context."""

from __future__ import annotations

from tilefoundry.utils.python_source import PythonExpr


class PrintContext:
    def __init__(self) -> None:
        self.imports: set[str] = set()

    def mesh_count(self) -> int:
        return 0

    def use(self, rendered: PythonExpr | str) -> str:
        if isinstance(rendered, PythonExpr):
            self.imports.update(rendered.imports)
            return rendered.text
        return rendered

    def mesh_alias(self, mesh) -> str | None:
        return None


class HirPrintContext(PrintContext):
    def __init__(self, mesh_name_map: dict[int, str] | None = None) -> None:
        super().__init__()
        self.mesh_name_map = mesh_name_map or {}

    def mesh_alias(self, mesh) -> str | None:
        return self.mesh_name_map.get(id(mesh))

    def mesh_count(self) -> int:
        return len(self.mesh_name_map)


class TirPrintContext(PrintContext):
    def __init__(self) -> None:
        super().__init__()
        self._mesh_aliases: list[dict[int, str]] = []

    def push_mesh(self, mesh, name: str) -> None:
        self._mesh_aliases.append({id(mesh): name})

    def pop_mesh(self) -> None:
        self._mesh_aliases.pop()

    def mesh_alias(self, mesh) -> str | None:
        for aliases in reversed(self._mesh_aliases):
            if id(mesh) in aliases:
                return aliases[id(mesh)]
        return None
