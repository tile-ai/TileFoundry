"""Context of verification."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types


@dataclass
class Context(typeinfer.TypeInferContext):
    """The context a per-stmt verify handler is annotated against.

    The enclosing scope a handler asks about is the TIR ``mesh_scope``
    tuple, set by whoever walks the statements
    (see [tir §1.3](docs/spec/tir.md#13-primfunction)).
    """

    mesh_scope: tuple = ()

    def enter_scope(self, scope: typing.Iterable[types.Mesh]) -> None:
        """Take *scope* as the enclosing mesh scope; the current mesh is its composition."""
        self.mesh_scope = tuple(scope)
        self.current_mesh = None
        for mesh in self.mesh_scope:
            self.current_mesh = (
                types.Mesh.from_meshes(self.current_mesh, mesh) if self.current_mesh else mesh
            )

    @typing.override
    def type_of(self, node: expr.Expr) -> types.Type:
        """The type the checked IR already holds: a check reads, it does not infer."""
        return node.type


__all__ = ["Context"]
