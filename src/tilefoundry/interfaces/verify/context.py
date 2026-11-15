"""Context of verification."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types


@dataclass
class Context(typeinfer.TypeInferContext):
    """The context a per-stmt verify handler is annotated against.

    The enclosing scopes a handler asks about are the TIR ``MeshScope``
    regions on its region chain: ``enclosing_meshes()`` and ``current_mesh``
    (see [tir §1.3](docs/spec/tir.md#13-primfunction)).
    """

    @typing.override
    def type_of(self, node: expr.Expr) -> types.Type:
        """The type the checked IR already holds: a check reads, it does not infer."""
        return node.type


__all__ = ["Context"]
