"""The mesh scope statement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr, types

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class MeshScope(Stmt):
    mesh: types.Mesh
    binding: expr.Var
    body: Sequential

__all__ = ['MeshScope']
