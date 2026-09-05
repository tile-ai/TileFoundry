from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.types.shard.mesh import Mesh


@dataclass(unsafe_hash=True)
class MeshRegion(Expr):
    """The participants a region of work runs on, as a node rather than a stack.

    Which participants run a region is a fact about the program's structure, so
    it is a region -- not a snapshot copied onto every expression inside, and not
    something read back off the layout a result happens to carry. The node states
    the participants and nothing else: what each unit sees of an operand is read
    off that operand's own ``ShardLayout``, the way ``Local`` already reads it, so
    there is no second copy of that to drift.
    """

    mesh: Mesh
    body: Expr
    params: tuple[Var, ...] = ()
    args: tuple[Expr, ...] = ()


__all__ = ["MeshRegion"]
