"""CUDA MMA op / atom model (cutlass ``MMA_Op`` → ``MMA_Atom``)."""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Mesh, ShardLayout


@dataclass(frozen=True)
class MmaOpSpec:
    """A named, fully-specified MMA instruction (cutlass ``MMA_Op``)."""
    name: str
    shape_mnk: tuple[int, int, int]
    dtype_a: DType
    dtype_b: DType
    dtype_c: DType
    operand_layout: str  # e.g. "TN" (A row-major, B col-major)


@dataclass(frozen=True)
class MmaAtom:
    """Realized MMA atom (cutlass ``MMA_Atom``) — op + fragment layouts + required scope."""
    op: MmaOpSpec
    A: ShardLayout
    B: ShardLayout
    C: ShardLayout
    required_scope: Mesh


__all__ = ["MmaOpSpec", "MmaAtom"]
