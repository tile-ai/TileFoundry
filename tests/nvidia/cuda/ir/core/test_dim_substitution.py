"""Choosing an extent for a dimension that was declared as a range."""

from __future__ import annotations

from pathlib import Path

from tests.core.ir.core.test_dim_substitution import CTX
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.types.dim import DimMul, simplify_dim
from tilefoundry.ir.types.substitute import (
    has_symbolic_dims,
)


def test_only_dim_vars_make_dimension_arithmetic_symbolic() -> None:
    persistent_gemm_tiled = Path(__file__).parents[2] / "fixtures/placed/persistent_gemm_tiled.py"
    PersistentGemmTiled = load_authored_ir(f"{persistent_gemm_tiled}:PersistentGemmTiled")
    outer = PersistentGemmTiled.entry_function().body.body
    assert isinstance(outer, LoopRegion)

    assert not has_symbolic_dims(outer.start)
    assert has_symbolic_dims(simplify_dim(DimMul, (64, CTX)))
