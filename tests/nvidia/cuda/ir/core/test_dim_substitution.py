"""Choosing an extent for a dimension that was declared as a range."""

from __future__ import annotations

from tests.core.ir.core.test_dim_substitution import CTX
from tests.nvidia.cuda.fixtures.placed.persistent_gemm_tiled import PersistentGemmTiled
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.types.dim import DimMul, simplify_dim
from tilefoundry.ir.types.substitute import (
    has_symbolic_dims,
)


def test_only_dim_vars_make_dimension_arithmetic_symbolic() -> None:
    outer = PersistentGemmTiled.entry_function().body.body
    assert isinstance(outer, LoopRegion)

    assert not has_symbolic_dims(outer.start)
    assert has_symbolic_dims(simplify_dim(DimMul, (64, CTX)))
