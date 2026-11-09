"""Choosing an extent for a dimension that was declared as a range."""

from __future__ import annotations

from pathlib import Path

from tests.core.ir.core.test_dim_substitution import CTX
from tilefoundry import cli, services
from tilefoundry.ir import core, dim


def test_dim_var_makes_arithmetic_symbolic() -> None:
    persistent_gemm_tiled = Path(__file__).parents[2] / "fixtures/placed/persistent_gemm_tiled.py"
    PersistentGemmTiled = cli.load_authored_ir(f"{persistent_gemm_tiled}:PersistentGemmTiled")
    outer = PersistentGemmTiled.entry_function().body.body
    assert isinstance(outer, core.LoopRegion)

    assert not dim.has_symbolic_dims(outer.start)
    assert dim.has_symbolic_dims(services.simplify_dim("mul", (64, CTX)))
