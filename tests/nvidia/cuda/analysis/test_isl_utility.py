"""isl_interop — range queries and conversion between IR dimensions and isl."""

from __future__ import annotations

from pathlib import Path

from tilefoundry import cli, services
from tilefoundry.ir import core, dim, expr, types


def test_dim_range_resolves_known_values():
    unknown = expr.Var(type=types.TensorType.from_umat_scalar(), name="runtime")
    assert services.dim_range(unknown) is None

    unknown.attach_metadata(expr.RangeMetadata(3, 11))
    assert services.dim_range(unknown) == (3, 11)

    persistent_gemm_tiled = Path(__file__).parents[1] / "fixtures/placed/persistent_gemm_tiled.py"
    PersistentGemmTiled = cli.load_authored_ir(f"{persistent_gemm_tiled}:PersistentGemmTiled")
    outer = PersistentGemmTiled.entry_function().body.body
    assert isinstance(outer, core.LoopRegion)

    def calls(value):
        if not isinstance(value, expr.Call):
            return ()
        return (value, *(nested for arg in value.args for nested in calls(arg)))

    coordinate = next(call for call in calls(outer.start) if isinstance(call.target, dim.MeshCoord))
    assert services.dim_range(coordinate) == (0, cli.load_namespace(str(persistent_gemm_tiled))[0]["BX"])
