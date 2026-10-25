"""isl_interop — range queries and conversion between IR dimensions and isl."""

from __future__ import annotations

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.ir.core import RangeMetadata, attach_metadata
from tilefoundry.ir.core.expr import Call, Var
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.isl_interop import (
    dim_range,
)
from tilefoundry.ir.types import TensorType


def test_dim_range_prefers_metadata_and_unknown_leaves_return_none():
    unknown = Var(type=TensorType.umat_scalar(), name="runtime")
    assert dim_range(unknown) is None

    attach_metadata(unknown, RangeMetadata(3, 11))
    assert dim_range(unknown) == (3, 11)

    persistent_gemm_tiled = Path(__file__).parents[1] / "fixtures/placed/persistent_gemm_tiled.py"
    PersistentGemmTiled = load_authored_ir(f"{persistent_gemm_tiled}:PersistentGemmTiled")
    outer = PersistentGemmTiled.entry_function().body.body
    assert isinstance(outer, LoopRegion)

    def calls(value):
        if not isinstance(value, Call):
            return ()
        return (value, *(nested for arg in value.args for nested in calls(arg)))

    coordinate = next(call for call in calls(outer.start) if isinstance(call.target, MeshCoord))
    assert dim_range(coordinate) == (0, load_namespace(str(persistent_gemm_tiled))[0]["BX"])
