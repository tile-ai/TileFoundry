"""Focused contracts for footprint and reuse-window helpers."""

from typing import cast

import isl

from tilefoundry.analysis.access import Access
from tilefoundry.analysis.footprint import MovingBoundary, axis_label
from tilefoundry.analysis.iteration_scope import IterationScope
from tilefoundry.ir.core import Call, Op, Var
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Mesh
from tilefoundry.ir.types.storage import StorageKind


def test_an_unnamed_mesh_axis_uses_its_position() -> None:
    mesh = Mesh(("cta",), layout=(2, 3), names=("row",))

    assert axis_label(mesh, 0) == "cta.row"
    assert axis_label(mesh, 1) == "cta[1]"


def test_time_comparison_keeps_mesh_parameters() -> None:
    value_type = TensorType(shape=(2,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
    buffer = Var(type=value_type, name="x")
    call = Call(type=value_type, target=Op(), args=(buffer,))
    coordinate = Call(type=value_type, target=Op(), args=())
    relation = isl.map(
        "[c] -> { [i] -> [o] : 0 <= c <= 1 and i = 0 and o = c; "
        "[i] -> [o] : 0 <= c <= 1 and i = 1 and o = 1 - c }"
    )
    boundary = MovingBoundary(
        scope=cast(IterationScope, object()),
        call=call,
        access=Access(input_index=0, relation=relation, buffer=buffer),
        dtype=DType.f32,
        label="x",
        mesh=None,
        reads=True,
        wave_units=2,
        wave_stated=True,
        mesh_parameters=(("c", coordinate),),
        axis_parameters=(),
        position=None,
    )

    assert not boundary.held(0).is_equal(boundary.held(-1))
    assert boundary.reached(0).is_equal(boundary.reached(-1))
