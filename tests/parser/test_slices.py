"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tests.fixtures.placed.grouped_window import GroupedWindow
from tilefoundry import func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.parser import ParseError
from tilefoundry.target import CudaTarget


def test_a_tile_window_bound_names_the_authored_fix() -> None:
    with pytest.raises(
        ParseError,
        match=(
            r"tile loop variable is already a window.*"
            r"x\[:, t, :\].*base = t \+ 0"
        ),
    ):

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def stage(
            x: Tensor[(1, 128, 64), "f32"],
            out: Tensor[(1, 128, 64), "f32"],
        ) -> Tensor[(1, 128, 64), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("unit",)) as mesh:
                acc = out
                for t in tile(128, 128):
                    window = tf.reshard(
                        x[:, t:t + 128, :],
                        (1, 128 @ mesh.unit, 64),
                        "smem",
                    )
                    acc = tf.insert_slice(acc, window, (0, t, 0))
                return acc


def test_a_window_of_a_grouped_box_takes_the_modes_it_steps_the_least_by() -> None:
    """An axis written as a group has no single stride for a start to walk.

    The window is the modes the axis steps the least by, as many of them as
    its size takes, kept where the group wrote them, and a layout holding
    such a group prints its groups and reads back as what it was.
    """
    function = next(f for f in GroupedWindow.functions if f.name == "corner")
    window = next(
        expr
        for expr in collect_exprs(function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Slice)
    )
    assert window.type.layout.offset == 0
    assert tuple(window.type.layout.outer.shape) == ((8,), (16,))
    assert tuple(window.type.layout.outer.strides) == ((64,), (1,))

    printed = as_script(GroupedWindow)
    assert "Layout(((2, 8), (4, 16)), ((512, 64), (16, 1)))" in printed
    assert as_script(import_dsl(printed, name="GroupedWindow")) == printed
