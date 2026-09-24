"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
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
