"""Programs priced by the mesh that runs them.

``UnderFilled`` runs 40 of the 132 declared lanes. ``SharedWindow`` has every
position write the same window; ``PlacedWindow`` gives each position a window
of its own.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
ROW = 8192


@module(entry="f", target=_H200, topologies=(Topology("cta", 132),))
class UnderFilled:
    """Forty of the 132 declared lanes run the whole program."""

    @func
    def f(x: Tensor[(40, ROW), "bf16"]):
        with Mesh(("cta",), layout=(40,), names=("lane",)) as cta:
            held = tf.reshard(x, (40 @ cta.lane, ROW), "rmem")
            return tf.reshard(held, (40, ROW), "gmem")


@module(entry="f", target=_H200, topologies=(Topology("cta", 4),))
class SharedWindow:
    """A zeros and a loop-placed insert_slice every position performs alike."""

    @func
    def f(y: Tensor[(1, 8, 16), "bf16"]) -> Tensor[(4, 8, 16), "bf16"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as _cta:
            out = tf.zeros(Tensor[(4, 8, 16), "bf16"])
            for t in tf.tile(4, 1):
                out = tf.insert_slice(out, y, (t, 0, 0))
            return out


@module(entry="f", target=_H200, topologies=(Topology("cta", 4),))
class PlacedWindow:
    """Offsets naming the mesh coordinate give each position a window of its own."""

    @func
    def f(y: Tensor[(1, 8, 16), "bf16"]) -> Tensor[(4, 8, 16), "bf16"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            out = tf.zeros(Tensor[(4, 8, 16), "bf16"])
            return tf.insert_slice(out, y, (cta.tile, 0, 0))
