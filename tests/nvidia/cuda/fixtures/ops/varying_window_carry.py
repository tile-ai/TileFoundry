"""Each CTA copies the rows at its own offset into a shared-memory loop carry.

The window starts at ``cta.c * 64``, so it differs per CTA (Varying on ``c``).
A carry declared plain says every CTA holds the same tile: that is an error.
Written ``@ {cta.c @ V()}``, the carry states one tile per CTA and loads.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

TOPOLOGIES = (Topology("cta", 2), Topology("thread", 32))


@module(entry="run", target=cuda.Target("nvidia.h200_sxm"), topologies=TOPOLOGIES)
class WindowCarryBroadcast:
    @func
    def run(x: Tensor[(128, 64), "f32"]) -> Tensor[(128, 64), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("c",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _lanes:
                out = tf.empty(Tensor[(128, 64), "f32"])
                tile = tf.empty(Tensor[(32, 64), "f32", (32, 64), "smem"])
                for k in range(2):
                    row = cta.c * 64 + k * 32
                    tile = tf.schedule((x[row:row + 32, :], tile), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (row, 0))
                return out


@module(entry="run", target=cuda.Target("nvidia.h200_sxm"), topologies=TOPOLOGIES)
class WindowCarryVarying:
    @func
    def run(x: Tensor[(128, 64), "f32"]) -> Tensor[(128, 64), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("c",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _lanes:
                out = tf.empty(Tensor[(128, 64), "f32"])
                tile = tf.empty(Tensor[(32, 64), "f32", (32, 64) @ {cta.c @ V()}, "smem"])
                for k in range(2):
                    row = cta.c * 64 + k * 32
                    tile = tf.schedule((x[row:row + 32, :], tile), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (row, 0))
                return out
