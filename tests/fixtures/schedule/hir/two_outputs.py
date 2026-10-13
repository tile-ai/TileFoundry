"""Lower two differently shaped return leaves to separate output buffers (#226)."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

TILE = Layout((64, 64), (64, 1))
COLUMN = Layout((64, 1), (1, 1))


@module(
    entry="pair",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class TwoOutputs:
    @func
    def pair(
        x: Tensor[(64, 64), "f32"],
    ) -> tuple[Tensor[(64, 64), "f32", "umat"], Tensor[(64, 1), "f32", "umat"]]:
        with Mesh(("cta",), layout=(1,), names=("one",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                tile_dst = tf.empty(Tensor[(64, 64), "f32", TILE, "smem"])
                tile = tf.schedule((x[0:64, :], tile_dst), op=T.copy())
                column_dst = tf.empty(Tensor[(64, 1), "f32", COLUMN, "smem"])
                column = tf.schedule((x[0:64, 0:1], column_dst), op=T.copy())
                return tile, column
