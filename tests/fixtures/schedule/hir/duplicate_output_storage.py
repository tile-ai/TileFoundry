"""Repeat an output leaf and reuse an induction name in a second loop."""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class LoopOutputs:
    @func
    def gemm(
        x: Tensor[(2, 4), "bf16"],
    ) -> tuple[Tensor[(2, 4), "bf16"], Tensor[(2, 4), "bf16"], Tensor[(2, 4), "bf16"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                direct = tf.zeros(Tensor[(2, 4), "bf16"])
                for row in range(2):
                    landing = tf.empty(Tensor[(1, 4), "bf16", Layout((1, 4), (4, 1)), "smem"])
                    update = tf.schedule((x[row : row + 1, :], landing), op=T.copy_async())
                    direct = tf.insert_slice(direct, update, (row, 0))
                scratch = x
                for row in range(2):
                    scratch = tf.zeros(Tensor[(2, 4), "bf16"])
                    landing = tf.empty(Tensor[(1, 4), "bf16", Layout((1, 4), (4, 1)), "smem"])
                    update = tf.schedule((x[row : row + 1, :], landing), op=T.copy_async())
                    scratch = tf.insert_slice(scratch, update, (row, 0))
                return direct, scratch, direct
