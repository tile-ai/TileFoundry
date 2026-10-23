"""Write three output buffers explicitly and reuse an induction name."""

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
                scratch = tf.empty(Tensor[(2, 4), "bf16"])
                scratch = tf.schedule((x, scratch), op=T.copy())
                for row in range(2):
                    temporary = tf.zeros(Tensor[(2, 4), "bf16"])
                    landing = tf.empty(Tensor[(1, 4), "bf16", Layout((1, 4), (4, 1)), "smem"])
                    update = tf.schedule((x[row : row + 1, :], landing), op=T.copy_async())
                    temporary = tf.insert_slice(temporary, update, (row, 0))
                    scratch = tf.schedule((temporary, scratch), op=T.copy())
                third = tf.empty(Tensor[(2, 4), "bf16"])
                third = tf.schedule((direct, third), op=T.copy())
                return direct, scratch, third
