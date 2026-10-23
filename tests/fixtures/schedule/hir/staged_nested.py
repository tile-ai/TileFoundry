"""Own a staged ring outside both HIR loops and index by linear iteration."""
from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)))
class StagedNested:
    @func
    def gemm(x: Tensor[(6, 8), "bf16"]) -> Tensor[(6, 8), "bf16", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                acc = tf.zeros(Tensor[(6, 8), "bf16", "rmem"])
                for i in range(2):
                    for j in range(3):
                        ring = tf.empty(Tensor[(1, 8), "bf16", Layout((1, 8), (8, 1)), "smem"], buffers=2)
                        row = tf.schedule((x[3 * i + j:3 * i + j + 1, :], ring), op=T.copy_async())
                        held = tf.schedule((row, tf.empty(Tensor[(1, 8), "bf16", (1, 8), "rmem"])), op=T.copy())
                        acc = tf.insert_slice(acc, held, (3 * i + j, 0))
                return acc
