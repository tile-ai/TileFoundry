"""Untiled baseline retained for the original placement regression workflow."""
from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 512)))
class GEMM_RELU_GEMM_UNTILED:
    @func
    def gemm(a: Tensor[(1024, 2048), "bf16"],
             b: Tensor[(2048, 2048), "bf16"],
             c: Tensor[(2048, 2048), "bf16"]) -> Tensor[(1024, 2048), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            x = tf.matmul(a, b)
            y = tf.relu(x)
            return tf.matmul(y, c)
