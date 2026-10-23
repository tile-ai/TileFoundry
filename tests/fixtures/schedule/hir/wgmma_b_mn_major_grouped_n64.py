"""Preserve MN-major B unit modes when two groups issue n64 WGMMA tiles."""
from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget

A = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), 16), ((512, 64), 1)))
B = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))))


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 256)))
class GroupedMnMajorB:
    @func
    def gemm(a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 128), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as threads:
                lhs = tf.schedule((a, tf.empty(Tensor[(64, 16), "bf16", A, "smem"])), op=T.copy())
                rhs = tf.schedule((b, tf.empty(Tensor[(16, 128), "bf16", B, "smem"])), op=T.copy())
                acc = tf.zeros(Tensor[(64, 128), "f32", ((4 @ threads.warp, 2, 8 @ threads.lane8, 2 @ threads.group, 8, 4 @ threads.lane4, 2), (16, 8, 1, 4096, 512, 128, 64)), "rmem"])
                result = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)))
                out = tf.zeros(Tensor[(64, 128), "f32"])
                return tf.insert_slice(out, result, (0, 0))
