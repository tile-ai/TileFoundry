"""Admit an MN-major B descriptor with a wide leading byte offset."""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

A_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), 16), ((512, 64), 1)))
V_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout((32, (2, 64)), (64, (2048, 1))))
V_TOP = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((2, 8), (2, 64)), ((512, 64), (2048, 1))))

@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 128)))
class MnMajorB:
    @func
    def gemm(a: Tensor[(64, 16), "f16"], v: Tensor[(32, 128), "f16"]) -> Tensor[(64, 128), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as threads:
                output = tf.zeros(Tensor[(64, 128), "f32"])
                lhs = tf.schedule((a, tf.empty(Tensor[(64, 16), "f16", A_SMEM, "smem"])), op=T.copy())
                rows = tf.schedule((v, tf.empty(Tensor[(32, 128), "f16", V_SMEM, "smem"])), op=T.copy())
                acc = tf.zeros(Tensor[(64, 128), "f32",
                    ((4 @ threads.warp, 2, 8 @ threads.lane8, 16, 4 @ threads.lane4, 2), (16, 8, 1, 512, 128, 64)), "rmem"])
                result = tf.schedule(
                    (acc, lhs, tf.bitcast(rows[0:16, :], V_TOP)),
                    op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(n=128, dtype="f16", form=T.cuda.sm90.Form.SS,
                                                         a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)))
                return tf.insert_slice(output, result, (0, 0))
