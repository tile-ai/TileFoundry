"""Slice two swizzle atoms of an MN-major WGMMA B operand."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

K_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((6, 2, 8), (3, 1, 64)), ((1024, 512, 64), (6144, 0, 1))))
DS_SMEM = ComposedLayout(Swizzle(1, 4, 3), 0, Layout(((8, 8), (6, 1, 16)), ((128, 16), (1024, 1024, 1))))


def _wg(n):
    return T.cuda.sm90.Wgmma(n=n, dtype="bf16", form=T.cuda.sm90.Form.SS,
                             a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 256)))
class Wide:
    @func
    def gemm(ds: Tensor[(64, 96), "bf16"], k: Tensor[(96, 192), "bf16"]) -> Tensor[(64, 128), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 128), names=("warpgroup", "participant")) as threads:
                with threads[0, :32] as _loader:
                    ds_s = tf.schedule((ds, tf.empty(Tensor[(64, 96), "bf16", DS_SMEM, "smem"])), op=T.copy_async_tensor())
                    k_s = tf.schedule((k, tf.empty(Tensor[(96, 192), "bf16", K_SMEM, "smem"])), op=T.copy_async_tensor())
                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as g:
                    acc = tf.schedule(
                        (tf.zeros(Tensor[(64, 128), "f32", ((4 @ g.warp, 2, 8 @ g.lane8, 16, 4 @ g.lane4, 2), (16, 8, 1, 512, 128, 64)), "rmem"]),
                         ds_s, k_s[:, 0:128]),
                        op=T.tiled_mma(atom=_wg(128)),
                    )
                    return tf.schedule((acc, tf.empty(Tensor[(64, 128), "f32", (64, 128)])), op=T.copy())
