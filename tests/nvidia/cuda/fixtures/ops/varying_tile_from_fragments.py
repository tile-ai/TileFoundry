"""Each CTA multiplies its own rows, then its threads stage their register slices in shared memory.

The rows are a window at a per-CTA offset, so the loaded tile, the accumulator
and the staged tile are Varying on ``row`` (one per CTA). Their meshes hold the
CTA topology; for the threads that issue the WGMMA and the copy, each is the
tile of their own CTA. Each thread writes the staged bytes of its own register
slice, as the accumulator's Split says.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

Q_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), (3, 4, 16)), ((512, 64), (4096, 16, 1))))
K_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((12, 8), (3, 4, 16)), ((512, 64), (6144, 16, 1))))
K_T16 = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((3, 4, 16), (2, 3, 2, 8)), ((6144, 16, 1), (3072, 1024, 512, 64))))


@module(entry="gemm", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 2), Topology("thread", 384)))
class TileFromFragments:
    @func
    def gemm(q: Tensor[(128, 192), "bf16"], k: Tensor[(96, 192), "bf16"]) -> Tensor[(128, 96), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("row",)) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                out = tf.zeros(Tensor[(128, 96), "f32"])
                with threads[0, :32] as _loader:
                    q_s = tf.schedule((q[cta.row * 64:cta.row * 64 + 64, :], tf.empty(Tensor[(64, 192), "bf16", Q_SMEM @ {cta.row @ V()}, "smem"])), op=T.copy_async_tensor())
                    k_s = tf.schedule((k, tf.empty(Tensor[(96, 192), "bf16", K_SMEM, "smem"])), op=T.copy_async_tensor())
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as cg:
                    s = tf.schedule(
                        (tf.zeros(Tensor[(64, 96), "f32", ((4 @ cg.warp, 2, 8 @ cg.lane8, 2 @ cg.group, 3, 2, 4 @ cg.lane4, 2), (16, 8, 1, 3072, 1024, 512, 128, 64)) @ {cta.row @ V()}, "rmem"]),
                         q_s, tf.bitcast(k_s, K_T16)),
                        op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)),
                    )
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _epilogue:
                    staged = tf.schedule((s, tf.empty(Tensor[(64, 96), "f32", (64, 96) @ {cta.row @ V()}, "smem"])), op=T.copy())
                with threads[0, :32] as _storer:
                    stored = tf.schedule((staged, tf.empty(Tensor[(64, 96), "f32", (64, 96)])), op=T.copy())
                    return tf.insert_slice(out, stored, (cta.row * 64, 0))
