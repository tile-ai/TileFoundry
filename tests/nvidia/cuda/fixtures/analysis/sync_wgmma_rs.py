"""An RS WGMMA issues four times and shares A with an ordinary register copy."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

V_SMEM = ComposedLayout(
    Swizzle(3, 4, 3), 0, Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1)))
)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class WgmmaRs:
    @func
    def gemm(
        p: Tensor[(64, 64), "bf16"], v: Tensor[(64, 64), "bf16"]
    ) -> tuple[Tensor[(64, 64), "f32", "umat"], Tensor[(64, 64), "bf16", "umat"]]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as m:
                a_fragment = ShardLayout(
                    layout=Layout((4, 2, 8, 4, 2, 4, 2), (16, 8, 1, 1024, 512, 128, 64)),
                    attrs=(S(0), S(2), S(5)),
                    mesh=m,
                )
                pr = tf.schedule(
                    (p, tf.empty(Tensor[(64, 64), "bf16", a_fragment, "rmem"])), op=T.copy()
                )
                kept = tf.zeros(Tensor[(64, 64), "bf16", a_fragment, "rmem"])
                vs = tf.schedule(
                    (v, tf.empty(Tensor[(64, 64), "bf16", V_SMEM, "smem"])), op=T.copy()
                )
                o = tf.zeros(Tensor[
                    (64, 64), "f32",
                    ((4 @ m.warp, 2, 8 @ m.lane8, 8, 4 @ m.lane4, 2,), (16, 8, 1, 512, 128, 64)),
                    "rmem",
                ])
                for _k in range(2):
                    o = tf.schedule(
                        (o, pr, vs),
                        op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(
                            n=64, dtype="bf16", form=T.cuda.sm90.Form.RS,
                            b_major=T.cuda.sm90.Major.MN,
                        )),
                    )
                    kept = tf.schedule((pr, kept), op=T.copy())
                output = tf.zeros(Tensor[(64, 64), "f32"])
                copied = tf.zeros(Tensor[(64, 64), "bf16"])
                return tf.insert_slice(output, o, (0, 0)), tf.insert_slice(copied, kept, (0, 0))
