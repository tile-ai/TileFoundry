"""Sparse decode with one natural-order key tile shared by QK and PV.

The loader's row loop carries both shared tiles. WGMMA's factored operands
retain row-major logical numbering: P is copied directly to the RS fragment,
and Q, K, V and the mask keep natural head and key numbering.
The register copy into the RS fragment requires redistribution; this fixture
checks TIR and evaluation, not the missing cross-thread shuffle codegen.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

NEG_INF = float("-inf")

NATURAL = ComposedLayout(Swizzle(3, 4, 3), 0, Layout((64, 64), (64, 1)))
ROW = Layout((1, 64), (64, 1))
Q_GROUPED = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))))
K_TRANSPOSED = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))))
V_GROUPED = ComposedLayout(
    Swizzle(3, 4, 3), 0, Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1)))
)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class SparseDecode:
    @func
    def gemm(
        q: Tensor[(1, 2, 128, 64), "bf16"],
        kv: Tensor[(1, 256, 1, 64), "bf16"],
        idx: Tensor[(64,), "i64"],
    ) -> Tensor[(64, 64), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            t = 1
            h0 = 32
            query = tf.reshape(q[0:1, t : t + 1, h0 : h0 + 64, :], (64, 64))
            table = tf.reshape(kv, (256, 64))
            with Mesh(("thread",), layout=(2, 128), names=("role", "participant")) as th:
                with th[0, :32] as _loader:
                    qs = tf.zeros(Tensor[(64, 64), "bf16", NATURAL, "smem"])
                    ks = tf.zeros(Tensor[(64, 64), "bf16", NATURAL, "smem"])
                    for r in range(64):
                        row_index = idx[r : r + 1]
                        with th[0, :32] as _carry:
                            qs_dst = tf.empty(
                                Tensor[
                                    (1, 64), "bf16",
                                    (1, 64),
                                    "smem",
                                ],
                            )
                            qs = tf.insert_slice(
                                qs,
                                tf.schedule(
                                    (query[r : r + 1, :], qs_dst), op=T.copy_async()
                                ),
                                (r, 0),
                            )
                            ks_dst = tf.empty(
                                Tensor[
                                    (1, 64), "bf16",
                                    (1, 64),
                                    "smem",
                                ],
                            )
                            ks = tf.insert_slice(
                                ks,
                                tf.schedule(
                                    (table, ks_dst, row_index), op=T.copy_async(fill=0)
                                ),
                                (r, 0),
                            )
                output = tf.zeros(Tensor[(64, 64), "bf16"])
                with Mesh(
                    th[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as m:
                    qg = tf.bitcast(qs, Q_GROUPED)
                    kt = tf.bitcast(ks, K_TRANSPOSED)
                    s = tf.zeros(Tensor[
                        (64, 64), "f32",
                        ((4 @ m.warp, 2, 8 @ m.lane8, 8, 4 @ m.lane4, 2,),
                         (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ])
                    s = tf.schedule(
                        (s, qg, kt),
                        op=T.tiled_mma(
                            atom=T.cuda.sm90.Wgmma(
                                n=64,
                                dtype="bf16",
                                form=T.cuda.sm90.Form.SS,
                                a_major=T.cuda.sm90.Major.K,
                                b_major=T.cuda.sm90.Major.K,
                            )
                        ),
                    )
                    ii_dst = tf.empty(
                        Tensor[
                            (64,), "i64",
                            (64,),
                            "rmem",
                        ],
                    )
                    ii = tf.schedule((idx, ii_dst), op=T.copy())
                    ii = ii + 0
                    live = tf.binary(ii >= 0, ii < 256, kind="and")
                    live = tf.bitcast(live, Layout(((8, 8),), ((8, 1),)))
                    live = tf.reshape(live, (1, 64))
                    s = tf.where(live, s, NEG_INF)
                    peak = tf.reduce(s, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                    p = tf.exp(s - peak)
                    denom = tf.reduce(p, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                    p = tf.cast(p / denom, "bf16")
                    pr_dst = tf.empty(
                        Tensor[
                            (64, 64), "bf16",
                            ShardLayout(
                                layout=Layout((4, 2, 8, 4, 2, 4, 2,), (16, 8, 1, 1024, 512, 128, 64)),
                                attrs=(S(0), S(2), S(5)),
                                mesh=m,
                            ),
                            "rmem",
                        ],
                    )
                    pr = tf.schedule(
                        (p, pr_dst),
                        op=T.copy(),
                    )
                    v = tf.bitcast(ks, V_GROUPED)
                    o = tf.zeros(Tensor[
                        (64, 64), "f32",
                        ((4 @ m.warp, 2, 8 @ m.lane8, 8, 4 @ m.lane4, 2,),
                         (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ])
                    o = tf.schedule(
                        (o, pr, v),
                        op=T.tiled_mma(
                            atom=T.cuda.sm90.Wgmma(
                                n=64,
                                dtype="bf16",
                                form=T.cuda.sm90.Form.RS,
                                b_major=T.cuda.sm90.Major.MN,
                            )
                        ),
                    )
                    result = tf.cast(o, "bf16")
                    result = tf.insert_slice(output, result, (0, 0))
        return result
