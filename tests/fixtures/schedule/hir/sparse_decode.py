"""Sparse decode with one natural-order key tile shared by QK and PV.

The loader's row loop carries both shared tiles. WGMMA's grouped operands
renumber dimensions, heads and keys: masks follow QK's key numbering, P's
bitcast follows V's numbering, and O's bitcast restores natural head numbering.
The register copy into the RS fragment requires redistribution; this fixture
checks TIR and evaluation, not the missing cross-thread shuffle codegen.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, ReduceKind, T, Tensor, Topology, tf
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split, Swizzle
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

NEG_INF = float("-inf")

_COMPUTE = ThreadMesh(
    (Topology("thread", 256),),
    ComposedLayout(None, 128, Layout((4, 8, 4), (32, 4, 1))),
    ("warp", "lane8", "lane4"),
)
ACC = ShardLayout(
    Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), (Split(2), Split(0), Split(4)), _COMPUTE
)
NATURAL = ComposedLayout(Swizzle(3, 4, 3), 0, Layout((64, 64), (64, 1)))
ROW = Layout((1, 64), (64, 1))
Q_GROUPED = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))))
K_TRANSPOSED = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))))
V_GROUPED = ComposedLayout(
    Swizzle(3, 4, 3), 0, Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1)))
)
P_V_KEYS = ShardLayout(
    Layout((8, 2, 4, 4, 2, 8), (1, 8, 16, 128, 64, 512)), (Split(2), Split(0), Split(3)), _COMPUTE
)
P_RS = ShardLayout(
    Layout((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
    (Split(3), Split(1), Split(5)),
    _COMPUTE,
)
O_NATURAL = ShardLayout(
    Layout((2, 4, 8, 2, 4, 8), (8, 16, 1, 64, 128, 512)), (Split(1), Split(2), Split(4)), _COMPUTE
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
                            qs = tf.insert_slice(
                                qs,
                                tf.schedule(
                                    (query[r : r + 1, :],), op=T.copy_async(smem_layout=ROW)
                                ),
                                (r, 0),
                            )
                            ks = tf.insert_slice(
                                ks,
                                tf.schedule(
                                    (table, row_index), op=T.copy_async(smem_layout=ROW, fill=0)
                                ),
                                (r, 0),
                            )
                with th[1, :] as _compute:
                    qg = tf.bitcast(qs, Q_GROUPED)
                    kt = tf.bitcast(ks, K_TRANSPOSED)
                    s = tf.zeros(Tensor[(64, 64), "f32", ACC, "rmem"])
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
                    ii = tf.schedule((idx,), op=T.copy(rmem_layout=Layout((64,), (1,))))
                    ii = ii + 0
                    live = tf.binary(ii >= 0, ii < 256, kind="and")
                    live = tf.bitcast(live, Layout(((8, 8),), ((8, 1),)))
                    live = tf.reshape(live, (1, 64))
                    s = tf.where(live, s, NEG_INF)
                    peak = tf.reduce(s, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                    p = tf.exp(s - peak)
                    denom = tf.reduce(p, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                    p = tf.cast(p / denom, "bf16")
                    pv = tf.bitcast(p, P_V_KEYS)
                    pr = tf.schedule((pv,), op=T.copy(rmem_layout=P_RS))
                    ks_view = tf.reshard(
                        ks,
                        ComposedLayout(
                            Swizzle(3, 4, 3), 0, Layout(((8, 8), (8, 8)), ((64, 512), (1, 8)))
                        ),
                        "smem",
                    )
                    v = tf.bitcast(ks_view, V_GROUPED)
                    o = tf.zeros(Tensor[(64, 64), "f32", ACC, "rmem"])
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
                    out = tf.bitcast(o, O_NATURAL)
                    result = tf.cast(out, "bf16")
        return result
