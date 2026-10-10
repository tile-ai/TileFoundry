"""A reduced sparse decode: gather, masked softmax(Q K^T), then P V.

The query is a strided four-dimensional window, indices use i64 arithmetic,
and a nested row loop carries two shared tiles. No instruction is selected.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, ReduceKind, Tensor, Topology, tf
from tilefoundry.ir.types import Layout
from tilefoundry.target import CudaTarget

NEG_INF = float("-inf")


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
        with Mesh(("cta",), layout=(1,), names=("block",)) as cta:
            t = 1
            h0 = 32
            query = tf.reshape(q[0:1, t : t + 1, h0 : h0 + 64, :], (64, 64))
            table = tf.reshape(kv, (256, 64))
            rows = tf.index_select(table, idx, fill_value=0)
            qs = tf.zeros(Tensor[(64, 64), "bf16", Layout((64, 64), (64, 1)), "smem"])
            ks = tf.zeros(Tensor[(64, 64), "bf16", Layout((64, 64), (64, 1)), "smem"])
            with cta[:] as _rows:
                for r in range(64):
                    qr = tf.reshard(query[r : r + 1, :], (1, 64), "smem")
                    kr = tf.reshard(rows[r : r + 1, :], (1, 64), "smem")
                    with cta[:] as _row:
                        qs = tf.insert_slice(qs, qr, (r, 0))
                        ks = tf.insert_slice(ks, kr, (r, 0))
            kt = tf.bitcast(ks, Layout((64, 64), (1, 64)))
            s = tf.matmul(qs, kt, out_dtype="f32")
            ii = tf.reshard(idx, (64,), "rmem") + 0
            live = tf.binary(ii >= 0, ii < 256, kind="and")
            live = tf.reshape(live, (1, 64))
            s = tf.where(live, s, NEG_INF)
            peak = tf.reduce(s, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
            p = tf.exp(s - peak)
            denom = tf.reduce(p, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
            p = tf.cast(p / denom, "bf16")
            v = tf.reshard(ks, Layout(((8, 8), (8, 8)), ((64, 512), (1, 8))), "smem")
            result = tf.cast(tf.matmul(p, v, out_dtype="f32"), "bf16")
        return result
