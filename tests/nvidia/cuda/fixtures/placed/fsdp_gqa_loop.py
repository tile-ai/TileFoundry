"""Two FSDP attention layers in a loop on two H200s; repeat_interleave expands each K/V head.

The tokens are split over the GPUs and K/V are gathered to Broadcast, so the
heads that repeat_interleave expands are the same on both GPUs, and so is their
gradient once it is reduced.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

L, T, C, H, KV, D = 2, 8, 16, 2, 1, 8
target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("cp",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


X, Mask = placed((T, C), Split(0)), placed((1, 1, T, T), Split(2))
FullK = placed((1, T, KV, D), Broadcast())
Q, Kv, O = placed((L, H * D, C), Split(1)), placed((L, KV * D, C), Split(1)), placed((L, C, H * D), Split(1))
FQ, FKv, FO = placed((H * D, C), Broadcast()), placed((KV * D, C), Broadcast()), placed((C, H * D), Broadcast())


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class FsdpGqaLoop:
    @func
    def layer(x: X, wq: FQ, wk: FKv, wv: FKv, wo: FO, mask: Mask):
        q = tf.transpose(tf.reshape(tf.matmul(x, wq, b_layout="NK"), new_shape=(1, T, H, D)), perm=(0, 2, 1, 3))
        k = tf.reshape(tf.matmul(x, wk, b_layout="NK"), new_shape=(1, T, KV, D))
        v = tf.reshape(tf.matmul(x, wv, b_layout="NK"), new_shape=(1, T, KV, D))
        k_all = tf.reshard(k, layout=FullK.layout, storage="gmem")
        v_all = tf.reshard(v, layout=FullK.layout, storage="gmem")
        kh = tf.repeat_interleave(tf.transpose(k_all, perm=(0, 2, 1, 3)), repeats=H // KV, axis=1)
        vh = tf.repeat_interleave(tf.transpose(v_all, perm=(0, 2, 1, 3)), repeats=H // KV, axis=1)
        p = tf.softmax(tf.matmul(q, kh, b_layout="NK") * 0.35355339 + mask, axis=-1)
        att = tf.reshape(tf.transpose(tf.matmul(p, vh), perm=(0, 2, 1, 3)), new_shape=(T, H * D))
        return x + tf.matmul(att, wo, b_layout="NK")

    @func
    def forward(x: X, wq: Q, wk: Kv, wv: Kv, wo: O, mask: Mask):
        h = x
        for t in range(L):
            i = tf.reshape(t, new_shape=(1,))
            fq = tf.reshard(tf.reshape(tf.index_select(wq, i, dim=0), new_shape=(H * D, C)), layout=FQ.layout, storage="gmem")
            fk = tf.reshard(tf.reshape(tf.index_select(wk, i, dim=0), new_shape=(KV * D, C)), layout=FKv.layout, storage="gmem")
            fv = tf.reshard(tf.reshape(tf.index_select(wv, i, dim=0), new_shape=(KV * D, C)), layout=FKv.layout, storage="gmem")
            fo = tf.reshard(tf.reshape(tf.index_select(wo, i, dim=0), new_shape=(C, H * D)), layout=FO.layout, storage="gmem")
            h = layer(h, fq, fk, fv, fo, mask)
        return tf.reduce(h * h, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
