"""Megatron tensor parallel on two GPUs: column-split q/k/v/gate/up, row-split o/down.

RMSNorm and SoftMax read Broadcast values whose gradient a column-split matmul makes Partial.
"""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

T, C, H, D, F = 8, 16, 4, 4, 32
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("tp",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


X = placed((T, C), Broadcast())
Norm = placed((C,), Broadcast())
Col, Row = placed((H * D, C), Split(0)), placed((C, H * D), Split(1))
ColF, RowF = placed((F, C), Split(0)), placed((C, F), Split(1))
Mask = placed((1, T, T), Broadcast())
Full = placed((T, C), Broadcast())


@module(entry="forward", target=target, topologies=(gpu,))
class TensorParallel:
    @func
    def forward(x: X, wn1: Norm, wq: Col, wk: Col, wv: Col, wo: Row, wn2: Norm, wg: ColF, wu: ColF, wd: RowF,
                mask: Mask):
        h = tf.rms_norm(x, wn1, eps=1e-6)
        q = tf.transpose(tf.reshape(tf.matmul(h, wq, b_layout="NK"), new_shape=(T, H, D)), perm=(1, 0, 2))
        k = tf.transpose(tf.reshape(tf.matmul(h, wk, b_layout="NK"), new_shape=(T, H, D)), perm=(1, 0, 2))
        v = tf.transpose(tf.reshape(tf.matmul(h, wv, b_layout="NK"), new_shape=(T, H, D)), perm=(1, 0, 2))
        p = tf.softmax(tf.matmul(q, k, b_layout="NK") * 0.5 + mask, axis=-1)
        att = tf.reshape(tf.transpose(tf.matmul(p, v), perm=(1, 0, 2)), new_shape=(T, H * D))
        o = tf.reshard(tf.matmul(att, wo, b_layout="NK"), layout=Full.layout, storage="gmem")
        x1 = x + o
        h2 = tf.rms_norm(x1, wn2, eps=1e-6)
        m = tf.silu(tf.matmul(h2, wg, b_layout="NK")) * tf.matmul(h2, wu, b_layout="NK")
        d = tf.reshard(tf.matmul(m, wd, b_layout="NK"), layout=Full.layout, storage="gmem")
        return x1 + d
