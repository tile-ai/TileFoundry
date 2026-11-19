"""The examples/fsdp_training model on two H200s: embedding, two GQA blocks, head, cross entropy.

Each block gathers its Split(0) weights to Broadcast; the embedding is a one-hot matmul.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

B, T, C, H, G, D, F, V = 8, 8, 16, 4, 2, 4, 64, 32
N = B * T
EPS = 1e-6
MEAN = 1.0 / N
target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("shard",))


def placed(shape, attr=Split(0)):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


A, Tokens = placed((N, C)), placed((N, V))
Emb, FullEmb = placed((V, C)), placed((V, C), Broadcast())
Square, FullSquare = placed((C, C)), placed((C, C), Broadcast())
Kv, FullKv = placed((G * D, C)), placed((G * D, C), Broadcast())
Up, FullUp = placed((F, C)), placed((F, C), Broadcast())
Down, FullDown = placed((C, F)), placed((C, F), Broadcast())
Rotation = placed((1, 1, 1, T, D, D), Broadcast())
Mask = placed((1, 1, 1, T, T), Broadcast())


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class FsdpModel:
    @func
    def block(x: A, wq: Square, wk: Kv, wv: Kv, wo: Square, wu: Up, wd: Down, rotation: Rotation, mask: Mask):
        q_full = tf.reshard(wq, layout=FullSquare.layout, storage="gmem")
        k_full = tf.reshard(wk, layout=FullKv.layout, storage="gmem")
        v_full = tf.reshard(wv, layout=FullKv.layout, storage="gmem")
        o_full = tf.reshard(wo, layout=FullSquare.layout, storage="gmem")
        u_full = tf.reshard(wu, layout=FullUp.layout, storage="gmem")
        d_full = tf.reshard(wd, layout=FullDown.layout, storage="gmem")
        ix = tf.rsqrt(tf.reduce(x * x, axes=(-1,)) + EPS)
        a = x * ix
        q0 = tf.transpose(tf.reshape(tf.matmul(a, q_full, b_layout="NK"), (B, T, G, H // G, D)), perm=(0, 2, 3, 1, 4))
        k0 = tf.transpose(tf.reshape(tf.matmul(a, k_full, b_layout="NK"), (B, T, G, 1, D)), perm=(0, 2, 3, 1, 4))
        v = tf.transpose(tf.reshape(tf.matmul(a, v_full, b_layout="NK"), (B, T, G, 1, D)), perm=(0, 2, 3, 1, 4))
        qn = q0 * tf.rsqrt(tf.reduce(q0 * q0, axes=(-1,)) + EPS)
        kn = k0 * tf.rsqrt(tf.reduce(k0 * k0, axes=(-1,)) + EPS)
        q = tf.reshape(tf.matmul(tf.reshape(qn, (B, G, H // G, T, 1, D)), rotation), (B, G, H // G, T, D))
        k = tf.reshape(tf.matmul(tf.reshape(kn, (B, G, 1, T, 1, D)), rotation), (B, G, 1, T, D))
        probability = tf.softmax(tf.matmul(q, k, b_layout="NK") * 0.5 + mask, axis=-1)
        attended = tf.reshape(tf.transpose(tf.matmul(probability, v), perm=(0, 3, 1, 2, 4)), (N, C))
        residual = x + tf.matmul(attended, o_full, b_layout="NK")
        r = residual * tf.rsqrt(tf.reduce(residual * residual, axes=(-1,)) + EPS)
        relu = tf.relu(tf.matmul(r, u_full, b_layout="NK"))
        return residual + tf.matmul(relu * relu, d_full, b_layout="NK")

    @func
    def forward(tokens: Tokens, targets: Tokens, rotation: Rotation, mask: Mask, we: Emb,
                q1: Square, k1: Kv, v1: Kv, o1: Square, u1: Up, d1: Down,
                q2: Square, k2: Kv, v2: Kv, o2: Square, u2: Up, d2: Down, wh: Emb):
        x = tf.matmul(tokens, tf.reshard(we, layout=FullEmb.layout, storage="gmem"))
        y1 = block(x, q1, k1, v1, o1, u1, d1, rotation, mask)
        y2 = block(y1, q2, k2, v2, o2, u2, d2, rotation, mask)
        z = y2 * tf.rsqrt(tf.reduce(y2 * y2, axes=(-1,)) + EPS)
        logits = tf.matmul(z, tf.reshard(wh, layout=FullEmb.layout, storage="gmem"), b_layout="NK")
        shifted = logits - tf.reduce(logits, axes=(-1,), kind=ReduceKind.MAX)
        log_z = tf.log(tf.reduce(tf.exp(shifted), axes=(-1,), kind=ReduceKind.SUM))
        nll = -tf.reduce(targets * (shifted - log_z), axes=(-1,), kind=ReduceKind.SUM)
        return tf.reduce(nll * MEAN, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
