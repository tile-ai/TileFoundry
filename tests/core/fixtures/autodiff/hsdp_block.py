"""One examples/fsdp_training block on a (replicate, shard) mesh of four GPUs."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

B, T, C, H, G, D, F = 8, 8, 16, 4, 2, 4, 64
N = B * T
EPS = 1e-6
target = cpu.Target()
gpu = Topology("gpu", 4)
mesh = Mesh((gpu,), Layout((2, 2)), ("replicate", "shard"))


def placed(shape, attrs=(Broadcast(), Split(0))):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, attrs, storage="gmem", owners=target.storage_owners), "gmem")


WHOLE = (Broadcast(), Broadcast())
A = placed((N, C), (Split(0), Split(0)))
Square, FullSquare = placed((C, C)), placed((C, C), WHOLE)
Kv, FullKv = placed((G * D, C)), placed((G * D, C), WHOLE)
Up, FullUp = placed((F, C)), placed((F, C), WHOLE)
Down, FullDown = placed((C, F)), placed((C, F), WHOLE)
Rotation = placed((1, 1, 1, T, D, D), WHOLE)
Mask = placed((1, 1, 1, T, T), WHOLE)


@module(entry="forward", target=target, topologies=(gpu,))
class HsdpBlock:
    @func
    def forward(x: A, wq: Square, wk: Kv, wv: Kv, wo: Square, wu: Up, wd: Down, rotation: Rotation, mask: Mask):
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
