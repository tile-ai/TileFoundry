"""Two GQA blocks with explicit FSDP/HSDP training and parameter residency.

One-hot matrix multiplication expresses the small-vocabulary embedding lookup
and its gradient using the existing HIR operations.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.expr import DType
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry_nvidia import cuda

B, T, C, H, G, D, F, V = 8, 8, 16, 4, 2, 4, 64, 32
N = B * T
EPS = 1e-6
BLOCK_SHAPES = ((C, C), (G * D, C), (G * D, C), (C, C), (F, C), (C, F))
PARAMETER_SHAPES = ((V, C), *BLOCK_SHAPES, *BLOCK_SHAPES, (V, C))
PARAMETER_NAMES = ("embedding", *(f"block{i}.{name}" for i in (1, 2)
                   for name in ("q", "k", "v", "o", "up", "down")), "head")
SUPPORTED_MESHES = ((1, 1), (2, 1), (4, 1), (4, 2), (16, 4))


def training_program(participants=2, *, retain=False, replicas=1):
    """Build explicit forward, backward and momentum SGD for one training step.

    Args:
        participants: Logical GPU positions, each receiving complete sequences.
        retain: Keep gathered parameters through backward when true.
        replicas: Number of replicated shard groups.

    Returns:
        A module taking one-hot tokens/targets, fixed RoPE and causal mask,
        fourteen parameters and their momentum buffers. Its entry returns
        logits, global mean loss, averaged gradients, parameters and momentum.
        Updated state preserves the input distribution.
    """
    if (participants, replicas) not in SUPPORTED_MESHES:
        raise ValueError(f"supported (participants, replicas): {SUPPORTED_MESHES}")
    batch = max(B, participants)
    tokens_per_batch = batch * T
    target = cuda.Target("nvidia.h200_sxm", device_count=participants)
    gpu = Topology("gpu", participants)
    mesh_shape = (participants,) if replicas == 1 else (replicas, participants // replicas)
    mesh_names = ("shard",) if replicas == 1 else ("replicate", "shard")
    mesh_layout = Layout(mesh_shape)
    mesh = Mesh((gpu,), mesh_layout, mesh_names)

    def layout(shape, attr):
        if participants == 1:
            attr = Broadcast()
        attrs = attr if isinstance(attr, tuple) else ((attr,) if replicas == 1 else (Broadcast(), attr))
        return ShardLayout.from_logical(shape, mesh, attrs, storage="gmem", owners=target.storage_owners)

    def tensor(shape, attr=Split(0)):
        return TensorType(shape, DType.f32, layout(shape, attr), "gmem")

    data_attrs = (Split(0),) * len(mesh_shape) if participants > 1 else (Broadcast(),)
    A, Tokens = tensor((tokens_per_batch, C), data_attrs), tensor((tokens_per_batch, V), data_attrs)
    Emb, FullEmb = tensor((V, C)), tensor((V, C), Broadcast())
    Sharded = TupleType(tuple(tensor(shape) for shape in BLOCK_SHAPES))
    Full = TupleType(tuple(tensor(shape, Broadcast()) for shape in BLOCK_SHAPES))
    Rotation = tensor((1, 1, 1, T, D, D), Broadcast())
    Mask = tensor((1, 1, 1, T, T), Broadcast())
    full_e = FullEmb.layout
    fq, fk, fv, fo, fu, fd = (field.layout for field in Full.fields)
    sq, sk, sv, so, su, sd = (field.layout for field in Sharded.fields)
    reduced_attrs = (Partial("sum"), Split(0)) if replicas > 1 else Split(0)
    rq, rk, rv, ro, ru, rd = (layout(shape, reduced_attrs) for shape in BLOCK_SHAPES)
    reduced_e = layout((V, C), reduced_attrs)
    inverse_p = 1.0 / participants
    local_mean_scale = participants / tokens_per_batch
    global_mean_scale = 1.0 / tokens_per_batch
    Inverse = tensor((tokens_per_batch, 1), data_attrs)
    HeadWeight = FullEmb if retain else Emb
    shard_e = Emb.layout

    @module(entry="forward", target=target, topologies=(gpu,))
    class Block:
        @func
        def gather(w: Sharded):
            return (tf.reshard(w[0], layout=fq, storage="gmem"),
                    tf.reshard(w[1], layout=fk, storage="gmem"),
                    tf.reshard(w[2], layout=fv, storage="gmem"),
                    tf.reshard(w[3], layout=fo, storage="gmem"),
                    tf.reshard(w[4], layout=fu, storage="gmem"),
                    tf.reshard(w[5], layout=fd, storage="gmem"))

        @func
        def forward(x: A, w: Full, rotation: Rotation, mask: Mask):
            ix = tf.rsqrt(tf.reduce(x * x, axes=(-1,)) + EPS)
            a = x * ix
            q0 = tf.transpose(tf.reshape(tf.matmul(a, w[0], b_layout="NK"),
                                        (batch, T, G, H // G, D)), perm=(0, 2, 3, 1, 4))
            k0 = tf.transpose(tf.reshape(tf.matmul(a, w[1], b_layout="NK"),
                                        (batch, T, G, 1, D)), perm=(0, 2, 3, 1, 4))
            v = tf.transpose(tf.reshape(tf.matmul(a, w[2], b_layout="NK"),
                                       (batch, T, G, 1, D)), perm=(0, 2, 3, 1, 4))
            iq = tf.rsqrt(tf.reduce(q0 * q0, axes=(-1,)) + EPS)
            ik = tf.rsqrt(tf.reduce(k0 * k0, axes=(-1,)) + EPS)
            qn = q0 * iq
            kn = k0 * ik
            q = tf.reshape(tf.matmul(tf.reshape(qn, (batch, G, H // G, T, 1, D)), rotation),
                           (batch, G, H // G, T, D))
            k = tf.reshape(tf.matmul(tf.reshape(kn, (batch, G, 1, T, 1, D)), rotation),
                           (batch, G, 1, T, D))
            scores = tf.matmul(q, k, b_layout="NK") * 0.5 + mask
            probability = tf.softmax(scores, axis=-1)
            attended = tf.reshape(tf.transpose(tf.matmul(probability, v), perm=(0, 3, 1, 2, 4)), (tokens_per_batch, C))
            residual = x + tf.matmul(attended, w[3], b_layout="NK")
            ir = tf.rsqrt(tf.reduce(residual * residual, axes=(-1,)) + EPS)
            r = residual * ir
            relu = tf.relu(tf.matmul(r, w[4], b_layout="NK"))
            y = residual + tf.matmul(relu * relu, w[5], b_layout="NK")
            return y, a, ix, qn, iq, kn, ik, q, k, v, probability, attended, r, ir, relu

    block_forward = Block.forward
    gather = Block.gather
    Saved = block_forward.return_type

    @module(entry="backward", target=target, topologies=(gpu,))
    class Backward:
        @func
        def backward(dy: A, s: Saved, w: Full, rotation: Rotation):
            y, a, ix, qn, iq, kn, ik, q, k, v, probability, attended, r, ir, relu = s
            pd = tf.matmul(dy, relu * relu, a_layout="KM")
            reduced_d = tf.reshard(pd, layout=rd, storage="gmem")
            gd = tf.reshard(reduced_d, layout=sd, storage="gmem") * inverse_p
            du = tf.matmul(dy, w[5]) * (2.0 * relu)
            pu = tf.matmul(du, r, a_layout="KM")
            reduced_u = tf.reshard(pu, layout=ru, storage="gmem")
            gu = tf.reshard(reduced_u, layout=su, storage="gmem") * inverse_p
            dr = tf.matmul(du, w[4])
            dresidual = dy + ir * (dr - r * tf.reduce(dr * r, axes=(-1,)))
            po = tf.matmul(dresidual, attended, a_layout="KM")
            reduced_o = tf.reshard(po, layout=ro, storage="gmem")
            go = tf.reshard(reduced_o, layout=so, storage="gmem") * inverse_p
            dc = tf.transpose(tf.reshape(tf.matmul(dresidual, w[3]), (batch, T, G, H // G, D)),
                              perm=(0, 2, 3, 1, 4))
            dv = tf.reduce(tf.matmul(probability, dc, a_layout="KM"), axes=(2,), kind=ReduceKind.SUM)
            dp = tf.matmul(dc, v, b_layout="NK")
            ds = probability * (dp - tf.reduce(dp * probability, axes=(-1,), kind=ReduceKind.SUM))
            dq = tf.matmul(ds, k) * 0.5
            dk = tf.reduce(tf.matmul(ds, q, a_layout="KM"), axes=(2,), kind=ReduceKind.SUM) * 0.5
            dqn = tf.reshape(tf.matmul(tf.reshape(dq, (batch, G, H // G, T, 1, D)),
                                      rotation, b_layout="NK"), (batch, G, H // G, T, D))
            dkn = tf.reshape(tf.matmul(tf.reshape(dk, (batch, G, 1, T, 1, D)),
                                      rotation, b_layout="NK"), (batch, G, 1, T, D))
            dq0 = iq * (dqn - qn * tf.reduce(dqn * qn, axes=(-1,)))
            dk0 = ik * (dkn - kn * tf.reduce(dkn * kn, axes=(-1,)))
            dqf = tf.reshape(tf.transpose(dq0, perm=(0, 3, 1, 2, 4)), (tokens_per_batch, C))
            dkf = tf.reshape(tf.transpose(dk0, perm=(0, 3, 1, 2, 4)), (tokens_per_batch, G * D))
            dvf = tf.reshape(tf.transpose(dv, perm=(0, 3, 1, 2, 4)), (tokens_per_batch, G * D))
            pq = tf.matmul(dqf, a, a_layout="KM")
            pk = tf.matmul(dkf, a, a_layout="KM")
            pv = tf.matmul(dvf, a, a_layout="KM")
            reduced_q = tf.reshard(pq, layout=rq, storage="gmem")
            reduced_k = tf.reshard(pk, layout=rk, storage="gmem")
            reduced_v = tf.reshard(pv, layout=rv, storage="gmem")
            gq = tf.reshard(reduced_q, layout=sq, storage="gmem") * inverse_p
            gk = tf.reshard(reduced_k, layout=sk, storage="gmem") * inverse_p
            gv = tf.reshard(reduced_v, layout=sv, storage="gmem") * inverse_p
            da = tf.matmul(dqf, w[0]) + tf.matmul(dkf, w[1]) + tf.matmul(dvf, w[2])
            dx = dresidual + ix * (da - a * tf.reduce(da * a, axes=(-1,)))
            return dx, gq, gk, gv, go, gu, gd

    backward = Backward.backward

    @module(entry="release_forward", target=target, topologies=(gpu,))
    class Stages:
        @func
        def release_forward(x: A, w: Sharded, rotation: Rotation, mask: Mask):
            full = gather(w)
            y, a, ix, qn, iq, kn, ik, q, k, v, p, attended, r, ir, relu = block_forward(x, full, rotation, mask)
            return (y, a, ix, qn, iq, kn, ik, q, k, v, p, attended, r, ir, relu,
                    w[0], w[1], w[2], w[3], w[4], w[5])

        @func
        def retain_forward(x: A, w: Sharded, rotation: Rotation, mask: Mask):
            full = gather(w)
            y, a, ix, qn, iq, kn, ik, q, k, v, p, attended, r, ir, relu = block_forward(x, full, rotation, mask)
            return (y, a, ix, qn, iq, kn, ik, q, k, v, p, attended, r, ir, relu,
                    full[0], full[1], full[2], full[3], full[4], full[5])

        @func
        def release_backward(dy: A, s: Saved, w: Sharded, rotation: Rotation):
            full = gather(w)
            return backward(dy, s, full, rotation)

        @func
        def retain_backward(dy: A, s: Saved, w: Full, rotation: Rotation):
            return backward(dy, s, w, rotation)

        @func
        def embedding(tokens: Tokens, w: Emb):
            full = tf.reshard(w, layout=full_e, storage="gmem")
            return tf.matmul(tokens, full)

        @func
        def head_forward(x: A, w: Emb):
            full = tf.reshard(w, layout=full_e, storage="gmem")
            inverse = tf.rsqrt(tf.reduce(x * x, axes=(-1,)) + EPS)
            z = x * inverse
            logits = tf.matmul(z, full, b_layout="NK")
            return logits, z, inverse, full

        @func
        def release_head(x: A, w: Emb):
            logits, z, inverse, full = head_forward(x, w)
            return logits, z, inverse, w

        @func
        def head_backward(dl: Tokens, z: A, inverse: Inverse, w: HeadWeight):
            full = tf.reshard(w, layout=full_e, storage="gmem")
            partial = tf.matmul(dl, z, a_layout="KM")
            reduced = tf.reshard(partial, layout=reduced_e, storage="gmem")
            gradient = tf.reshard(reduced, layout=shard_e, storage="gmem") * inverse_p
            dz = tf.matmul(dl, full)
            dx = inverse * (dz - z * tf.reduce(dz * z, axes=(-1,)))
            return dx, gradient

        @func
        def embedding_backward(tokens: Tokens, dx: A):
            partial = tf.matmul(tokens, dx, a_layout="KM")
            reduced = tf.reshard(partial, layout=reduced_e, storage="gmem")
            return tf.reshard(reduced, layout=shard_e, storage="gmem") * inverse_p

    forward = getattr(Stages, "retain_forward" if retain else "release_forward")
    backward_stage = getattr(Stages, "retain_backward" if retain else "release_backward")
    embedding = getattr(Stages, "embedding")
    head_forward = getattr(Stages, "head_forward" if retain else "release_head")
    head_backward = Stages.head_backward
    embedding_backward = Stages.embedding_backward

    @module(entry="step", target=target, topologies=(gpu,))
    class Training:
        @func
        def step(tokens: Tokens, targets: Tokens, rotation: Rotation, mask: Mask,
                 we: Emb, q1: Tensor[(C, C), "f32", sq], k1: Tensor[(G * D, C), "f32", sk],
                 v1: Tensor[(G * D, C), "f32", sv], o1: Tensor[(C, C), "f32", so],
                 u1: Tensor[(F, C), "f32", su], d1: Tensor[(C, F), "f32", sd],
                 q2: Tensor[(C, C), "f32", sq], k2: Tensor[(G * D, C), "f32", sk],
                 v2: Tensor[(G * D, C), "f32", sv], o2: Tensor[(C, C), "f32", so],
                 u2: Tensor[(F, C), "f32", su], d2: Tensor[(C, F), "f32", sd], wh: Emb,
                 me: Emb, mq1: Tensor[(C, C), "f32", sq], mk1: Tensor[(G * D, C), "f32", sk],
                 mv1: Tensor[(G * D, C), "f32", sv], mo1: Tensor[(C, C), "f32", so],
                 mu1: Tensor[(F, C), "f32", su], md1: Tensor[(C, F), "f32", sd],
                 mq2: Tensor[(C, C), "f32", sq], mk2: Tensor[(G * D, C), "f32", sk],
                 mv2: Tensor[(G * D, C), "f32", sv], mo2: Tensor[(C, C), "f32", so],
                 mu2: Tensor[(F, C), "f32", su], md2: Tensor[(C, F), "f32", sd], mh: Emb):
            with Mesh(("gpu",), layout=mesh_layout, names=mesh_names) as dp:
                x = embedding(tokens, we)
                (y1, a1, ix1, qn1, iq1, kn1, ik1, qr1, kr1, vr1, p1, at1, r1, ir1, relu1,
                 hq1, hk1, hv1, ho1, hu1, hd1) = forward(x, (q1, k1, v1, o1, u1, d1), rotation, mask)
                (y2, a2, ix2, qn2, iq2, kn2, ik2, qr2, kr2, vr2, p2, at2, r2, ir2, relu2,
                 hq2, hk2, hv2, ho2, hu2, hd2) = forward(y1, (q2, k2, v2, o2, u2, d2), rotation, mask)
                logits, z, inverse, held_head = head_forward(y2, wh)
                maximum = tf.reduce(logits, axes=(-1,), kind=ReduceKind.MAX)
                shifted = logits - maximum
                log_z = tf.log(tf.reduce(tf.exp(shifted), axes=(-1,), kind=ReduceKind.SUM))
                nll = -tf.reduce(targets * (shifted - log_z), axes=(-1,), kind=ReduceKind.SUM)
                loss = tf.reduce(nll, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False) * global_mean_scale
                dl = (tf.softmax(logits, axis=-1) - targets) * local_mean_scale
                dy, gh = head_backward(dl, z, inverse, held_head)
                dx2, gq2, gk2, gv2, go2, gu2, gd2 = backward_stage(
                    dy, (y2, a2, ix2, qn2, iq2, kn2, ik2, qr2, kr2, vr2, p2, at2, r2, ir2, relu2),
                    (hq2, hk2, hv2, ho2, hu2, hd2), rotation)
                dx1, gq1, gk1, gv1, go1, gu1, gd1 = backward_stage(
                    dx2, (y1, a1, ix1, qn1, iq1, kn1, ik1, qr1, kr1, vr1, p1, at1, r1, ir1, relu1),
                    (hq1, hk1, hv1, ho1, hu1, hd1), rotation)
                g1 = (gq1, gk1, gv1, go1, gu1, gd1)
                g2 = (gq2, gk2, gv2, go2, gu2, gd2)
                ge = embedding_backward(tokens, dx1)
                ne = 0.9 * me + ge
                nq1 = 0.9 * mq1 + g1[0]
                nk1 = 0.9 * mk1 + g1[1]
                nv1 = 0.9 * mv1 + g1[2]
                no1 = 0.9 * mo1 + g1[3]
                nu1 = 0.9 * mu1 + g1[4]
                nd1 = 0.9 * md1 + g1[5]
                nq2 = 0.9 * mq2 + g2[0]
                nk2 = 0.9 * mk2 + g2[1]
                nv2 = 0.9 * mv2 + g2[2]
                no2 = 0.9 * mo2 + g2[3]
                nu2 = 0.9 * mu2 + g2[4]
                nd2 = 0.9 * md2 + g2[5]
                nh = 0.9 * mh + gh
                gradients = (ge, g1[0], g1[1], g1[2], g1[3], g1[4], g1[5],
                             g2[0], g2[1], g2[2], g2[3], g2[4], g2[5], gh)
                weights = (we - 0.01 * ne, q1 - 0.01 * nq1, k1 - 0.01 * nk1,
                           v1 - 0.01 * nv1, o1 - 0.01 * no1, u1 - 0.01 * nu1, d1 - 0.01 * nd1,
                           q2 - 0.01 * nq2, k2 - 0.01 * nk2, v2 - 0.01 * nv2,
                           o2 - 0.01 * no2, u2 - 0.01 * nu2, d2 - 0.01 * nd2, wh - 0.01 * nh)
                momentum = (ne, nq1, nk1, nv1, no1, nu1, nd1, nq2, nk2, nv2, no2, nu2, nd2, nh)
                return logits, loss, gradients, weights, momentum

    return Training
