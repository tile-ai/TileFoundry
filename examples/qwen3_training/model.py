"""Qwen3-MoE stages as HIR, cut into the fused kernels one configuration runs.

A kernel is a child Module whose call is one kernel launch: its inputs and results are the
values that reach gmem; what it computes inside is the HIR equivalent of that kernel,
not its schedule. The cuts sit where a configuration communicates, so they change
with it: with ``ep > 1`` two all-to-alls cut the MoE into Router, Experts and
Combine; with ``ep == 1`` there is no exchange and the MoE is one kernel. FSDP gathers
and the CP gather of K and V sit between kernels, in the layer.

    layer(h):  AttnIn → [gather K, V over cp] → FlashAttention → AttnOut
               → Router → [all-to-all] → Experts → [all-to-all] → Combine    (ep > 1)
               → MoE                                                         (ep == 1)
"""

from __future__ import annotations

import dataclasses

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

from .configs import AXES, GPUS, Config, Size

EPS = 1e-6
NEG = -1e30
TARGET = cuda.Target("nvidia.h200_sxm", device_count=GPUS)
TOPOLOGIES = (Topology("gpu", GPUS), Topology("cta", 1))
INNER = AXES[1:]


def mesh_of(config: Config):
    return Mesh((TOPOLOGIES[0],), Layout(tuple(config.size_of(axis) for axis in AXES)), AXES)


def placed(config: Config, s: int, shape, split=None, dtype=DType.f32):
    """*shape* on stage *s*; *split* maps a dim to the axes that split it, major first.

    An axis of size 1 splits nothing, so it is left out: the layout is the one a
    configuration without that axis would write.
    """
    split = {
        dim: tuple(a for a in axes if config.size_of(a) > 1) for dim, axes in (split or {}).items()
    }
    attrs = []
    for axis in AXES:
        dims = [dim for dim, axes in split.items() if axis in axes]
        attrs.append(Split(dims[0]) if dims else Broadcast())
    layout = ShardLayout.from_logical(
        shape, mesh_of(config)[s], tuple(attrs), storage="gmem", owners=TARGET.storage_owners
    )
    return TensorType(shape, dtype, layout, "gmem")


@dataclasses.dataclass(frozen=True)
class Shapes:
    """Every type of stage *s*: activations, gathered weights, sharded weights."""

    config: Config
    size: Size
    s: int

    @property
    def dims(self):
        c, z = self.config, self.size
        local = z.tokens // c.cp
        return dict(
            B=c.data * c.ep,
            T=z.tokens,
            C=z.hidden,
            H=z.q_heads,
            KV=z.kv_heads,
            D=z.head_dim,
            G=z.q_heads // z.kv_heads,
            E=z.experts,
            K=z.top_k,
            F=z.expert_hidden,
            V=z.vocab,
            L=z.layers // c.pp,
            DA=c.data,
            EP=c.ep,
            CP=c.cp,
            NL=local,
            CAP=local * z.top_k // z.experts,
            EL=z.experts // c.ep,
        )

    def types(self):
        c, s = self.config, self.s
        d = self.dims
        B, T, C, H, KV, D = d["B"], d["T"], d["C"], d["H"], d["KV"], d["D"]
        E, F, V, L = d["E"], d["F"], d["V"], d["L"]
        DA, EP, CP, CAP, EL = d["DA"], d["EP"], d["CP"], d["CAP"], d["EL"]
        tokens = {0: ("data", "ep"), 1: ("cp",)}
        dense = c.dense

        def on(shape, split=None, dtype=DType.f32):
            return placed(c, s, shape, split, dtype)

        def weight(shape, split, kept=None):
            """The gathered weight (still split by *kept*), the stored shard, the stacked shards."""
            stacked = {dim + 1: axes for dim, axes in split.items()}
            return on(shape, kept), on(shape, split), on((L,) + shape, stacked)

        t = dict(
            Hid=on((B, T, C), tokens),
            Q=on((B, T, H, D), tokens),
            KVt=on((B, T, KV, D), tokens),
            KVall=on((B, T, KV, D), {0: ("data", "ep")}),
            O=on((B, T, H * D), tokens),
            Cos=on((T, D)),
            Pos=on((T,), {0: ("cp",)}, DType.i32),
            Rows=on((T,), {0: ("cp",)}, DType.i64),
            Cols=on((T,), None, DType.i64),
            Slots=on((E * CAP,), None, DType.i64),
            Route=on((E * CAP,), None, DType.i64),
            Sent=on((DA, EP, CP, EP, EL * CAP, C), {0: ("data",), 1: ("ep",), 2: ("cp",)}),
            Recv=on((DA, EP, CP, EP, EL * CAP, C), {0: ("data",), 2: ("cp",), 3: ("ep",)}),
            Gate=on((DA, EP, CP, E * CAP), {0: ("data",), 1: ("ep",), 2: ("cp",)}),
            Ids=on((B * T,), None, DType.i32),
            Targets=on((B, T), tokens, DType.i64),
            Vocab=on((V,), None, DType.i64),
        )
        experts = {0: ("ep",)}
        for name, shape, split, kept in (
            ("Norm", (C,), {0: dense}, None),
            ("QNorm", (D,), {0: dense}, None),
            ("Wq", (H * D, C), {0: dense}, None),
            ("Wk", (KV * D, C), {0: dense}, None),
            ("Wo", (C, H * D), {0: dense}, None),
            ("Wr", (E, C), {0: dense}, None),
            ("Wg", (E, F, C), {**experts, 1: c.expert}, experts),
            ("Wd", (E, C, F), {**experts, 2: c.expert}, experts),
            ("Table", (V, C), {0: dense}, None),
        ):
            t[name + "F"], t[name], t[name + "S"] = weight(shape, split, kept)
        return t


def build_kernels(config: Config, size: Size, s: int) -> dict:
    """The fused kernels of stage *s*, named ``<Kernel><s>``."""
    d = Shapes(config, size, s).dims
    t = Shapes(config, size, s).types()
    B, T, C, H, KV, D, G = d["B"], d["T"], d["C"], d["H"], d["KV"], d["D"], d["G"]
    E, K, F, V = d["E"], d["K"], d["F"], d["V"]
    DA, EP, CP, NL, CAP, EL = d["DA"], d["EP"], d["CP"], d["NL"], d["CAP"], d["EL"]
    Hid, Q, KVt, KVall, O = t["Hid"], t["Q"], t["KVt"], t["KVall"], t["O"]
    Cos, Pos, Rows, Cols, Slots, Route = (
        t["Cos"],
        t["Pos"],
        t["Rows"],
        t["Cols"],
        t["Slots"],
        t["Route"],
    )
    Sent, Recv, Gate, Ids, Targets, Vocab = (
        t["Sent"],
        t["Recv"],
        t["Gate"],
        t["Ids"],
        t["Targets"],
        t["Vocab"],
    )
    NormF, QNormF, WqF, WkF, WoF, WrF = (
        t["NormF"],
        t["QNormF"],
        t["WqF"],
        t["WkF"],
        t["WoF"],
        t["WrF"],
    )
    WgF, WdF, TableF = t["WgF"], t["WdF"], t["TableF"]
    scale = D**-0.5
    pp, data, ep, cp = config.pp, config.data, config.ep, config.cp

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class AttnIn:
        """RMSNorm, the Q/K/V projections, the per-head Q and K norms, and RoPE."""

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(
            h: Hid,
            wn: NormF,
            wq: WqF,
            wk: WkF,
            wv: WkF,
            gq: QNormF,
            gk: QNormF,
            cos: Cos,
            sin: Cos,
            pos: Pos,
        ):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    x = tf.rms_norm(h, wn, eps=EPS)
                    q = tf.rms_norm(
                        tf.reshape(tf.matmul(x, wq, b_layout="NK"), new_shape=(B, T, H, D)),
                        gq,
                        eps=EPS,
                    )
                    k = tf.rms_norm(
                        tf.reshape(tf.matmul(x, wk, b_layout="NK"), new_shape=(B, T, KV, D)),
                        gk,
                        eps=EPS,
                    )
                    v = tf.reshape(tf.matmul(x, wv, b_layout="NK"), new_shape=(B, T, KV, D))
                    rq, rk = tf.rope(q, k, cos, sin, pos)
                    return (
                        tf.reshard(rq, layout=Q.layout, storage="gmem"),
                        tf.reshard(rk, layout=KVt.layout, storage="gmem"),
                        tf.reshard(v, layout=KVt.layout, storage="gmem"),
                    )

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class FlashAttention:
        """Causal GQA attention over the whole sequence's K and V; one FA kernel.

        The scores never reach gmem in the kernel; the stage always recomputes this
        kernel, so its backward keeps only q, k and v.
        """

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(q: Q, k: KVall, v: KVall):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    qh = tf.transpose(
                        tf.reshape(q, new_shape=(B, T, KV, G, D)), perm=(0, 2, 3, 1, 4)
                    )
                    kh = tf.transpose(
                        tf.reshape(k, new_shape=(B, T, KV, 1, D)), perm=(0, 2, 3, 1, 4)
                    )
                    vh = tf.transpose(
                        tf.reshape(v, new_shape=(B, T, KV, 1, D)), perm=(0, 2, 3, 1, 4)
                    )
                    rows = tf.reshape(tf.arange(type=Rows), new_shape=(T, 1))
                    cols = tf.reshape(tf.arange(type=Cols), new_shape=(1, T))
                    mask = tf.reshape(
                        tf.cast(tf.cmp_gt(cols, rows), "f32") * NEG, new_shape=(1, 1, 1, T, T)
                    )
                    p = tf.softmax(tf.matmul(qh, kh, b_layout="NK") * scale + mask, axis=-1)
                    o = tf.transpose(tf.matmul(p, vh), perm=(0, 3, 1, 2, 4))
                    return tf.reshard(
                        tf.reshape(o, new_shape=(B, T, H * D)), layout=O.layout, storage="gmem"
                    )

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class AttnOut:
        """The output projection, the residual add, and the MoE's RMSNorm."""

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(o: O, h: Hid, wo: WoF, wn: NormF):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    h1 = h + tf.matmul(o, wo, b_layout="NK")
                    return h1, tf.reshard(
                        tf.rms_norm(h1, wn, eps=EPS), layout=Hid.layout, storage="gmem"
                    )

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class Router:
        """Router logits, softmax, top-k, the normalized gate of each slot, and the permute.

        Which local token fills each slot is the step's input ``route`` (fixed
        capacity, the same on every unit); the gate of a slot is its token's
        probability for the slot's expert over the sum of that token's top-k.
        """

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(x: Hid, wr: WrF, route: Route):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    tokens = tf.reshape(x, new_shape=(DA, EP, CP, NL, C))
                    probs = tf.softmax(tf.matmul(tokens, wr, b_layout="NK"), axis=-1)
                    top, _ = tf.topk(probs, k=K, axis=-1)
                    total = tf.reduce(top, axes=(4,), kind=ReduceKind.SUM, keepdim=False)
                    slot = route * E + tf.floor_div(tf.arange(type=Slots), CAP)
                    chosen = tf.index_select(
                        tf.reshape(probs, new_shape=(DA, EP, CP, NL * E)), slot, dim=3
                    )
                    gate = chosen / tf.index_select(total, route, dim=3)
                    sent = tf.reshape(
                        tf.index_select(tokens, route, dim=3),
                        new_shape=(DA, EP, CP, EP, EL * CAP, C),
                    )
                    return tf.reshard(sent, layout=Sent.layout, storage="gmem"), tf.reshard(
                        gate, layout=Gate.layout, storage="gmem"
                    )

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class Experts:
        """Each local expert's SwiGLU MLP over the slots every ep source sent it."""

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(x: Recv, wg: WgF, wu: WgF, wd: WdF):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    grouped = tf.transpose(
                        tf.reshape(x, new_shape=(DA, EP, CP, EP, EL, CAP, C)),
                        perm=(3, 4, 0, 2, 1, 5, 6),
                    )
                    rows = tf.reshape(grouped, new_shape=(EP, EL, DA, CP, EP * CAP, C))
                    wg_view = tf.reshape(wg, new_shape=(EP, EL, 1, 1, F, C))
                    wu_view = tf.reshape(wu, new_shape=(EP, EL, 1, 1, F, C))
                    wd_view = tf.reshape(wd, new_shape=(EP, EL, 1, 1, C, F))
                    g = tf.matmul(rows, wg_view, b_layout="NK")
                    u = tf.matmul(rows, wu_view, b_layout="NK")
                    y = tf.matmul(tf.silu(g) * u, wd_view, b_layout="NK")
                    back = tf.transpose(
                        tf.reshape(y, new_shape=(EP, EL, DA, CP, EP, CAP, C)),
                        perm=(2, 4, 3, 0, 1, 5, 6),
                    )
                    return tf.reshard(
                        tf.reshape(back, new_shape=(DA, EP, CP, EP, EL * CAP, C)),
                        layout=Recv.layout,
                        storage="gmem",
                    )

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class Combine:
        """Each slot's output times its gate, added back to its token with the residual."""

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(y: Sent, gate: Gate, route: Route, h: Hid):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    weighted = tf.reshape(y, new_shape=(DA, EP, CP, E * CAP, C)) * tf.reshape(
                        gate, new_shape=(DA, EP, CP, E * CAP, 1)
                    )
                    out = tf.index_add(
                        tf.reshape(h, new_shape=(DA, EP, CP, NL, C)), route, weighted, dim=3
                    )
                    return tf.reshard(
                        tf.reshape(out, new_shape=(B, T, C)), layout=Hid.layout, storage="gmem"
                    )

    AttnIn, FlashAttention, AttnOut, Router, Experts, Combine = (
        dataclasses.replace(kernel, name=f"{kernel.name}{s}")
        for kernel in (AttnIn, FlashAttention, AttnOut, Router, Experts, Combine)
    )

    moe = None
    if ep == 1:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class MoE:
            """With ``ep == 1`` nothing is exchanged: router, experts and combine are one kernel."""

            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
            router, experts, combine = Router, Experts, Combine

            @func
            def forward(x: Hid, h: Hid, wr: WrF, wg: WgF, wu: WgF, wd: WdF, route: Route):
                with mesh:
                    with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                        sent, gate = router(x, wr, route)
                        return combine(experts(sent, wg, wu, wd), gate, route, h)

        moe = dataclasses.replace(MoE, name=f"MoE{s}")

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class Embed:
        """Every unit looks up the whole stage's token ids, then keeps its own slice (D57)."""

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(ids: Ids, table: TableF):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    rows = tf.reshape(tf.index_select(table, ids, dim=0), new_shape=(B, T, C))
                    return tf.reshard(rows, layout=Hid.layout, storage="gmem")

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class Head:
        """Final RMSNorm, LM head and cross-entropy; one fused kernel that never stores logits.

        The stage always recomputes this kernel. The loss is each unit's sum over its own
        tokens; the step does not average it.
        """

        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

        @func
        def forward(h: Hid, wn: NormF, head: TableF, targets: Targets):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    probs = tf.softmax(
                        tf.matmul(tf.rms_norm(h, wn, eps=EPS), head, b_layout="NK"), axis=-1
                    )
                    vocab = tf.reshape(tf.arange(type=Vocab), new_shape=(1, 1, V))
                    hit = tf.cast(tf.cmp_eq(tf.reshape(targets, new_shape=(B, T, 1)), vocab), "f32")
                    picked = tf.reduce(probs * hit, axes=(2,), kind=ReduceKind.SUM, keepdim=False)
                    return tf.reduce(
                        tf.neg(tf.log(picked)), axes=(0, 1), kind=ReduceKind.SUM, keepdim=False
                    )

    named = {
        kernel.name.rstrip("0123456789"): kernel
        for kernel in (AttnIn, FlashAttention, AttnOut, Router, Experts, Combine)
    }
    for kernel in (Embed, Head):
        named[kernel.name] = dataclasses.replace(kernel, name=f"{kernel.name}{s}")
    if moe is not None:
        named["MoE"] = moe
    return named


def build_stage(config: Config, size: Size, s: int):
    """The forward Module of stage *s*: ``Stage<s>``, with ``Layer<s>`` and its kernels."""
    size.check(config)
    shapes = Shapes(config, size, s)
    d, t = shapes.dims, shapes.types()
    kernels = build_kernels(config, size, s)
    L = d["L"]
    Hid, KVall, Recv, Sent, Cos, Pos, Route = (
        t[k] for k in ("Hid", "KVall", "Recv", "Sent", "Cos", "Pos", "Route")
    )
    Ids, Targets = t["Ids"], t["Targets"]
    NormF, QNormF, WqF, WkF, WoF, WrF, WgF, WdF, TableF = (
        t[k] for k in ("NormF", "QNormF", "WqF", "WkF", "WoF", "WrF", "WgF", "WdF", "TableF")
    )
    Norm, QNorm, Wq, Wk, Wo, Wr, Wg, Wd, Table = (
        t[k] for k in ("Norm", "QNorm", "Wq", "Wk", "Wo", "Wr", "Wg", "Wd", "Table")
    )
    NormS, QNormS, WqS, WkS, WoS, WrS, WgS, WdS = (
        t[k] for k in ("NormS", "QNormS", "WqS", "WkS", "WoS", "WrS", "WgS", "WdS")
    )
    C, H, KV, D, E, F = d["C"], d["H"], d["KV"], d["D"], d["E"], d["F"]
    pp, data, ep, cp = config.pp, config.data, config.ep, config.cp
    first = s == 0

    if ep > 1:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class Layer:
            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
            attn_in, attention, attn_out = (
                kernels["AttnIn"],
                kernels["FlashAttention"],
                kernels["AttnOut"],
            )
            router, experts, combine = kernels["Router"], kernels["Experts"], kernels["Combine"]

            @func
            def forward(
                h: Hid,
                wn1: Norm,
                wq: Wq,
                wk: Wk,
                wv: Wk,
                gq: QNorm,
                gk: QNorm,
                wo: Wo,
                wn2: Norm,
                wr: Wr,
                wg: Wg,
                wu: Wg,
                wd: Wd,
                cos: Cos,
                sin: Cos,
                pos: Pos,
                route: Route,
            ):
                with mesh:
                    with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                        wn1_full = tf.reshard(wn1, layout=NormF.layout, storage="gmem")
                        wq_full = tf.reshard(wq, layout=WqF.layout, storage="gmem")
                        wk_full = tf.reshard(wk, layout=WkF.layout, storage="gmem")
                        wv_full = tf.reshard(wv, layout=WkF.layout, storage="gmem")
                        gq_full = tf.reshard(gq, layout=QNormF.layout, storage="gmem")
                        gk_full = tf.reshard(gk, layout=QNormF.layout, storage="gmem")
                        q, k, v = attn_in(
                            h, wn1_full, wq_full, wk_full, wv_full, gq_full, gk_full, cos, sin, pos
                        )
                        k_all = tf.reshard(k, layout=KVall.layout, storage="gmem")
                        v_all = tf.reshard(v, layout=KVall.layout, storage="gmem")
                        o = attention(q, k_all, v_all)
                        wo_full = tf.reshard(wo, layout=WoF.layout, storage="gmem")
                        wn2_full = tf.reshard(wn2, layout=NormF.layout, storage="gmem")
                        h1, x = attn_out(o, h, wo_full, wn2_full)
                        wr_full = tf.reshard(wr, layout=WrF.layout, storage="gmem")
                        sent, gate = router(x, wr_full, route)
                        received = tf.reshard(sent, layout=Recv.layout, storage="gmem")
                        wg_full = tf.reshard(wg, layout=WgF.layout, storage="gmem")
                        wu_full = tf.reshard(wu, layout=WgF.layout, storage="gmem")
                        wd_full = tf.reshard(wd, layout=WdF.layout, storage="gmem")
                        y = experts(received, wg_full, wu_full, wd_full)
                        returned = tf.reshard(y, layout=Sent.layout, storage="gmem")
                        return combine(returned, gate, route, h1)
    else:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class Layer:
            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
            attn_in, attention, attn_out, moe = (
                kernels["AttnIn"],
                kernels["FlashAttention"],
                kernels["AttnOut"],
                kernels["MoE"],
            )

            @func
            def forward(
                h: Hid,
                wn1: Norm,
                wq: Wq,
                wk: Wk,
                wv: Wk,
                gq: QNorm,
                gk: QNorm,
                wo: Wo,
                wn2: Norm,
                wr: Wr,
                wg: Wg,
                wu: Wg,
                wd: Wd,
                cos: Cos,
                sin: Cos,
                pos: Pos,
                route: Route,
            ):
                with mesh:
                    with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                        wn1_full = tf.reshard(wn1, layout=NormF.layout, storage="gmem")
                        wq_full = tf.reshard(wq, layout=WqF.layout, storage="gmem")
                        wk_full = tf.reshard(wk, layout=WkF.layout, storage="gmem")
                        wv_full = tf.reshard(wv, layout=WkF.layout, storage="gmem")
                        gq_full = tf.reshard(gq, layout=QNormF.layout, storage="gmem")
                        gk_full = tf.reshard(gk, layout=QNormF.layout, storage="gmem")
                        q, k, v = attn_in(
                            h, wn1_full, wq_full, wk_full, wv_full, gq_full, gk_full, cos, sin, pos
                        )
                        k_all = tf.reshard(k, layout=KVall.layout, storage="gmem")
                        v_all = tf.reshard(v, layout=KVall.layout, storage="gmem")
                        o = attention(q, k_all, v_all)
                        wo_full = tf.reshard(wo, layout=WoF.layout, storage="gmem")
                        wn2_full = tf.reshard(wn2, layout=NormF.layout, storage="gmem")
                        h1, x = attn_out(o, h, wo_full, wn2_full)
                        wr_full = tf.reshard(wr, layout=WrF.layout, storage="gmem")
                        wg_full = tf.reshard(wg, layout=WgF.layout, storage="gmem")
                        wu_full = tf.reshard(wu, layout=WgF.layout, storage="gmem")
                        wd_full = tf.reshard(wd, layout=WdF.layout, storage="gmem")
                        return moe(x, h1, wr_full, wg_full, wu_full, wd_full, route)

    Layer = dataclasses.replace(Layer, name=f"Layer{s}")

    if first:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class Stage:
            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
            layers = ModuleList(Layer, L)
            embed = kernels["Embed"]

            @func
            def forward(
                ids: Ids,
                table: Table,
                wn1: NormS,
                wq: WqS,
                wk: WkS,
                wv: WkS,
                gq: QNormS,
                gk: QNormS,
                wo: WoS,
                wn2: NormS,
                wr: WrS,
                wg: WgS,
                wu: WgS,
                wd: WdS,
                cos: Cos,
                sin: Cos,
                pos: Pos,
                route: Route,
            ):
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    h = embed(ids, tf.reshard(table, layout=TableF.layout, storage="gmem"))
                    for t in range(L):
                        i = tf.reshape(t, new_shape=(1,))
                        h = layers[t](
                            h,
                            tf.reshard(
                                tf.reshape(tf.index_select(wn1, i, dim=0), new_shape=(C,)),
                                layout=Norm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wq, i, dim=0), new_shape=(H * D, C)),
                                layout=Wq.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wk, i, dim=0), new_shape=(KV * D, C)),
                                layout=Wk.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wv, i, dim=0), new_shape=(KV * D, C)),
                                layout=Wk.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(gq, i, dim=0), new_shape=(D,)),
                                layout=QNorm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(gk, i, dim=0), new_shape=(D,)),
                                layout=QNorm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wo, i, dim=0), new_shape=(C, H * D)),
                                layout=Wo.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wn2, i, dim=0), new_shape=(C,)),
                                layout=Norm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wr, i, dim=0), new_shape=(E, C)),
                                layout=Wr.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wg, i, dim=0), new_shape=(E, F, C)),
                                layout=Wg.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wu, i, dim=0), new_shape=(E, F, C)),
                                layout=Wg.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wd, i, dim=0), new_shape=(E, C, F)),
                                layout=Wd.layout,
                                storage="gmem",
                            ),
                            cos,
                            sin,
                            pos,
                            route,
                        )
                    return h
    else:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class Stage:
            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
            layers = ModuleList(Layer, L)
            loss = kernels["Head"]

            @func
            def forward(
                h: Hid,
                wn1: NormS,
                wq: WqS,
                wk: WkS,
                wv: WkS,
                gq: QNormS,
                gk: QNormS,
                wo: WoS,
                wn2: NormS,
                wr: WrS,
                wg: WgS,
                wu: WgS,
                wd: WdS,
                cos: Cos,
                sin: Cos,
                pos: Pos,
                route: Route,
                norm: Norm,
                lm_head: Table,
                targets: Targets,
            ):
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    for t in range(L):
                        i = tf.reshape(t, new_shape=(1,))
                        h = layers[t](
                            h,
                            tf.reshard(
                                tf.reshape(tf.index_select(wn1, i, dim=0), new_shape=(C,)),
                                layout=Norm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wq, i, dim=0), new_shape=(H * D, C)),
                                layout=Wq.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wk, i, dim=0), new_shape=(KV * D, C)),
                                layout=Wk.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wv, i, dim=0), new_shape=(KV * D, C)),
                                layout=Wk.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(gq, i, dim=0), new_shape=(D,)),
                                layout=QNorm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(gk, i, dim=0), new_shape=(D,)),
                                layout=QNorm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wo, i, dim=0), new_shape=(C, H * D)),
                                layout=Wo.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wn2, i, dim=0), new_shape=(C,)),
                                layout=Norm.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wr, i, dim=0), new_shape=(E, C)),
                                layout=Wr.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wg, i, dim=0), new_shape=(E, F, C)),
                                layout=Wg.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wu, i, dim=0), new_shape=(E, F, C)),
                                layout=Wg.layout,
                                storage="gmem",
                            ),
                            tf.reshard(
                                tf.reshape(tf.index_select(wd, i, dim=0), new_shape=(E, C, F)),
                                layout=Wd.layout,
                                storage="gmem",
                            ),
                            cos,
                            sin,
                            pos,
                            route,
                        )
                    norm_full = tf.reshard(norm, layout=NormF.layout, storage="gmem")
                    head_full = tf.reshard(lm_head, layout=TableF.layout, storage="gmem")
                    return loss(h, norm_full, head_full, targets)

    return dataclasses.replace(Stage, name=f"Stage{s}")
