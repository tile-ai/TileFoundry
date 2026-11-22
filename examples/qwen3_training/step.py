"""One training step of two pipeline stages, as HIR: the Module the export reads.

Each stage is differentiated on its own (D24); the step moves ``h`` to stage 1 and
``dh`` back to stage 0, once per microbatch, and adds up the weight gradients. Then
each stage updates its weights:

    Muon (q, k, v, o, expert gate, up, down):
        gradient --many-to-one--> owner mesh[s, 0, 0, 0]
        momentum, 5 Newton-Schulz iterations on the owner
        update --one-to-many--> the weight's own shards;  w -= LR * update
    the rest: SGD with momentum, each unit on its own shard

The Muon here is ours, written for its communication and memory; it is not DMuon.
The FlashAttention and Head kernels are always recomputed: they keep no
scores and no logits.
"""

from __future__ import annotations

import dataclasses

from tilefoundry.dsl import *

from .configs import AXES, Config, Size
from .model import INNER, TARGET, TOPOLOGIES, Shapes, build_stage, mesh_of, placed

RECOMPUTE = ("none", "regather", "layer")
LAYER_WEIGHTS = ("wn1", "wq", "wk", "wv", "gq", "gk", "wo", "wn2", "wr", "wg", "wu", "wd")
EXPERT_WEIGHTS = ("wg", "wu", "wd")
"""Weights the Experts kernel reshapes: the backward reads the reshape, so regather names it too."""
LR = 0.02
MOMENTUM = 0.95
NS = (3.4445, -4.775, 2.0315)


def selectors(recompute: str, s: int) -> tuple[str, ...]:
    """What stage *s* recomputes in its backward under *recompute*."""
    if recompute not in RECOMPUTE:
        raise ValueError(f"recompute {recompute!r}: expected one of {', '.join(RECOMPUTE)}")
    always = (f"FlashAttention{s}",) + ((f"Head{s}",) if s == 1 else ())
    chosen = {
        "none": (),
        "regather": tuple(f"Layer{s}:{name}_full" for name in LAYER_WEIGHTS)
        + tuple(f"Experts{s}:{name}_view" for name in EXPERT_WEIGHTS),
        "layer": (f"Layer{s}",),
    }[recompute]
    return always + chosen


def build_muon(config: Config, s: int, name: str, stored: TensorType):
    """Muon for one stacked matrix of stage *s*; the owner unit orthogonalizes it.

    The momentum is sharded like its weight. The owner gets the whole momentum and
    runs Newton-Schulz; each unit gets the update back and keeps its own shard. The
    core moves a Split value to a part of its units in two steps, so the owner's copy
    comes from a gather on every unit of the stage, and the update goes back as a
    broadcast from the owner.
    """
    shape = stored.shape
    Owned = TensorType(
        shape,
        DType.f32,
        ShardLayout.from_logical(
            shape, mesh_of(config)[s, 0, 0, 0], (), storage="gmem", owners=TARGET.storage_owners
        ),
        "gmem",
    )
    Whole = placed(config, s, shape)
    a, b, c = NS
    last_two = (len(shape) - 2, len(shape) - 1)
    pp, data, ep, cp = config.pp, config.data, config.ep, config.cp

    if shape[-2] > shape[-1]:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class NewtonSchulz:
            """A tall matrix: the Gram matrix is XᵀX, and it multiplies X from the right."""

            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

            @func
            def forward(m: Owned):
                with mesh:
                    with Mesh(mesh[s, 0, 0, 0], layout=(1,), names=("owner",)) as _owner:
                        x = m * tf.rsqrt(
                            tf.reduce(m * m, axes=last_two, kind=ReduceKind.SUM, keepdim=True)
                            + 1e-7
                        )
                        for _ in range(5):
                            gram = tf.matmul(x, x, a_layout="KM")
                            x = x * a + tf.matmul(x, gram * b + tf.matmul(gram, gram) * c)
                        return x
    else:

        @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
        class NewtonSchulz:
            """A wide matrix: the Gram matrix is XXᵀ, and it multiplies X from the left."""

            mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)

            @func
            def forward(m: Owned):
                with mesh:
                    with Mesh(mesh[s, 0, 0, 0], layout=(1,), names=("owner",)) as _owner:
                        x = m * tf.rsqrt(
                            tf.reduce(m * m, axes=last_two, kind=ReduceKind.SUM, keepdim=True)
                            + 1e-7
                        )
                        for _ in range(5):
                            gram = tf.matmul(x, x, b_layout="NK")
                            x = x * a + tf.matmul(gram * b + tf.matmul(gram, gram) * c, x)
                        return x

    @module(entry="forward", target=TARGET, topologies=TOPOLOGIES)
    class Muon:
        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
        orthogonalize = dataclasses.replace(NewtonSchulz, name=f"NewtonSchulz{name}{s}")

        @func
        def forward(g: stored, m: stored, w: stored):
            with mesh:
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    m_next = m * MOMENTUM + g
                    whole = tf.reshard(m_next, layout=Whole.layout, storage="gmem")
                owned = tf.reshard(whole, layout=Owned.layout, storage="gmem")
                with Mesh(mesh[s, 0, 0, 0], layout=(1,), names=("owner",)) as _owner:
                    update = orthogonalize(owned)
                published = tf.reshard(update, layout=Whole.layout, storage="gmem")
                with Mesh(mesh[s], layout=(data, ep, cp), names=INNER) as _stage:
                    return w - tf.reshard(
                        published, layout=stored.layout, storage="gmem"
                    ) * LR, m_next

    return dataclasses.replace(Muon, name=f"Muon{name}{s}")


def placed_stack(config: Config, s: int, one: TensorType, count: int):
    """The layout of *count* values of type *one*: a new leading dim that no axis splits."""
    attrs = tuple(
        Split(attr.axis + 1) if isinstance(attr, Split) else attr for attr in one.layout.attrs
    )
    return ShardLayout.from_logical(
        (count,) + one.shape,
        mesh_of(config)[s],
        attrs,
        storage="gmem",
        owners=TARGET.storage_owners,
    )


def build_step(config: Config, size: Size, recompute: str = "none"):
    """The training step of *config*: ``size.microbatches`` microbatches, then one update."""
    for s in (0, 1):
        selectors(recompute, s)
    t0, t1 = Shapes(config, size, 0).types(), Shapes(config, size, 1).types()
    stage0, stage1 = build_stage(config, size, 0), build_stage(config, size, 1)
    F0, B0 = differentiate(
        stage0, wrt=("table",) + LAYER_WEIGHTS, recompute=selectors(recompute, 0)
    )
    F1, B1 = differentiate(
        stage1, wrt=("h", "norm", "lm_head") + LAYER_WEIGHTS, recompute=selectors(recompute, 1)
    )
    M = size.microbatches
    pp, data, ep, cp = config.pp, config.data, config.ep, config.cp
    Loss = F1.entry_function().return_type.fields[0]
    Seed = B1.entry_function().params[0].type
    Ids, Hid0, Cos0, Pos0, Route0 = t0["Ids"], t0["Hid"], t0["Cos"], t0["Pos"], t0["Route"]
    Targets, Hid1, Cos1, Pos1, Route1 = t1["Targets"], t1["Hid"], t1["Cos"], t1["Pos"], t1["Route"]
    IdsM = TensorType((M,) + Ids.shape, Ids.dtype, placed_stack(config, 0, Ids, M), "gmem")
    TargetsM = TensorType(
        (M,) + Targets.shape, Targets.dtype, placed_stack(config, 1, Targets, M), "gmem"
    )
    Table0, NormS0, QNormS0, WqS0, WkS0, WoS0, WrS0, WgS0, WdS0 = (
        t0[k] for k in ("Table", "NormS", "QNormS", "WqS", "WkS", "WoS", "WrS", "WgS", "WdS")
    )
    Norm1, Table1, NormS1, QNormS1, WqS1, WkS1, WoS1, WrS1, WgS1, WdS1 = (
        t1[k]
        for k in ("Norm", "Table", "NormS", "QNormS", "WqS", "WkS", "WoS", "WrS", "WgS", "WdS")
    )
    muon = {}
    for s, t in ((0, t0), (1, t1)):
        for name in ("Wq", "Wk", "Wo", "Wg", "Wd"):
            muon[name, s] = build_muon(config, s, name, t[name + "S"])
    MuonQ0, MuonK0, MuonO0, MuonG0, MuonD0 = (
        muon[name, 0] for name in ("Wq", "Wk", "Wo", "Wg", "Wd")
    )
    MuonQ1, MuonK1, MuonO1, MuonG1, MuonD1 = (
        muon[name, 1] for name in ("Wq", "Wk", "Wo", "Wg", "Wd")
    )

    @module(entry="step", target=TARGET, topologies=TOPOLOGIES)
    class Step:
        mesh = Mesh(("gpu",), (pp, data, ep, cp), names=AXES)
        forward0, backward0, forward1, backward1 = F0, B0, F1, B1
        muon_q0, muon_k0, muon_o0, muon_g0, muon_d0 = MuonQ0, MuonK0, MuonO0, MuonG0, MuonD0
        muon_q1, muon_k1, muon_o1, muon_g1, muon_d1 = MuonQ1, MuonK1, MuonO1, MuonG1, MuonD1

        @func
        def step(
            ids: IdsM,
            targets: TargetsM,
            cos0: Cos0,
            sin0: Cos0,
            pos0: Pos0,
            route0: Route0,
            cos1: Cos1,
            sin1: Cos1,
            pos1: Pos1,
            route1: Route1,
            table: Table0,
            wn1_0: NormS0,
            wq0: WqS0,
            wk0: WkS0,
            wv0: WkS0,
            gq0: QNormS0,
            gk0: QNormS0,
            wo0: WoS0,
            wn2_0: NormS0,
            wr0: WrS0,
            wg0: WgS0,
            wu0: WgS0,
            wd0: WdS0,
            wn1_1: NormS1,
            wq1: WqS1,
            wk1: WkS1,
            wv1: WkS1,
            gq1: QNormS1,
            gk1: QNormS1,
            wo1: WoS1,
            wn2_1: NormS1,
            wr1: WrS1,
            wg1: WgS1,
            wu1: WgS1,
            wd1: WdS1,
            norm: Norm1,
            lm_head: Table1,
            m_table: Table0,
            m_wn1_0: NormS0,
            m_wq0: WqS0,
            m_wk0: WkS0,
            m_wv0: WkS0,
            m_gq0: QNormS0,
            m_gk0: QNormS0,
            m_wo0: WoS0,
            m_wn2_0: NormS0,
            m_wr0: WrS0,
            m_wg0: WgS0,
            m_wu0: WgS0,
            m_wd0: WdS0,
            m_wn1_1: NormS1,
            m_wq1: WqS1,
            m_wk1: WkS1,
            m_wv1: WkS1,
            m_gq1: QNormS1,
            m_gk1: QNormS1,
            m_wo1: WoS1,
            m_wn2_1: NormS1,
            m_wr1: WrS1,
            m_wg1: WgS1,
            m_wu1: WgS1,
            m_wd1: WdS1,
            m_norm: Norm1,
            m_lm_head: Table1,
        ):
            with mesh:
                with Mesh(mesh[0], layout=(data, ep, cp), names=INNER) as _stage0:
                    d_table, d_wn1_0, d_wq0, d_wk0, d_wv0, d_gq0, d_gk0 = (
                        tf.zeros(type=Table0),
                        tf.zeros(type=NormS0),
                        tf.zeros(type=WqS0),
                        tf.zeros(type=WkS0),
                        tf.zeros(type=WkS0),
                        tf.zeros(type=QNormS0),
                        tf.zeros(type=QNormS0),
                    )
                    d_wo0, d_wn2_0, d_wr0, d_wg0, d_wu0, d_wd0 = (
                        tf.zeros(type=WoS0),
                        tf.zeros(type=NormS0),
                        tf.zeros(type=WrS0),
                        tf.zeros(type=WgS0),
                        tf.zeros(type=WgS0),
                        tf.zeros(type=WdS0),
                    )
                with Mesh(mesh[1], layout=(data, ep, cp), names=INNER) as _stage1:
                    loss = tf.zeros(type=Loss)
                    seed = tf.zeros(type=Seed) + 1.0
                    d_norm, d_lm_head, d_wn1_1, d_wq1, d_wk1, d_wv1, d_gq1, d_gk1 = (
                        tf.zeros(type=Norm1),
                        tf.zeros(type=Table1),
                        tf.zeros(type=NormS1),
                        tf.zeros(type=WqS1),
                        tf.zeros(type=WkS1),
                        tf.zeros(type=WkS1),
                        tf.zeros(type=QNormS1),
                        tf.zeros(type=QNormS1),
                    )
                    d_wo1, d_wn2_1, d_wr1, d_wg1, d_wu1, d_wd1 = (
                        tf.zeros(type=WoS1),
                        tf.zeros(type=NormS1),
                        tf.zeros(type=WrS1),
                        tf.zeros(type=WgS1),
                        tf.zeros(type=WgS1),
                        tf.zeros(type=WdS1),
                    )
                for i in range(M):
                    at = tf.reshape(i, new_shape=(1,))
                    with Mesh(mesh[0], layout=(data, ep, cp), names=INNER) as _stage0:
                        ids_i = tf.reshard(
                            tf.reshape(tf.index_select(ids, at, dim=0), new_shape=Ids.shape),
                            layout=Ids.layout,
                            storage="gmem",
                        )
                        h, saved0 = forward0(
                            ids_i,
                            table,
                            wn1_0,
                            wq0,
                            wk0,
                            wv0,
                            gq0,
                            gk0,
                            wo0,
                            wn2_0,
                            wr0,
                            wg0,
                            wu0,
                            wd0,
                            cos0,
                            sin0,
                            pos0,
                            route0,
                        )
                    sent = tf.reshard(h, layout=Hid1.layout, storage="gmem")
                    with Mesh(mesh[1], layout=(data, ep, cp), names=INNER) as _stage1:
                        targets_i = tf.reshard(
                            tf.reshape(
                                tf.index_select(targets, at, dim=0), new_shape=Targets.shape
                            ),
                            layout=Targets.layout,
                            storage="gmem",
                        )
                        loss_i, saved1 = forward1(
                            sent,
                            wn1_1,
                            wq1,
                            wk1,
                            wv1,
                            gq1,
                            gk1,
                            wo1,
                            wn2_1,
                            wr1,
                            wg1,
                            wu1,
                            wd1,
                            cos1,
                            sin1,
                            pos1,
                            route1,
                            norm,
                            lm_head,
                            targets_i,
                        )
                        loss = loss + loss_i
                        (
                            dh,
                            g_norm,
                            g_lm_head,
                            g_wn1_1,
                            g_wq1,
                            g_wk1,
                            g_wv1,
                            g_gq1,
                            g_gk1,
                            g_wo1,
                            g_wn2_1,
                            g_wr1,
                            g_wg1,
                            g_wu1,
                            g_wd1,
                        ) = backward1(seed, saved1)
                        d_norm, d_lm_head, d_wn1_1, d_wq1, d_wk1, d_wv1, d_gq1, d_gk1 = (
                            d_norm + g_norm,
                            d_lm_head + g_lm_head,
                            d_wn1_1 + g_wn1_1,
                            d_wq1 + g_wq1,
                            d_wk1 + g_wk1,
                            d_wv1 + g_wv1,
                            d_gq1 + g_gq1,
                            d_gk1 + g_gk1,
                        )
                        d_wo1, d_wn2_1, d_wr1, d_wg1, d_wu1, d_wd1 = (
                            d_wo1 + g_wo1,
                            d_wn2_1 + g_wn2_1,
                            d_wr1 + g_wr1,
                            d_wg1 + g_wg1,
                            d_wu1 + g_wu1,
                            d_wd1 + g_wd1,
                        )
                    returned = tf.reshard(dh, layout=Hid0.layout, storage="gmem")
                    with Mesh(mesh[0], layout=(data, ep, cp), names=INNER) as _stage0:
                        (
                            g_table,
                            g_wn1_0,
                            g_wq0,
                            g_wk0,
                            g_wv0,
                            g_gq0,
                            g_gk0,
                            g_wo0,
                            g_wn2_0,
                            g_wr0,
                            g_wg0,
                            g_wu0,
                            g_wd0,
                        ) = backward0(returned, saved0)
                        d_table, d_wn1_0, d_wq0, d_wk0, d_wv0, d_gq0, d_gk0 = (
                            d_table + g_table,
                            d_wn1_0 + g_wn1_0,
                            d_wq0 + g_wq0,
                            d_wk0 + g_wk0,
                            d_wv0 + g_wv0,
                            d_gq0 + g_gq0,
                            d_gk0 + g_gk0,
                        )
                        d_wo0, d_wn2_0, d_wr0, d_wg0, d_wu0, d_wd0 = (
                            d_wo0 + g_wo0,
                            d_wn2_0 + g_wn2_0,
                            d_wr0 + g_wr0,
                            d_wg0 + g_wg0,
                            d_wu0 + g_wu0,
                            d_wd0 + g_wd0,
                        )
                with Mesh(mesh[0], layout=(data, ep, cp), names=INNER) as _stage0:
                    new_wq0, m_wq0 = muon_q0(d_wq0, m_wq0, wq0)
                    new_wk0, m_wk0 = muon_k0(d_wk0, m_wk0, wk0)
                    new_wv0, m_wv0 = muon_k0(d_wv0, m_wv0, wv0)
                    new_wo0, m_wo0 = muon_o0(d_wo0, m_wo0, wo0)
                    new_wg0, m_wg0 = muon_g0(d_wg0, m_wg0, wg0)
                    new_wu0, m_wu0 = muon_g0(d_wu0, m_wu0, wu0)
                    new_wd0, m_wd0 = muon_d0(d_wd0, m_wd0, wd0)
                    m_table, m_wn1_0, m_gq0, m_gk0, m_wn2_0, m_wr0 = (
                        m_table * MOMENTUM + d_table,
                        m_wn1_0 * MOMENTUM + d_wn1_0,
                        m_gq0 * MOMENTUM + d_gq0,
                        m_gk0 * MOMENTUM + d_gk0,
                        m_wn2_0 * MOMENTUM + d_wn2_0,
                        m_wr0 * MOMENTUM + d_wr0,
                    )
                    table, wn1_0, gq0, gk0, wn2_0, wr0 = (
                        table - m_table * LR,
                        wn1_0 - m_wn1_0 * LR,
                        gq0 - m_gq0 * LR,
                        gk0 - m_gk0 * LR,
                        wn2_0 - m_wn2_0 * LR,
                        wr0 - m_wr0 * LR,
                    )
                with Mesh(mesh[1], layout=(data, ep, cp), names=INNER) as _stage1:
                    new_wq1, m_wq1 = muon_q1(d_wq1, m_wq1, wq1)
                    new_wk1, m_wk1 = muon_k1(d_wk1, m_wk1, wk1)
                    new_wv1, m_wv1 = muon_k1(d_wv1, m_wv1, wv1)
                    new_wo1, m_wo1 = muon_o1(d_wo1, m_wo1, wo1)
                    new_wg1, m_wg1 = muon_g1(d_wg1, m_wg1, wg1)
                    new_wu1, m_wu1 = muon_g1(d_wu1, m_wu1, wu1)
                    new_wd1, m_wd1 = muon_d1(d_wd1, m_wd1, wd1)
                    m_norm, m_lm_head, m_wn1_1, m_gq1, m_gk1, m_wn2_1, m_wr1 = (
                        m_norm * MOMENTUM + d_norm,
                        m_lm_head * MOMENTUM + d_lm_head,
                        m_wn1_1 * MOMENTUM + d_wn1_1,
                        m_gq1 * MOMENTUM + d_gq1,
                        m_gk1 * MOMENTUM + d_gk1,
                        m_wn2_1 * MOMENTUM + d_wn2_1,
                        m_wr1 * MOMENTUM + d_wr1,
                    )
                    norm, lm_head, wn1_1, gq1, gk1, wn2_1, wr1 = (
                        norm - m_norm * LR,
                        lm_head - m_lm_head * LR,
                        wn1_1 - m_wn1_1 * LR,
                        gq1 - m_gq1 * LR,
                        gk1 - m_gk1 * LR,
                        wn2_1 - m_wn2_1 * LR,
                        wr1 - m_wr1 * LR,
                    )
                    return (
                        loss,
                        (
                            table,
                            wn1_0,
                            new_wq0,
                            new_wk0,
                            new_wv0,
                            gq0,
                            gk0,
                            new_wo0,
                            wn2_0,
                            wr0,
                            new_wg0,
                            new_wu0,
                            new_wd0,
                            wn1_1,
                            new_wq1,
                            new_wk1,
                            new_wv1,
                            gq1,
                            gk1,
                            new_wo1,
                            wn2_1,
                            wr1,
                            new_wg1,
                            new_wu1,
                            new_wd1,
                            norm,
                            lm_head,
                        ),
                        (
                            m_table,
                            m_wn1_0,
                            m_wq0,
                            m_wk0,
                            m_wv0,
                            m_gq0,
                            m_gk0,
                            m_wo0,
                            m_wn2_0,
                            m_wr0,
                            m_wg0,
                            m_wu0,
                            m_wd0,
                            m_wn1_1,
                            m_wq1,
                            m_wk1,
                            m_wv1,
                            m_gq1,
                            m_gk1,
                            m_wo1,
                            m_wn2_1,
                            m_wr1,
                            m_wg1,
                            m_wu1,
                            m_wd1,
                            m_norm,
                            m_lm_head,
                        ),
                    )

    return dataclasses.replace(Step, name=f"Step_{config.name}_{recompute}")
