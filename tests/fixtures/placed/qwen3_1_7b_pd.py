"""Qwen3-1.7B prefill and decode layer fixture.

The context bound comes from the ``max_position_embeddings`` scalar in the
model config; importing the executable model during test collection would
build a second IR program. ``SEQ`` is bounded by this fixture's 8192-row
prefill chunk envelope, not by the model limit. ``CAP`` is the independently
allocated 4608-position cache depth, while attention scans the prior context
plus the current sequence through ``CTX + SEQ``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_CONFIG = Path(__file__).parents[2] / "models" / "qwen3_1_7b" / "config.json"
MAX_POSITION_EMBEDDINGS = json.loads(_CONFIG.read_text(encoding="utf-8"))["max_position_embeddings"]

SEQ = DimVar("seq", 1, 8193)
CTX = DimVar("ctx_len", 0, MAX_POSITION_EMBEDDINGS)

CAP = 4608

HID = 2048
HQ, HKV, D = 16, 8, 128
G = HQ // HKV
QN, KN = HQ * D, HKV * D
QKV_N = QN + 2 * KN
FFN = 6144
L = 28
V = 151936
CTAS = 128
THREADS = 256

ROWS = 128
BN, BK = 128, 64
BKV = 128

DN_QKV, DN_O, DN_FFN, DN_DOWN = 32, 16, 48, 16

_H200 = CudaTarget("nvidia.h200_sxm")


@module(
    entry="model",
    target=_H200,
    topologies=(Topology("cta", CTAS), Topology("thread", THREADS)),
)
class PrefillLayer:
    @func
    def layer_prefill(
        x: Tensor[(SEQ, HID), "bf16"],
        w_qkv: ConstTensor[(HID, QKV_N), "bf16"],
        w_o: ConstTensor[(QN, HID), "bf16"],
        w_gu: ConstTensor[(HID, 2, FFN), "bf16"],
        w_down: ConstTensor[(FFN, HID), "bf16"],
        g_in: ConstTensor[(HID,), "bf16"],
        g_q: ConstTensor[(D,), "bf16"],
        g_k: ConstTensor[(D,), "bf16"],
        g_post: ConstTensor[(HID,), "bf16"],
        kc: Tensor[(1, CAP, HKV, D), "bf16"],
        vc: Tensor[(1, CAP, HKV, D), "bf16"],
        cur: Tensor[(), "i32"],
        cos: Tensor[(ROWS, D), "f32"],
        sin: Tensor[(ROWS, D), "f32"],
        pos: Tensor[(ROWS,), "i32"],
        xn: Tensor[(SEQ, HID), "bf16"],
        qkv: Tensor[(SEQ, QKV_N), "f32"],
        q: Tensor[(SEQ, HKV, G, D), "bf16"],
        attn: Tensor[(SEQ, HKV, G, D), "bf16"],
        h1: Tensor[(SEQ, HID), "bf16"],
        gu: Tensor[(SEQ, 2, FFN), "f32"],
        act: Tensor[(SEQ, FFN), "bf16"],
        out: Tensor[(SEQ, HID), "bf16"],
    ) -> Tensor[(SEQ, HID), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("cta",)) as _cta:
            xb = xn
            for m in tile(SEQ, ROWS):  # noqa: F405
                xr = tf.rms_norm(x[m, 0:HID], g_in)
                xb = tf.insert_slice(xb, tf.reshard(xr, (ROWS, HID), "gmem"), (m, 0))

            p = qkv
            for m in tile(SEQ, ROWS):  # noqa: F405
                for n in tile(QKV_N, BN):  # noqa: F405
                    acc = tf.zeros(Tensor[(ROWS, BN), "f32", (ROWS, BN), "rmem"])
                    for k in tile(HID, BK):  # noqa: F405
                        xt = tf.reshard(xb[m, k], (ROWS, BK), "smem")
                        wt = tf.reshard(
                            tf.reshape(w_qkv[k, n], (BK, BN)),
                            (BK, BN),
                            "smem",
                        )
                        mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                        acc = acc + tf.reshard(mm, (ROWS, BN), "rmem")
                    p = tf.insert_slice(p, tf.reshard(acc, (ROWS, BN), "gmem"), (m, n))

            qb = q
            kc2 = kc
            vc2 = vc
            for m in tile(SEQ, ROWS):  # noqa: F405
                q4 = tf.reshape(tf.cast(p[m, 0:QN], dtype="bf16"), (1, ROWS, HQ, D))
                k4 = tf.reshape(tf.cast(p[m, QN : QN + KN], dtype="bf16"), (1, ROWS, HKV, D))
                v4 = tf.reshape(
                    tf.cast(p[m, QN + KN : QKV_N], dtype="bf16"),
                    (1, ROWS, HKV, D),
                )
                qr, kr = tf.rope(tf.rms_norm(q4, g_q), tf.rms_norm(k4, g_k), cos, sin, pos)
                width = tf.reshape(cur, ())
                kc2 = tf.cache_update(kc2, cur, width, kr)
                vc2 = tf.cache_update(vc2, cur, width, v4)
                qb = tf.insert_slice(qb, tf.reshape(qr, (ROWS, HKV, G, D)), (m, 0, 0, 0))

            ab = attn
            for m in tile(SEQ, ROWS):  # noqa: F405
                for kv in tile(HKV, 1):  # noqa: F405
                    for g in tile(G, 1):  # noqa: F405
                        qh = tf.reshard(tf.reshape(qb[m, kv, g, 0:D], (ROWS, D)), (ROWS, D), "smem")
                        mx = tf.zeros(Tensor[(ROWS, 1), "f32", (ROWS, 1), "rmem"]) - 30000.0
                        lr = tf.zeros(Tensor[(ROWS, 1), "f32", (ROWS, 1), "rmem"])
                        acc = tf.zeros(Tensor[(ROWS, D), "f32", (ROWS, D), "rmem"])
                        for c in tile(CTX + SEQ, BKV):  # noqa: F405
                            khb = tf.reshard(
                                tf.reshape(kc2[0:1, c, kv, 0:D], (BKV, D)), (BKV, D), "smem"
                            )
                            kh = tf.transpose(khb, perm=(1, 0))
                            vh = tf.reshard(
                                tf.reshape(vc2[0:1, c, kv, 0:D], (BKV, D)),
                                (BKV, D),
                                "smem",
                            )
                            sc = tf.reshard(
                                tf.cast(tf.matmul(qh, kh), dtype="f32"),
                                (ROWS, BKV),
                                "rmem",
                            )
                            mn = tf.maximum(mx, tf.reduce(sc, axes=(1,), keepdim=True, kind="max"))
                            scale = tf.exp(mx - mn)
                            pr = tf.exp(sc - mn)
                            lr = lr * scale + tf.reduce(pr, axes=(1,), keepdim=True, kind="sum")
                            pb = tf.reshard(tf.cast(pr, dtype="bf16"), (ROWS, BKV), "smem")
                            pv = tf.cast(tf.matmul(pb, vh), dtype="f32")
                            acc = acc * scale + tf.reshard(pv, (ROWS, D), "rmem")
                            mx = mn
                        ah = tf.reshard(tf.cast(tf.div(acc, lr), dtype="bf16"), (ROWS, D), "gmem")
                        ab = tf.insert_slice(ab, tf.reshape(ah, (ROWS, 1, 1, D)), (m, kv, g, 0))

            af = tf.reshape(ab, (SEQ, QN))
            hb = h1
            for m in tile(SEQ, ROWS):  # noqa: F405
                for n in tile(HID, BN):  # noqa: F405
                    op = tf.zeros(Tensor[(ROWS, BN), "f32", (ROWS, BN), "rmem"])
                    for k in tile(QN, BK):  # noqa: F405
                        xt = tf.reshard(af[m, k], (ROWS, BK), "smem")
                        wt = tf.reshard(
                            tf.reshape(w_o[k, n], (BK, BN)),
                            (BK, BN),
                            "smem",
                        )
                        mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                        op = op + tf.reshard(mm, (ROWS, BN), "rmem")
                    xr = tf.reshard(x[m, n], (ROWS, BN), "rmem")
                    sm = tf.cast(tf.cast(xr, dtype="f32") + op, dtype="bf16")
                    hb = tf.insert_slice(hb, tf.reshard(sm, (ROWS, BN), "gmem"), (m, n))

            gb = gu
            for m in tile(SEQ, ROWS):  # noqa: F405
                xn1 = tf.rms_norm(hb[m, 0:HID], g_post)
                for n in tile(FFN, BN):  # noqa: F405
                    gv = tf.zeros(Tensor[(ROWS, BN), "f32", (ROWS, BN), "rmem"])
                    for k in tile(HID, BK):  # noqa: F405
                        wk = tf.reshape(w_gu[k, 0:1, n], (BK, BN))
                        xt = tf.reshard(xn1[0:ROWS, k], (ROWS, BK), "smem")
                        wt = tf.reshard(wk, (BK, BN), "smem")
                        mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                        gv = gv + tf.reshard(mm, (ROWS, BN), "rmem")
                    gb = tf.insert_slice(
                        gb,
                        tf.reshape(tf.reshard(gv, (ROWS, BN), "gmem"), (ROWS, 1, BN)),
                        (m, 0, n),
                    )
                for n in tile(FFN, BN):  # noqa: F405
                    gv = tf.zeros(Tensor[(ROWS, BN), "f32", (ROWS, BN), "rmem"])
                    for k in tile(HID, BK):  # noqa: F405
                        wk = tf.reshape(w_gu[k, 1:2, n], (BK, BN))
                        xt = tf.reshard(xn1[0:ROWS, k], (ROWS, BK), "smem")
                        wt = tf.reshard(wk, (BK, BN), "smem")
                        mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                        gv = gv + tf.reshard(mm, (ROWS, BN), "rmem")
                    gb = tf.insert_slice(
                        gb,
                        tf.reshape(tf.reshard(gv, (ROWS, BN), "gmem"), (ROWS, 1, BN)),
                        (m, 1, n),
                    )

            acb = act
            for m in tile(SEQ, ROWS):  # noqa: F405
                for n in tile(FFN, BN):  # noqa: F405
                    gate = tf.reshard(tf.reshape(gb[m, 0:1, n], (ROWS, BN)), (ROWS, BN), "rmem")
                    up = tf.reshard(tf.reshape(gb[m, 1:2, n], (ROWS, BN)), (ROWS, BN), "rmem")
                    sw = tf.cast(tf.silu(gate) * up, dtype="bf16")
                    acb = tf.insert_slice(acb, tf.reshard(sw, (ROWS, BN), "gmem"), (m, n))

            o = out
            for m in tile(SEQ, ROWS):  # noqa: F405
                for n in tile(HID, BN):  # noqa: F405
                    dp = tf.zeros(Tensor[(ROWS, BN), "f32", (ROWS, BN), "rmem"])
                    for k in tile(FFN, BK):  # noqa: F405
                        xt = tf.reshard(acb[m, k], (ROWS, BK), "smem")
                        wt = tf.reshard(
                            tf.reshape(w_down[k, n], (BK, BN)),
                            (BK, BN),
                            "smem",
                        )
                        mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                        dp = dp + tf.reshard(mm, (ROWS, BN), "rmem")
                    hr = tf.reshard(hb[m, n], (ROWS, BN), "rmem")
                    sm = tf.cast(tf.cast(hr, dtype="f32") + dp, dtype="bf16")
                    o = tf.insert_slice(o, tf.reshard(sm, (ROWS, BN), "gmem"), (m, n))
        return o

    @func
    def layer_decode(
        x: Tensor[(SEQ, HID), "bf16"],
        w_qkv: ConstTensor[(HID, QKV_N), "bf16"],
        w_o: ConstTensor[(QN, HID), "bf16"],
        w_gu: ConstTensor[(HID, 2, FFN), "bf16"],
        w_down: ConstTensor[(FFN, HID), "bf16"],
        g_in: ConstTensor[(HID,), "bf16"],
        g_q: ConstTensor[(D,), "bf16"],
        g_k: ConstTensor[(D,), "bf16"],
        g_post: ConstTensor[(HID,), "bf16"],
        kc: Tensor[(1, CAP, HKV, D), "bf16"],
        vc: Tensor[(1, CAP, HKV, D), "bf16"],
        cur: Tensor[(), "i32"],
        cos: Tensor[(ROWS, D), "f32"],
        sin: Tensor[(ROWS, D), "f32"],
        pos: Tensor[(ROWS,), "i32"],
        xn: Tensor[(SEQ, HID), "bf16"],
        qkv: Tensor[(SEQ, QKV_N), "f32"],
        q: Tensor[(SEQ, HKV, G, D), "bf16"],
        attn: Tensor[(SEQ, HKV, G, D), "bf16"],
        h1: Tensor[(SEQ, HID), "bf16"],
        gu: Tensor[(SEQ, 2, FFN), "f32"],
        act: Tensor[(SEQ, FFN), "bf16"],
        out: Tensor[(SEQ, HID), "bf16"],
    ) -> Tensor[(SEQ, HID), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("cta",)) as _cta:
            xr = tf.rms_norm(x[0:1, 0:HID], g_in)
            xb = tf.insert_slice(xn, tf.reshard(xr, (1, HID), "gmem"), (0, 0))

            p = qkv
            for n in tile(QKV_N, DN_QKV):  # noqa: F405
                acc = tf.zeros(Tensor[(1, DN_QKV), "f32", (1, DN_QKV), "rmem"])
                for k in tile(HID, BK):  # noqa: F405
                    xt = tf.reshard(xb[0:1, k], (1, BK), "smem")
                    wt = tf.reshard(
                        tf.reshape(w_qkv[k, n], (BK, DN_QKV)),
                        (BK, DN_QKV),
                        "smem",
                    )
                    mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                    acc = acc + tf.reshard(mm, (1, DN_QKV), "rmem")
                p = tf.insert_slice(p, tf.reshard(acc, (1, DN_QKV), "gmem"), (0, n))

            q4 = tf.reshape(tf.cast(p[0:1, 0:QN], dtype="bf16"), (1, 1, HQ, D))
            k4 = tf.reshape(tf.cast(p[0:1, QN : QN + KN], dtype="bf16"), (1, 1, HKV, D))
            v4 = tf.reshape(tf.cast(p[0:1, QN + KN : QKV_N], dtype="bf16"), (1, 1, HKV, D))
            qr, kr = tf.rope(
                tf.rms_norm(q4, g_q),
                tf.rms_norm(k4, g_k),
                cos[0:1, 0:D],
                sin[0:1, 0:D],
                pos[0:1],
            )
            width = tf.reshape(cur, ())
            kc2 = tf.cache_update(kc, cur, width, kr)
            vc2 = tf.cache_update(vc, cur, width, v4)
            qb = tf.insert_slice(q, tf.reshape(qr, (1, HKV, G, D)), (0, 0, 0, 0))

            ab = attn
            for kv in tile(HKV, 1):  # noqa: F405
                for g in tile(G, 1):  # noqa: F405
                    qh = tf.reshard(tf.reshape(qb[0:1, kv, g, 0:D], (1, D)), (1, D), "smem")
                    mx = tf.zeros(Tensor[(1, 1), "f32", (1, 1), "rmem"]) - 30000.0
                    lr = tf.zeros(Tensor[(1, 1), "f32", (1, 1), "rmem"])
                    acc = tf.zeros(Tensor[(1, D), "f32", (1, D), "rmem"])
                    for c in tile(CTX + SEQ, BKV):  # noqa: F405
                        khb = tf.reshard(
                            tf.reshape(kc2[0:1, c, kv, 0:D], (BKV, D)), (BKV, D), "smem"
                        )
                        kh = tf.transpose(khb, perm=(1, 0))
                        vh = tf.reshard(
                            tf.reshape(vc2[0:1, c, kv, 0:D], (BKV, D)), (BKV, D), "smem"
                        )
                        sc = tf.reshard(tf.cast(tf.matmul(qh, kh), dtype="f32"), (1, BKV), "rmem")
                        mn = tf.maximum(mx, tf.reduce(sc, axes=(1,), keepdim=True, kind="max"))
                        scale = tf.exp(mx - mn)
                        pr = tf.exp(sc - mn)
                        lr = lr * scale + tf.reduce(pr, axes=(1,), keepdim=True, kind="sum")
                        pb = tf.reshard(tf.cast(pr, dtype="bf16"), (1, BKV), "smem")
                        pv = tf.cast(tf.matmul(pb, vh), dtype="f32")
                        acc = acc * scale + tf.reshard(pv, (1, D), "rmem")
                        mx = mn
                    ah = tf.reshard(tf.cast(tf.div(acc, lr), dtype="bf16"), (1, D), "gmem")
                    ab = tf.insert_slice(ab, tf.reshape(ah, (1, 1, 1, D)), (0, kv, g, 0))

            af = tf.reshape(ab, (SEQ, QN))
            hb = h1
            for n in tile(HID, DN_O):  # noqa: F405
                op = tf.zeros(Tensor[(1, DN_O), "f32", (1, DN_O), "rmem"])
                for k in tile(QN, BK):  # noqa: F405
                    xt = tf.reshard(af[0:1, k], (1, BK), "smem")
                    wt = tf.reshard(
                        tf.reshape(w_o[k, n], (BK, DN_O)),
                        (BK, DN_O),
                        "smem",
                    )
                    mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                    op = op + tf.reshard(mm, (1, DN_O), "rmem")
                xr = tf.reshard(x[0:1, n], (1, DN_O), "rmem")
                sm = tf.cast(tf.cast(xr, dtype="f32") + op, dtype="bf16")
                hb = tf.insert_slice(hb, tf.reshard(sm, (1, DN_O), "gmem"), (0, n))

            gb = gu
            xn1 = tf.rms_norm(hb[0:1, 0:HID], g_post)
            for n in tile(FFN, DN_FFN):  # noqa: F405
                gv = tf.zeros(Tensor[(1, DN_FFN), "f32", (1, DN_FFN), "rmem"])
                for k in tile(HID, BK):  # noqa: F405
                    xt = tf.reshard(xn1[0:1, k], (1, BK), "smem")
                    wt = tf.reshard(
                        tf.reshape(w_gu[k, 0:1, n], (BK, DN_FFN)),
                        (BK, DN_FFN),
                        "smem",
                    )
                    mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                    gv = gv + tf.reshard(mm, (1, DN_FFN), "rmem")
                gb = tf.insert_slice(
                    gb,
                    tf.reshape(tf.reshard(gv, (1, DN_FFN), "gmem"), (1, 1, DN_FFN)),
                    (0, 0, n),
                )
            for n in tile(FFN, DN_FFN):  # noqa: F405
                uv = tf.zeros(Tensor[(1, DN_FFN), "f32", (1, DN_FFN), "rmem"])
                for k in tile(HID, BK):  # noqa: F405
                    xt = tf.reshard(xn1[0:1, k], (1, BK), "smem")
                    wt = tf.reshard(
                        tf.reshape(w_gu[k, 1:2, n], (BK, DN_FFN)),
                        (BK, DN_FFN),
                        "smem",
                    )
                    mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                    uv = uv + tf.reshard(mm, (1, DN_FFN), "rmem")
                gb = tf.insert_slice(
                    gb,
                    tf.reshape(tf.reshard(uv, (1, DN_FFN), "gmem"), (1, 1, DN_FFN)),
                    (0, 1, n),
                )

            acb = act
            for n in tile(FFN, DN_FFN):  # noqa: F405
                gate = tf.reshard(tf.reshape(gb[0:1, 0:1, n], (1, DN_FFN)), (1, DN_FFN), "rmem")
                up = tf.reshard(tf.reshape(gb[0:1, 1:2, n], (1, DN_FFN)), (1, DN_FFN), "rmem")
                sw = tf.cast(tf.silu(gate) * up, dtype="bf16")
                acb = tf.insert_slice(acb, tf.reshard(sw, (1, DN_FFN), "gmem"), (0, n))

            o = out
            for n in tile(HID, DN_DOWN):  # noqa: F405
                dp = tf.zeros(Tensor[(1, DN_DOWN), "f32", (1, DN_DOWN), "rmem"])
                for k in tile(FFN, BK):  # noqa: F405
                    xt = tf.reshard(acb[0:1, k], (1, BK), "smem")
                    wt = tf.reshard(
                        tf.reshape(w_down[k, n], (BK, DN_DOWN)),
                        (BK, DN_DOWN),
                        "smem",
                    )
                    mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                    dp = dp + tf.reshard(mm, (1, DN_DOWN), "rmem")
                xr = tf.reshard(hb[0:1, n], (1, DN_DOWN), "rmem")
                sm = tf.cast(tf.cast(xr, dtype="f32") + dp, dtype="bf16")
                o = tf.insert_slice(o, tf.reshard(sm, (1, DN_DOWN), "gmem"), (0, n))
        return o

    @func
    def model(
        ids: Tensor[(SEQ,), "i32"],
        w_embed: ConstTensor[(V, HID), "bf16"],
        w_qkv: ConstTensor[(L, HID, QKV_N), "bf16"],
        w_o: ConstTensor[(L, QN, HID), "bf16"],
        w_gu: ConstTensor[(L, HID, 2, FFN), "bf16"],
        w_down: ConstTensor[(L, FFN, HID), "bf16"],
        g_in: ConstTensor[(L, HID), "bf16"],
        g_q: ConstTensor[(L, D), "bf16"],
        g_k: ConstTensor[(L, D), "bf16"],
        g_post: ConstTensor[(L, HID), "bf16"],
        g_final: ConstTensor[(HID,), "bf16"],
        w_head: ConstTensor[(HID, V), "bf16"],
        kc: Tensor[(L, CAP, HKV, D), "bf16"],
        vc: Tensor[(L, CAP, HKV, D), "bf16"],
        cur: Tensor[(), "i32"],
        cos: Tensor[(ROWS, D), "f32"],
        sin: Tensor[(ROWS, D), "f32"],
        pos: Tensor[(ROWS,), "i32"],
        x: Tensor[(SEQ, HID), "bf16"],
        xn: Tensor[(SEQ, HID), "bf16"],
        qkv: Tensor[(SEQ, QKV_N), "f32"],
        q: Tensor[(SEQ, HKV, G, D), "bf16"],
        attn: Tensor[(SEQ, HKV, G, D), "bf16"],
        h1: Tensor[(SEQ, HID), "bf16"],
        gu: Tensor[(SEQ, 2, FFN), "f32"],
        act: Tensor[(SEQ, FFN), "bf16"],
        out: Tensor[(SEQ, HID), "bf16"],
        logits: Tensor[(SEQ, V), "f32"],
    ) -> Tensor[(SEQ, V), "f32"]:
        pass

    @model.specialize(DimVarRangePat("seq", 2, 8193))  # noqa: F821
    def prefill(
        ids: Tensor[(SEQ,), "i32"],
        w_embed: ConstTensor[(V, HID), "bf16"],
        w_qkv: ConstTensor[(L, HID, QKV_N), "bf16"],
        w_o: ConstTensor[(L, QN, HID), "bf16"],
        w_gu: ConstTensor[(L, HID, 2, FFN), "bf16"],
        w_down: ConstTensor[(L, FFN, HID), "bf16"],
        g_in: ConstTensor[(L, HID), "bf16"],
        g_q: ConstTensor[(L, D), "bf16"],
        g_k: ConstTensor[(L, D), "bf16"],
        g_post: ConstTensor[(L, HID), "bf16"],
        g_final: ConstTensor[(HID,), "bf16"],
        w_head: ConstTensor[(HID, V), "bf16"],
        kc: Tensor[(L, CAP, HKV, D), "bf16"],
        vc: Tensor[(L, CAP, HKV, D), "bf16"],
        cur: Tensor[(), "i32"],
        cos: Tensor[(ROWS, D), "f32"],
        sin: Tensor[(ROWS, D), "f32"],
        pos: Tensor[(ROWS,), "i32"],
        x: Tensor[(SEQ, HID), "bf16"],
        xn: Tensor[(SEQ, HID), "bf16"],
        qkv: Tensor[(SEQ, QKV_N), "f32"],
        q: Tensor[(SEQ, HKV, G, D), "bf16"],
        attn: Tensor[(SEQ, HKV, G, D), "bf16"],
        h1: Tensor[(SEQ, HID), "bf16"],
        gu: Tensor[(SEQ, 2, FFN), "f32"],
        act: Tensor[(SEQ, FFN), "bf16"],
        out: Tensor[(SEQ, HID), "bf16"],
        logits: Tensor[(SEQ, V), "f32"],
    ) -> Tensor[(SEQ, V), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("cta",)) as _cta:
            h = x
            for m in tile(SEQ, ROWS):  # noqa: F405
                h = tf.insert_slice(
                    h,
                    tf.reshard(
                        tf.index_select(w_embed, ids[m], dim=0),
                        (ROWS, HID),
                        "gmem",
                    ),
                    (m, 0),
                )

            for i in tile(L, 1):  # noqa: F405
                h = layer_prefill(  # noqa: F405
                    h,
                    tf.reshape(w_qkv[i, 0:HID, 0:QKV_N], (HID, QKV_N)),
                    tf.reshape(w_o[i, 0:QN, 0:HID], (QN, HID)),
                    tf.reshape(w_gu[i, 0:HID, 0:2, 0:FFN], (HID, 2, FFN)),
                    tf.reshape(w_down[i, 0:FFN, 0:HID], (FFN, HID)),
                    tf.reshape(g_in[i, 0:HID], (HID,)),
                    tf.reshape(g_q[i, 0:D], (D,)),
                    tf.reshape(g_k[i, 0:D], (D,)),
                    tf.reshape(g_post[i, 0:HID], (HID,)),
                    kc[i, 0:CAP, 0:HKV, 0:D],
                    vc[i, 0:CAP, 0:HKV, 0:D],
                    cur,
                    cos,
                    sin,
                    pos,
                    xn,
                    qkv,
                    q,
                    attn,
                    h1,
                    gu,
                    act,
                    out,
                )

            lg = logits
            for m in tile(SEQ, ROWS):  # noqa: F405
                hf = tf.rms_norm(h[m, 0:HID], g_final)
                for n in tile(V, BN):  # noqa: F405
                    acc = tf.zeros(Tensor[(ROWS, BN), "f32", (ROWS, BN), "rmem"])
                    for k in tile(HID, BK):  # noqa: F405
                        xt = tf.reshard(hf[0:ROWS, k], (ROWS, BK), "smem")
                        wt = tf.reshard(
                            w_head[k, n],
                            (BK, BN),
                            "smem",
                        )
                        mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                        acc = acc + tf.reshard(mm, (ROWS, BN), "rmem")
                    lg = tf.insert_slice(lg, tf.reshard(acc, (ROWS, BN), "gmem"), (m, n))
        return lg

    @model.specialize(DimVarRangePat("seq", 1, 2))  # noqa: F821
    def decode(
        ids: Tensor[(SEQ,), "i32"],
        w_embed: ConstTensor[(V, HID), "bf16"],
        w_qkv: ConstTensor[(L, HID, QKV_N), "bf16"],
        w_o: ConstTensor[(L, QN, HID), "bf16"],
        w_gu: ConstTensor[(L, HID, 2, FFN), "bf16"],
        w_down: ConstTensor[(L, FFN, HID), "bf16"],
        g_in: ConstTensor[(L, HID), "bf16"],
        g_q: ConstTensor[(L, D), "bf16"],
        g_k: ConstTensor[(L, D), "bf16"],
        g_post: ConstTensor[(L, HID), "bf16"],
        g_final: ConstTensor[(HID,), "bf16"],
        w_head: ConstTensor[(HID, V), "bf16"],
        kc: Tensor[(L, CAP, HKV, D), "bf16"],
        vc: Tensor[(L, CAP, HKV, D), "bf16"],
        cur: Tensor[(), "i32"],
        cos: Tensor[(ROWS, D), "f32"],
        sin: Tensor[(ROWS, D), "f32"],
        pos: Tensor[(ROWS,), "i32"],
        x: Tensor[(SEQ, HID), "bf16"],
        xn: Tensor[(SEQ, HID), "bf16"],
        qkv: Tensor[(SEQ, QKV_N), "f32"],
        q: Tensor[(SEQ, HKV, G, D), "bf16"],
        attn: Tensor[(SEQ, HKV, G, D), "bf16"],
        h1: Tensor[(SEQ, HID), "bf16"],
        gu: Tensor[(SEQ, 2, FFN), "f32"],
        act: Tensor[(SEQ, FFN), "bf16"],
        out: Tensor[(SEQ, HID), "bf16"],
        logits: Tensor[(SEQ, V), "f32"],
    ) -> Tensor[(SEQ, V), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("cta",)) as _cta:
            e = tf.reshard(tf.index_select(w_embed, ids[0:1], dim=0), (1, HID), "gmem")
            h = tf.insert_slice(x, e, (0, 0))

            for i in tile(L, 1):  # noqa: F405
                h = layer_decode(  # noqa: F405
                    h,
                    tf.reshape(w_qkv[i, 0:HID, 0:QKV_N], (HID, QKV_N)),
                    tf.reshape(w_o[i, 0:QN, 0:HID], (QN, HID)),
                    tf.reshape(w_gu[i, 0:HID, 0:2, 0:FFN], (HID, 2, FFN)),
                    tf.reshape(w_down[i, 0:FFN, 0:HID], (FFN, HID)),
                    tf.reshape(g_in[i, 0:HID], (HID,)),
                    tf.reshape(g_q[i, 0:D], (D,)),
                    tf.reshape(g_k[i, 0:D], (D,)),
                    tf.reshape(g_post[i, 0:HID], (HID,)),
                    kc[i, 0:CAP, 0:HKV, 0:D],
                    vc[i, 0:CAP, 0:HKV, 0:D],
                    cur,
                    cos,
                    sin,
                    pos,
                    xn,
                    qkv,
                    q,
                    attn,
                    h1,
                    gu,
                    act,
                    out,
                )

            lg = logits
            hf = tf.rms_norm(h[0:1, 0:HID], g_final)
            for n in tile(V, BN):  # noqa: F405
                acc = tf.zeros(Tensor[(1, BN), "f32", (1, BN), "rmem"])
                for k in tile(HID, BK):  # noqa: F405
                    xt = tf.reshard(hf[0:1, k], (1, BK), "smem")
                    wt = tf.reshard(w_head[k, n], (BK, BN), "smem")
                    mm = tf.cast(tf.matmul(xt, wt), dtype="f32")
                    acc = acc + tf.reshard(mm, (1, BN), "rmem")
                lg = tf.insert_slice(lg, tf.reshard(acc, (1, BN), "gmem"), (0, n))
        return lg
