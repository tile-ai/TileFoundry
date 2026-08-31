#!/usr/bin/env python
"""Emit model.py: the whole Nemotron-3.5-Lightning decode step as one authored @func.

The step is one program, so its 52 layers are written out rather than looped: a
`for` in a @func body is a runtime loop over one weight tensor, and these layers
neither share weights nor share a shape. This file is the only place the
repetition lives; `model.py` is what everything else reads.

`--variant N` selects one rung of the placement ladder; each rung is meant to be
regenerated and re-`analyze`d, which is how the one that ships was chosen:

    0  one cta Mesh over the whole step, nothing placed on it: every value in
       gmem and every CTA doing all of the work
    1  every projection's output axis split over the mesh, its weight slice staged
       to smem -- one stage per projection, the reshard back to gmem is its barrier
    2  the attention scan blocked: K/V stream through smem, online (m, l, acc)
       state, so no score row the length of the context is ever materialised
    3  the MoE expert stage on a mesh of its own, since 1856 has no divisor near
       the grid and variant 1 therefore left the experts unplaced
    4  the KV scan split across a second mesh axis, with an explicit log-sum-exp
       combine, so long context divides over the grid rather than over one CTA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Shapes come from the config.json **in this directory**, not from the
#: checkpoint's: the two are identical key for key, and keeping a copy here means
#: generation does not need the weights to be present at all.
CFG = json.loads((Path(__file__).parent / "config.json").read_text())
KINDS = [{"mamba": "linear_attention", "attention": "full_attention"}.get(k, k)
         for k in CFG["layers_block_type"]]

E = CFG["n_routed_experts"]
K = CFG["num_experts_per_tok"]

#: The mesh extent. A `with Mesh` scope has to cover its topology level exactly,
#: and a Split axis has to divide by the mesh extent, so the extent has to divide
#: every output axis this program places: 2688, 4096, 6144, 3712, 1856, 256, 128
#: and 131072. Their gcd is 64. 132 -- the target's SM count -- divides none of
#: them, and 128 misses 1856 = 2**6 * 29, which is the MoE expert width. So the
#: HIR states a 64-wide division and the kernel splits finer, with a ragged
#: remainder the divisibility rule here cannot express.
U = 64
UE = U
#: Attention KV blocking: the K and V tile a scan step stages into smem.
#: 512 x 128 bf16 per head is 128 KB for K and V together at HKV=2.
ABLK = 128
#: Long-context KV workers: the second axis of the variant-4 attention mesh.
#: 2 KV heads x 32 workers is 64, the same width as every other stage.
WRK = 32
#: The short-context placement's worker axis, and where the two change over.
WRKH = 4
CROSSOVER = 2048


def sig_states(i, kind):
    if kind == "linear_attention":
        return [(f"l{i}_conv_state", "(1, CONV, WIN)", "_DT"),
                (f"l{i}_ssm_state", "(1, MH, MD, SS)", '"f32"')]
    if kind == "full_attention":
        return [(f"l{i}_k_cache", "(1, CF, HKV, DH)", "_DT"),
                (f"l{i}_v_cache", "(1, CF, HKV, DH)", "_DT"),
                (f"l{i}_k_tail", "(1, CT, HKV, DH)", "_DT"),
                (f"l{i}_v_tail", "(1, CT, HKV, DH)", "_DT")]
    return []


def sig_consts(i, kind):
    if kind == "linear_attention":
        return [(f"l{i}_w_in", "(PROJ, H)"), (f"l{i}_conv_w", "(CONV, KER)"),
                (f"l{i}_conv_b", "(CONV,)"), (f"l{i}_a_log", "(MH,)"),
                (f"l{i}_dt_bias", "(MH,)"), (f"l{i}_d_skip", "(MH,)"),
                (f"l{i}_gamma_gdn", "(MI,)"), (f"l{i}_w_out", "(H, MI)")]
    if kind == "full_attention":
        return [(f"l{i}_w_q", "(QP, H)"), (f"l{i}_w_k", "(KVP, H)"),
                (f"l{i}_w_v", "(KVP, H)"), (f"l{i}_w_o", "(H, QP)")]
    return [(f"l{i}_w_router", "(E, H)"), (f"l{i}_w_up", "(E, I, H)"),
            (f"l{i}_w_down", "(E, H, I)"), (f"l{i}_w_sh_up", "(IS, H)"),
            (f"l{i}_w_sh_down", "(H, IS)")]


def fresh(i, kind):
    if kind == "linear_attention":
        return [f"l{i}_conv_out", f"l{i}_ssm_out"]
    if kind == "full_attention":
        return [f"l{i}_k_new", f"l{i}_v_new"]
    return []


def proj(var, out, x, w, n, shardable=True, kdim="H", nk=True, axis="cta.u"):
    """`out = x @ w` contracted over *kdim*, placed as *var* asks.

    *nk* says how `w` is stored: True for `(out, in)` -- what HF stores for every
    "up" projection, read with ``b_layout="NK"`` so a CTA owning output rows
    reads contiguous bytes -- and False for `(in, out)`, which is how the "down"
    projections are stored so a CTA owning a slice of the contraction reads
    contiguously instead.

    v0  one matmul, everything in gmem: one CTA's share of the work is all of it.
    v1+ the output axis split over the mesh. The split is a same-storage reshard,
        a zero-copy view that moves nothing and only says who owns what; the
        reshard that *completes* it afterwards is the stage's barrier, because it
        reads a gmem shard produced under a different CTA ownership.
    """
    b = ", b_layout=\"NK\"" if nk else ""
    if var == 0 or not shardable:
        return [f"{out} = tf.matmul({x}, {w}{b})"]
    shape = f"({n} @ {axis}, {kdim})" if nk else f"({kdim}, {n} @ {axis})"
    return [
        f"{out}_w = tf.reshard({w}, {shape}, \"gmem\")",
        f"{out}_s = tf.matmul({x}, {out}_w{b})",
        f"{out} = tf.reshard({out}_s, (1, {n}), \"gmem\")",
    ]


def prenorm(i):
    p = f"l{i}"
    return [
        f"{p}_xf = tf.cast(h, dtype=\"f32\")",
        f"{p}_ms = tf.reduce(tf.square({p}_xf), axes=(-1,), keepdim=True, kind=\"mean\")",
        f"{p}_nz = tf.cast({p}_xf * tf.rsqrt({p}_ms + tf.full_like({p}_ms, value=EPS)),"
        f" dtype=_DT)",
        f"{p}_h2 = tf.reshape({p}_nz * tf.reshape({p}_gamma, new_shape=(1, 1, H)),"
        f" new_shape=(1, H))",
    ]


def mamba_body(i, var):
    p = f"l{i}"
    out = []
    # The three halves of `in_proj` are consumed by three different things, so
    # they are three matmuls over three row windows of one stored weight.
    out += proj(var, f"{p}_gate", f"{p}_h2", f"{p}_w_in[0:MI, :]", "MI")
    out += proj(var, f"{p}_col0", f"{p}_h2", f"{p}_w_in[MI:MI + CONV, :]", "CONV")
    out += proj(var, f"{p}_dt", f"{p}_h2", f"{p}_w_in[MI + CONV:PROJ, :]", "MH",
                shardable=False)
    out += [
        f"{p}_col = tf.reshape({p}_col0, new_shape=(1, CONV, 1))",
        f"{p}_win = tf.concat([{p}_conv_state, {p}_col], axis=2)",
        f"{p}_conv_out = {p}_win[:, :, 1:KER]",
        f"{p}_cs = tf.reduce({p}_win * tf.reshape({p}_conv_w, new_shape=(1, CONV, KER)),"
        f" axes=(-1,), keepdim=False, kind=\"sum\")",
        f"{p}_xbc = tf.silu({p}_cs + tf.reshape({p}_conv_b, new_shape=(1, CONV)))",
        f"{p}_x = tf.reshape({p}_xbc[:, 0:MI], new_shape=(1, MH, MD))",
        f"{p}_bg = tf.reshape({p}_xbc[:, MI:MI + NG * SS], new_shape=(1, NG, SS))",
        f"{p}_cg = tf.reshape({p}_xbc[:, MI + NG * SS:CONV], new_shape=(1, NG, SS))",
        f"{p}_b = tf.reshape(tf.repeat_interleave({p}_bg, repeats=HPG, axis=1),"
        f" new_shape=(1, MH, 1, SS))",
        f"{p}_c = tf.reshape(tf.repeat_interleave({p}_cg, repeats=HPG, axis=1),"
        f" new_shape=(1, MH, SS, 1))",
        f"{p}_dta = tf.clamp(tf.softplus(tf.reshape({p}_dt, new_shape=(1, MH, 1))"
        f" + tf.reshape({p}_dt_bias, new_shape=(1, MH, 1))),"
        f" min_val=DTMIN, max_val=FMAX)",
        f"{p}_dte = tf.reshape({p}_dta, new_shape=(1, MH, 1, 1))",
        f"{p}_an = tf.reshape(-tf.exp(tf.cast({p}_a_log, dtype=\"f32\")),"
        f" new_shape=(1, MH, 1, 1))",
        f"{p}_da = tf.exp(tf.cast({p}_dte, dtype=\"f32\") * {p}_an)",
        f"{p}_dbx = tf.cast(({p}_dte * {p}_b) * tf.reshape({p}_x, new_shape=(1, MH, MD, 1)),"
        f" dtype=\"f32\")",
        f"{p}_ssm_out = {p}_ssm_state * {p}_da + {p}_dbx",
        f"{p}_y = tf.reshape(tf.matmul(tf.cast({p}_ssm_out, dtype=_DT),"
        f" tf.cast({p}_c, dtype=_DT)), new_shape=(1, MH, MD))",
        f"{p}_yd = {p}_y + {p}_x * tf.reshape({p}_d_skip, new_shape=(1, MH, 1))",
        f"{p}_yf = tf.cast(tf.reshape({p}_yd, new_shape=(1, MI)), dtype=\"f32\")"
        f" * tf.silu(tf.cast({p}_gate, dtype=\"f32\"))",
        f"{p}_yg = tf.reshape({p}_yf, new_shape=(1, NG, GRP))",
        f"{p}_gms = tf.reduce(tf.square({p}_yg), axes=(-1,), keepdim=True, kind=\"mean\")",
        f"{p}_yn = tf.reshape({p}_yg * tf.rsqrt({p}_gms + tf.full_like({p}_gms, value=EPS)),"
        f" new_shape=(1, MI))",
        f"{p}_scan = tf.reshape({p}_gamma_gdn, new_shape=(1, MI))"
        f" * tf.cast({p}_yn, dtype=_DT)",
    ]
    out += proj(var, f"{p}_mx", f"{p}_scan", f"{p}_w_out", "H", kdim="MI")
    out += [f"{p}_mix = tf.reshape({p}_mx, new_shape=(1, 1, H))"]
    return out


def attn_body(i, var):
    p = f"l{i}"
    out = []
    out += proj(var, f"{p}_q0", f"{p}_h2", f"{p}_w_q", "QP")
    out += proj(var, f"{p}_k0", f"{p}_h2", f"{p}_w_k", "KVP")
    out += proj(var, f"{p}_v0", f"{p}_h2", f"{p}_w_v", "KVP")
    out += [
        f"{p}_q = tf.reshape({p}_q0, new_shape=(1, 1, HQ, DH))",
        f"{p}_k_new = tf.reshape({p}_k0, new_shape=(1, 1, HKV, DH))",
        f"{p}_v_new = tf.reshape({p}_v0, new_shape=(1, 1, HKV, DH))",
        # This token's row lands in the remainder view: it is the last position
        # that exists, and the whole blocks behind it are read-only.
        f"{p}_kta = tf.cache_update({p}_k_tail, cur_pos, one, {p}_k_new)",
        f"{p}_vta = tf.cache_update({p}_v_tail, cur_pos, one, {p}_v_new)",
        f"{p}_ka = {p}_k_cache",
        f"{p}_va = {p}_v_cache",
        # Group-query attention without expanding the cache: the 16 query heads
        # of a group are the M rows of one matmul against their own K/V head.
        f"{p}_qg = tf.reshape({p}_q, new_shape=(1, HKV, GQA, DH))",
    ]
    if var >= 5:
        # The scan is a call now, and the placement it runs is stated beside it:
        # `attend` is the prototype the two `DimVarRangePat` variants hang off,
        # and it is what the runtime keys its own two bodies on.
        #
        # The step calls the long placement rather than the prototype, and not
        # because it wants only that one. Specialising a caller at a bound
        # dimension refuses to rebuild through a callee that has variants of its
        # own ("the callee dispatches on its own variants, which this rebuild
        # does not choose"), and the other way round -- putting the variants on
        # the entry, which is the shape the authoring tutorial shows -- needs a
        # return annotation on the prototype, which a step that returns 59
        # tensors has no way to write. So the dispatch stands where it can be
        # read and checked (`check model.py:attend --dim ctx_full=0,4096`), and
        # the body the step names is the one that runs at the lengths this is
        # about. See ISSUES.md; both limits have a repro under repro/.
        out += [f"{p}_ctx = attend_by_context({p}_qg, {p}_k_cache, {p}_v_cache,"
                f" {p}_kta, {p}_vta)"]
        out += proj(var, f"{p}_mx", f"{p}_ctx", f"{p}_w_o", "H", kdim="QP")
        out += [f"{p}_mix = tf.reshape({p}_mx, new_shape=(1, 1, H))"]
        return out
    out += _online_scan(p, var)
    if var >= 4:
        out += [
            f"{p}_am = tf.reshard({p}_m, (WRK, HKV @ kv.g, GQA, 1), \"smem\")",
            f"{p}_al = tf.reshard({p}_l, (WRK, HKV @ kv.g, GQA, 1), \"smem\")",
            f"{p}_aa = tf.reshard({p}_acc, (WRK, HKV @ kv.g, GQA, DH), \"smem\")",
            f"{p}_gm = tf.reduce({p}_am, axes=(0,), keepdim=False, kind=\"max\")",
            f"{p}_cw = tf.exp({p}_am - {p}_gm)",
            f"{p}_gl = tf.reduce({p}_cw * {p}_al, axes=(0,), keepdim=False, kind=\"sum\")",
            f"{p}_ga = tf.reduce({p}_cw * {p}_aa, axes=(0,), keepdim=False, kind=\"sum\")",
            f"{p}_ct = tf.cast({p}_ga / {p}_gl, dtype=_DT)",
        ]
    else:
        out += [f"{p}_ct = tf.cast({p}_acc / {p}_l, dtype=_DT)"]
    out += [
        f"{p}_ctx = tf.reshape(tf.reshard({p}_ct, (1, HKV, GQA, DH), \"gmem\"),"
        f" new_shape=(1, QP))",
    ]
    out += proj(var, f"{p}_mx", f"{p}_ctx", f"{p}_w_o", "H", kdim="QP")
    out += [f"{p}_mix = tf.reshape({p}_mx, new_shape=(1, 1, H))"]
    return out


def _scan_block(p, sfx, src_k, src_v, base, blk, size, smem):
    """One block of the online-softmax merge: stage K/V, score it, merge it in.

    *smem* says whether the block's K, V and scores are staged on chip. At
    variant 0 and 1 they are not, so a score row the length of the context lands
    in gmem; from variant 2 the block is what streams and nothing that long is
    ever written.
    """
    q = f"{p}_qs" if smem else f"{p}_qg"
    if smem:
        stage = [
            f"{p}_kb{sfx} = tf.reshard(tf.transpose({src_k}[:, {base}:{base} + {size}, :, :],"
            f" perm=(0, 2, 1, 3)), {blk}, \"smem\")",
            f"{p}_vb{sfx} = tf.reshard(tf.transpose({src_v}[:, {base}:{base} + {size}, :, :],"
            f" perm=(0, 2, 1, 3)), {blk}, \"smem\")",
        ]
    else:
        stage = [
            f"{p}_kb{sfx} = tf.transpose({src_k}[:, {base}:{base} + {size}, :, :],"
            f" perm=(0, 2, 1, 3))",
            f"{p}_vb{sfx} = tf.transpose({src_v}[:, {base}:{base} + {size}, :, :],"
            f" perm=(0, 2, 1, 3))",
        ]
    return stage + [
        f"{p}_rw{sfx} = tf.cast(tf.matmul({q}, {p}_kb{sfx}, b_layout=\"NK\"), dtype=\"f32\")",
        f"{p}_sb{sfx} = {p}_rw{sfx} * tf.full_like({p}_rw{sfx}, value=QSCALE)",
        f"{p}_bm{sfx} = tf.reduce({p}_sb{sfx}, axes=(-1,), keepdim=True, kind=\"max\")",
        f"{p}_nm{sfx} = tf.max({p}_m, {p}_bm{sfx})",
        f"{p}_cr{sfx} = tf.exp({p}_m - {p}_nm{sfx})",
        f"{p}_pw{sfx} = tf.exp({p}_sb{sfx} - {p}_nm{sfx})",
        f"{p}_l = {p}_l * {p}_cr{sfx} + tf.reduce({p}_pw{sfx}, axes=(-1,), keepdim=True,"
        f" kind=\"sum\")",
        f"{p}_acc = {p}_acc * {p}_cr{sfx} + tf.cast(tf.matmul(tf.cast({p}_pw{sfx}, dtype=_DT),"
        f" {p}_vb{sfx}), dtype=\"f32\")",
        f"{p}_m = {p}_nm{sfx}",
    ]


def _online_scan(p, var):
    """The scan over one attention layer's context.

    The context arrives as two views of one buffer: `CF` positions in whole
    ABLK-sized blocks, and `CT` more that did not fill a block. Splitting it that
    way on the host is what lets the loop have a static block size at every
    context length -- a `tile` whose last window ran past the end would be a
    window the evaluator refuses, and padding the cache would need a mask this
    IR has no way to state.

    v0/v1  two blocks, nothing staged: the whole-context score row is a gmem
           value, which is what the traffic report then charges for.
    v2/v3  the whole blocks looped one ABLK at a time through smem.
    v4     the same loop striped over a worker axis and combined afterwards.
    """
    smem = var >= 2
    if var >= 4:
        st = "(WRK @ kv.w, HKV @ kv.g, GQA, 1)"
        ac = "(WRK @ kv.w, HKV @ kv.g, GQA, DH)"
        qs = "(1, HKV @ kv.g, GQA, DH)"
        blk = "(1, HKV @ kv.g, ABLK, DH)"
        tblk = "(1, HKV @ kv.g, CT, DH)"
        head = [f"for {p}_t in tile(CF, ABLK * WRK):",
                f"    {p}_b0 = {p}_t + kv.w * ABLK"]
    else:
        st, ac = "(1, HKV, GQA, 1)", "(1, HKV, GQA, DH)"
        qs = "(1, HKV, GQA, DH)"
        blk = "(1, HKV, ABLK, DH)"
        tblk = "(1, HKV, CT, DH)"
        if smem:
            head = [f"for {p}_t in tile(CF, ABLK):", f"    {p}_b0 = {p}_t + 0"]
        else:
            head = None
    out = []
    if smem:
        out.append(f"{p}_qs = tf.reshard({p}_qg, {qs}, \"smem\")")
    out += [
        f"{p}_m0 = tf.zeros(Tensor[{st}, \"f32\", {'\"smem\"' if smem else '\"gmem\"'}])",
        f"{p}_a0 = tf.zeros(Tensor[{ac}, \"f32\", {'\"smem\"' if smem else '\"gmem\"'}])",
        f"{p}_m = tf.full_like({p}_m0, value=NEGINF)",
        f"{p}_l = tf.full_like({p}_m0, value=0.0)",
        f"{p}_acc = tf.full_like({p}_a0, value=0.0)",
    ]
    if head is None:
        out += _scan_block(p, "f", f"{p}_ka", f"{p}_va", "0", "(1, HKV, CF, DH)", "CF", False)
    else:
        out += head
        out += ["    " + ln for ln in
                _scan_block(p, "", f"{p}_ka", f"{p}_va", f"{p}_b0", blk, "ABLK", smem)]
    out += _scan_block(p, "t", f"{p}_kta", f"{p}_vta", "0", tblk, "CT", smem)
    return out


def moe_body(i, var):
    p = f"l{i}"
    lines = [
        f"{p}_lg = tf.matmul(tf.cast({p}_h2, dtype=\"f32\"),"
        f" tf.cast({p}_w_router, dtype=\"f32\"), b_layout=\"NK\")",
        f"{p}_sig = tf.sigmoid({p}_lg)",
        f"{p}_ch = {p}_sig + tf.reshape({p}_e_bias, new_shape=(1, E))",
        f"{p}_tv, {p}_ti = tf.topk({p}_ch, k=K, axis=-1, sorted=False)",
        f"{p}_flat = tf.reshape({p}_ti, new_shape=(K,))",
        f"{p}_pick = tf.reshape(tf.index_select(tf.reshape({p}_sig, new_shape=(E,)),"
        f" {p}_flat, dim=0), new_shape=(1, K))",
        f"{p}_den = tf.reduce({p}_pick, axes=(-1,), keepdim=True, kind=\"sum\")",
        f"{p}_gw = ({p}_pick / ({p}_den + tf.full_like({p}_den, value=1e-20)))"
        f" * tf.full_like({p}_den, value=RSCALE)",
    ]
    # One routed slot at a time. `w_up[e:e + 1]` with a runtime `e` is a window
    # whose consumer accounts for what it moves, so the traffic charged is the
    # one expert this token routed to -- not the whole 128-expert bank an
    # `index_select` would have to admit it could reach.
    for j in range(K):
        lines += [
            f"{p}_e{j} = {p}_flat[{j}]",
            f"{p}_u{j} = tf.reshape({p}_w_up[{p}_e{j}:{p}_e{j} + 1, :, :], new_shape=(I, H))",
            f"{p}_d{j} = tf.reshape({p}_w_down[{p}_e{j}:{p}_e{j} + 1, :, :], new_shape=(H, I))",
        ]
        if var < 3:
            lines += [
                f"{p}_m{j} = tf.square(tf.relu(tf.matmul({p}_h2, {p}_u{j}, b_layout=\"NK\")))",
                f"{p}_r{j} = tf.cast(tf.reshape(tf.matmul({p}_m{j}, {p}_d{j}, b_layout=\"NK\"),"
                f" new_shape=(H,)), dtype=\"f32\")",
            ]
        else:
            # Both halves of an expert split on the same axis of its own mesh: a
            # CTA owns a slice of the intermediate, computes it from the whole
            # token, and contracts its own slice away again. The `down` weight is
            # stored (in, out) so that slice is contiguous bytes.
            lines += [
                f"{p}_uw{j} = tf.reshard({p}_u{j}, (I @ cta.u, H), \"gmem\")",
                f"{p}_us{j} = tf.square(tf.relu(tf.matmul({p}_h2, {p}_uw{j},"
                f" b_layout=\"NK\")))",
                # The down half contracts the whole intermediate, so the split
                # the up half made has to be completed first: this reshard is
                # the barrier between an expert's two matmuls.
                f"{p}_ms{j} = tf.reshard({p}_us{j}, (1, I), \"gmem\")",
                f"{p}_dw{j} = tf.reshard({p}_d{j}, (H @ cta.u, I), \"gmem\")",
                f"{p}_rs{j} = tf.matmul({p}_ms{j}, {p}_dw{j}, b_layout=\"NK\")",
                f"{p}_r{j} = tf.cast(tf.reshape(tf.reshard({p}_rs{j}, (1, H), \"gmem\"),"
                f" new_shape=(H,)), dtype=\"f32\")",
            ]
        lines += [f"{p}_a{j} = {p}_r{j} * tf.reshape({p}_gw[:, {j}:{j} + 1], new_shape=(1,))"]
    acc = " + ".join(f"{p}_a{j}" for j in range(K))
    lines += [f"{p}_sum = tf.cast({acc}, dtype=_DT)"]
    # The shared expert is ungated like the routed ones: up, relu squared, down.
    lines += proj(var, f"{p}_smid0", f"{p}_h2", f"{p}_w_sh_up", "IS")
    lines += [f"{p}_smid = tf.square(tf.relu({p}_smid0))"]
    lines += proj(var, f"{p}_sh0", f"{p}_smid", f"{p}_w_sh_down", "H", kdim="IS")
    lines += [
        f"{p}_sh = tf.reshape({p}_sh0, new_shape=(H,))",
        f"{p}_mix = tf.reshape({p}_sum + {p}_sh, new_shape=(1, 1, H))",
    ]
    return lines


BODY = {"linear_attention": mamba_body, "full_attention": attn_body, "moe": moe_body}

HEADER = '''"""Nemotron-3.5-Lightning-30B-A3B as one authored HIR program: the decode step.

Generated by ``gen_model.py --variant {variant}``; edit the generator, not this file.

**One program.** `decode_step` is the whole step -- embedding, all {nlayers}
layers, the closing norm, the head -- in a single `@func`. Not one `@func` per
layer and not a Python loop over layers: a `@func` boundary is a device call
whose two sides cannot share a staged tile, and a `for` in an HIR body is a
runtime loop over one weight tensor, while these layers hold {nmamba} Mamba-2,
{nattn} attention and {nmoe} MoE mixers with no shared weight between them.
Writing them out is what lets a placement decision reach across a layer
boundary.

**Stage boundaries.** Inside the one program the stages are separated by
`tf.reshard` calls that change which mesh position owns a value. A reshard that
reads a gmem shard produced under a different `cta` ownership lowers to a grid
barrier ahead of the read (`tilefoundry spec hir`, Reshard -- "Cross-CTA
fence"), so a stage boundary is a property of the dataflow written here rather
than a separately authored statement. Each `tf.reshard(..., "gmem")` that closes
a projection is one such boundary, and the runtime twin's kernel puts a
`T.sync_grid()` at exactly those points.

**State.** Each Mamba layer's convolution window and SSM matrix are ordinary
`Tensor` parameters, handed in and handed back. Each attention layer's K/V cache
is handed in at its *current* length -- `ctx_len` is the DimVar the whole program
is dispatched on -- and `tf.cache_update` writes this token's row at `cur_pos`,
which lowering may realise on the cache's own buffer. Nothing is copied or
re-laid-out between steps.

**Weight layout.** Every weight is declared in the layout the published
checkpoint stores -- `(out, in)` -- and read with ``b_layout="NK"``. That is the
value `torch.nn.functional.linear` computes, and it is the layout this
placement wants: every stage splits its *output* axis over the mesh, so a CTA
owning a slice of the output rows reads a contiguous run of bytes, and no stage
has to accumulate a partial sum across CTAs. `hf_alias.py` is therefore almost
all plain renames.
"""
from __future__ import annotations

import json
from pathlib import Path

from tilefoundry import func, module
from tilefoundry.dsl import (  # noqa: F401
    ConstTensor, DimVar, DimVarRangePat, Mesh, Tensor, tf,
)
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


def published(path: Path | None = None) -> dict:
    """The checkpoint's own configuration, read from the file beside this module."""
    path = Path(__file__).parent / "config.json" if path is None else path
    return json.loads(path.read_text(encoding="utf-8"))


config = published()

#: Published layer kinds, with the checkpoint's legacy spellings mapped to the
#: names `transformers` uses.
LAYER_KINDS = [{{"mamba": "linear_attention", "attention": "full_attention"}}.get(k, k)
               for k in config["layers_block_type"]]

_DT = "bf16"

H = config["hidden_size"]
V = config["vocab_size"]
EPS = config["layer_norm_epsilon"]

# Mamba-2. `intermediate` is heads x head_dim, not `expand * hidden`.
MH = config["mamba_num_heads"]
MD = config["mamba_head_dim"]
SS = config["ssm_state_size"]
NG = config["n_groups"]
MI = MH * MD
#: x, B and C leave the projection in one tensor and share one depthwise conv.
CONV = MI + 2 * NG * SS
KER = config["conv_kernel"]
#: The conv window carried between steps: the kernel spans KER positions ending
#: at this token, so KER - 1 of them are prior.
WIN = KER - 1
PROJ = MI + CONV + MH
GRP = MI // NG
HPG = MH // NG
DTMIN = config["time_step_min"]
FMAX = 3.4028234663852886e38
NEGINF = -3.0e38

# Attention. This model applies no rotary embedding -- the published
# `NemotronHAttention.forward` never calls `apply_rotary_pos_emb`, so
# `rope_theta` in the config reaches nothing.
HQ = config["num_attention_heads"]
HKV = config["num_key_value_heads"]
DH = config["head_dim"]
QP = HQ * DH
KVP = HKV * DH
GQA = HQ // HKV
QSCALE = DH ** -0.5

# MoE. The experts are un-gated: up then relu-squared then down.
E = config["n_routed_experts"]
K = config["num_experts_per_tok"]
I = config["moe_intermediate_size"]
IS = config["moe_shared_expert_intermediate_size"]
RSCALE = config["routed_scaling_factor"]

CAP = config["max_position_embeddings"]

#: Placement extents; gen_model.py records why each is what it is.
U = {U}
UE = {UE}
ABLK = {ABLK}
WRK = {WRK}
#: Workers under the head axis, for the short-context placement. 33 x 4 is the
#: SM count, which is what the kernel's grid is.
WRKH = {WRKH}
#: Where the two attention placements change over, read off the per-unit work
#: of each (`attention.py`), not tuned: at ctx_full = 2048 walking the whole
#: context with one query head passes carrying all 32 heads over a stripe.
CROSSOVER = {CROSSOVER}

#: An attention layer is handed its context as two views of one buffer: `CF`
#: positions that fill whole ABLK-sized blocks, and `CT` more that do not. The
#: split is what lets the scan's block size stay static at every context length,
#: and it is free -- both are views, and this token's row is written into the
#: remainder by `cache_update`. Together they are the whole context.
CF = DimVar("ctx_full", 0, CAP + 1)
CT = DimVar("ctx_tail", 1, ABLK + 1)

_H200 = CudaTarget("nvidia.h200_sxm")
'''

TAIL = '''
    # ---------------- orchestration, shared by both twins ----------------

    def init_caches(self, device=None, capacity=None):
        """Every layer's state, and the two step-indexed scalars a step reads.

        The K/V caches are allocated once at their full capacity and never
        reallocated: a step is handed two views of the prefix that exists so
        far -- whole blocks, then a remainder -- and `tf.cache_update` writes
        this token's row into the remainder. `pos` is a table so that `cur_pos`
        is a view of a device tensor rather than a host-to-device copy on every
        step.
        """
        import torch  # noqa: PLC0415

        device = torch.accelerator.current_accelerator() if device is None else device
        # Rounded up to a whole block: a step is handed the remainder as a
        # view starting at a multiple of ABLK, and an implementation that
        # moves that view a block at a time reads to the end of the block.
        cap = CAP if capacity is None else int(capacity)
        cap = ((cap + ABLK - 1) // ABLK) * ABLK
        state = {"cap": cap, "step": 0, "layers": [],
                 # On the host, not on the accelerator. `cur_pos` is a view of
                 # this table, and an implementation that has to know which row
                 # of the cache to write would otherwise read it back off the
                 # device -- one synchronisation at the top of every step, with
                 # the card idle while the host prepares the next one.
                 "pos": torch.arange(cap, dtype=torch.int32),
                 "one": torch.ones(1, device=device, dtype=torch.int32)}
        for kind in LAYER_KINDS:
            if kind == "linear_attention":
                state["layers"].append((
                    torch.zeros(1, CONV, WIN, dtype=torch.bfloat16, device=device),
                    torch.zeros(1, MH, MD, SS, dtype=torch.float32, device=device),
                ))
            elif kind == "full_attention":
                state["layers"].append((
                    torch.zeros(1, cap, HKV, DH, dtype=torch.bfloat16, device=device),
                    torch.zeros(1, cap, HKV, DH, dtype=torch.bfloat16, device=device),
                ))
            else:
                state["layers"].append(())
        return state

    def prepare_inputs_for_generation(self, input_ids, step, caches, device=None):
        """The token and every layer's state, in the order `decode_step` declares.

        An attention layer is handed the prefix of its cache that exists once
        this token has a slot: a view, never a copy. Which slot that is comes
        back to `append_cache` through the caches themselves.
        """
        import torch  # noqa: PLC0415

        device = torch.accelerator.current_accelerator() if device is None else device
        caches["step"] = step
        total = step + 1
        # Whole ABLK-sized blocks, then whatever is left -- at least this token's
        # own row, so the remainder is never empty and `cache_update` always has
        # somewhere to write.
        full = ((total - 1) // ABLK) * ABLK
        tail = total - full
        token_ids = input_ids[step].reshape(1).to(device=device, dtype=torch.int64)
        acts = [token_ids, caches["pos"][tail - 1:tail], caches["one"]]
        for kind, entry in zip(LAYER_KINDS, caches["layers"]):
            if kind == "linear_attention":
                acts += [entry[0], entry[1]]
            elif kind == "full_attention":
                acts += [entry[0][:, :full], entry[1][:, :full],
                         entry[0][:, full:total], entry[1][:, full:total]]
        return tuple(acts)

    def forward(self, *acts):
        """One decode step: the logits, and every layer's fresh state beside them."""
        produced = self.decode_step(*acts)
        return produced[0], produced[1:]

    def append_cache(self, caches, fresh):
        """Every layer's state advanced by the step it just took.

        A Mamba layer hands back its whole new window and matrix, so the step
        already did the advancing and there is nothing to join on. An attention
        layer hands back this token's K/V row; when the step realised
        `cache_update` on the cache's own buffer that row *is* the buffer's row,
        which is what the pointer comparison asks before copying.
        """
        step = caches["step"]
        out, at = [], 0
        for kind, entry in zip(LAYER_KINDS, caches["layers"]):
            if kind == "linear_attention":
                out.append((fresh[at], fresh[at + 1]))
                at += 2
            elif kind == "full_attention":
                for buf, row in zip(entry, (fresh[at], fresh[at + 1])):
                    slot = buf[:, step : step + 1]
                    if slot.data_ptr() != row.data_ptr():
                        slot.copy_(row.reshape(slot.shape))
                out.append(entry)
                at += 2
            else:
                out.append(())
        caches["layers"] = out
        return caches
'''


def scan_funcs(var: int) -> str:
    """The attention scan as a dispatch prototype and its two placements.

    A `Call` from one HIR `Function` to another is a device call inside the same
    kernel invocation, so pulling the scan out of the layer body does not make a
    second launch of it. What it buys is that the two placements can be *stated*:
    below the crossover the query head is the axis with units in it, above it the
    context is, and `DimVarRangePat` says where the change is rather than a
    comment saying it. The kernel keys its two bodies on the same number.

    The remainder block is merged once, into the already-merged state, and not
    inside the worker-sharded scan -- a worker-sharded merge of it puts the same
    positions into every worker's partial and the combine then adds them up as
    many times as there are workers.
    """
    if var < 5:
        return ""
    sig = ("        qg: Tensor[(1, HKV, GQA, DH), _DT],\n"
           "        k_cache: Tensor[(1, CF, HKV, DH), _DT],\n"
           "        v_cache: Tensor[(1, CF, HKV, DH), _DT],\n"
           "        k_tail: Tensor[(1, CT, HKV, DH), _DT],\n"
           "        v_tail: Tensor[(1, CT, HKV, DH), _DT],\n"
           "    ) -> Tensor[(1, QP), _DT]:")
    out = []
    for name, layout, names, qax, wax, step, boff, W in (
        ("attend_by_head", "(HKV, GQA, WRKH)", '("g", "q", "w")', "GQA @ kv.q",
         "WRKH @ kv.w, ", "ABLK * WRKH", "kv.w * ABLK", "WRKH"),
        ("attend_by_context", "(HKV, WRK)", '("g", "w")', "GQA",
         "WRK @ kv.w, ", "ABLK * WRK", "kv.w * ABLK", "WRK"),
    ):
        out += [
            "    @func",
            f"    def {name}(",
            sig,
            f"        with Mesh((\"cta\",), layout={layout}, names={names}) as kv:",
            f"            qs = tf.reshard(qg, (1, HKV @ kv.g, {qax}, DH), \"smem\")",
            f"            m0 = tf.zeros(Tensor[({wax}HKV @ kv.g, {qax}, 1), \"f32\", \"smem\"])",
            f"            a0 = tf.zeros(Tensor[({wax}HKV @ kv.g, {qax}, DH), \"f32\", \"smem\"])",
            "            m = tf.full_like(m0, value=NEGINF)",
            "            l = tf.full_like(m0, value=0.0)",
            "            acc = tf.full_like(a0, value=0.0)",
            f"            for t in tile(CF, {step}):",
            f"                b0 = t + {boff}",
            "                kb = tf.reshard(tf.transpose(k_cache[:, b0:b0 + ABLK, :, :],"
            " perm=(0, 2, 1, 3)), (1, HKV @ kv.g, ABLK, DH), \"smem\")",
            "                vb = tf.reshard(tf.transpose(v_cache[:, b0:b0 + ABLK, :, :],"
            " perm=(0, 2, 1, 3)), (1, HKV @ kv.g, ABLK, DH), \"smem\")",
            "                rw = tf.cast(tf.matmul(qs, kb, b_layout=\"NK\"), dtype=\"f32\")",
            "                sb = rw * tf.full_like(rw, value=QSCALE)",
            "                bm = tf.reduce(sb, axes=(-1,), keepdim=True, kind=\"max\")",
            "                nm = tf.max(m, bm)",
            "                cr = tf.exp(m - nm)",
            "                pw = tf.exp(sb - nm)",
            "                l = l * cr + tf.reduce(pw, axes=(-1,), keepdim=True, kind=\"sum\")",
            "                acc = acc * cr + tf.cast(tf.matmul(tf.cast(pw, dtype=_DT), vb),"
            " dtype=\"f32\")",
            "                m = nm",
            f"            am = tf.reshard(m, ({W}, HKV @ kv.g, {qax}, 1), \"smem\")",
            f"            al = tf.reshard(l, ({W}, HKV @ kv.g, {qax}, 1), \"smem\")",
            f"            aa = tf.reshard(acc, ({W}, HKV @ kv.g, {qax}, DH), \"smem\")",
            "            gm = tf.reduce(am, axes=(0,), keepdim=True, kind=\"max\")",
            "            cw = tf.exp(am - gm)",
            "            gl = tf.reduce(cw * al, axes=(0,), keepdim=True, kind=\"sum\")",
            "            ga = tf.reduce(cw * aa, axes=(0,), keepdim=True, kind=\"sum\")",
            "            kbt = tf.reshard(tf.transpose(k_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),"
            " (1, HKV @ kv.g, CT, DH), \"smem\")",
            "            vbt = tf.reshard(tf.transpose(v_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),"
            " (1, HKV @ kv.g, CT, DH), \"smem\")",
            "            rwt = tf.cast(tf.matmul(qs, kbt, b_layout=\"NK\"), dtype=\"f32\")",
            "            sbt = rwt * tf.full_like(rwt, value=QSCALE)",
            "            bmt = tf.reduce(sbt, axes=(-1,), keepdim=True, kind=\"max\")",
            "            nmt = tf.max(gm, bmt)",
            "            crt = tf.exp(gm - nmt)",
            "            pwt = tf.exp(sbt - nmt)",
            "            lt = gl * crt + tf.reduce(pwt, axes=(-1,), keepdim=True, kind=\"sum\")",
            "            act = ga * crt + tf.cast(tf.matmul(tf.cast(pwt, dtype=_DT), vbt),"
            " dtype=\"f32\")",
            "            ct = tf.cast(act / lt, dtype=_DT)",
            "            return tf.reshape(tf.reshard(ct, (1, HKV, GQA, DH), \"gmem\"),"
            " new_shape=(1, QP))",
            "",
        ]
    out += [
        "    @func",
        "    def attend(",
        sig,
        "        pass",
        "",
        "    @attend.specialize(DimVarRangePat(\"ctx_full\", 0, CROSSOVER))",
        "    def attend_short(",
        sig,
        "        return attend_by_head(qg, k_cache, v_cache, k_tail, v_tail)",
        "",
        "    @attend.specialize(DimVarRangePat(\"ctx_full\", CROSSOVER, CAP + 1))",
        "    def attend_long(",
        sig,
        "        return attend_by_context(qg, k_cache, v_cache, k_tail, v_tail)",
        "",
    ]
    return "\n".join(out) + "\n"

def build(var: int) -> str:
    out = [HEADER.format(variant=var, nlayers=len(KINDS),
                         nmamba=KINDS.count("linear_attention"),
                         nattn=KINDS.count("full_attention"),
                         nmoe=KINDS.count("moe"),
                         U=U, UE=UE, ABLK=ABLK, WRK=WRK,
                         WRKH=WRKH, CROSSOVER=CROSSOVER)]
    out.append("\n\n@module(entry=\"decode_step\", target=_H200,\n"
               "        topologies=(Topology(\"cta\", U), Topology(\"thread\", 256)))\n"
               "class Nemotron35Lightning30BA3B:\n"
               "    \"\"\"The published model, and its decode step as one program.\"\"\"\n\n"
               + scan_funcs(var)
               + "    @func\n"
               "    def decode_step(\n")
    sig = ["        token_ids: Tensor[(1,), \"i64\"],",
           "        cur_pos: Tensor[(1,), \"i32\"],",
           "        one: Tensor[(1,), \"i32\"],"]
    for i, kind in enumerate(KINDS):
        for name, shape, dt in sig_states(i, kind):
            sig.append(f"        {name}: Tensor[{shape}, {dt}],")
    sig.append("        table: ConstTensor[(V, H), _DT],")
    for i, kind in enumerate(KINDS):
        sig.append(f"        l{i}_gamma: ConstTensor[(H,), _DT],")
        for name, shape in sig_consts(i, kind):
            sig.append(f"        {name}: ConstTensor[{shape}, _DT],")
        if kind == "moe":
            sig.append(f"        l{i}_e_bias: ConstTensor[(E,), \"f32\"],")
    sig.append("        gamma_final: ConstTensor[(H,), _DT],")
    sig.append("        w_head: ConstTensor[(V, H), _DT],")
    sig.append("    ):")
    out.append("\n".join(sig) + "\n")

    ind = "        "
    body = [
        ind + "# The one mesh every stage of this step is placed on; `cta.u` is",
        ind + "# the axis an output row lands on. Everything is inside it: an op",
        ind + "# authored outside a cta Mesh has no execution domain to be",
        ind + "# scheduled against.",
        ind + "with Mesh((\"cta\",), layout=(U,), names=(\"u\",)) as cta:",
    ]
    ind += "    "
    body += [
        ind + "# The embedding is this token's own row, read as a window so the",
        ind + "# traffic charged is the row and not the whole table.",
        ind + "tid = token_ids[0]",
        ind + "h = tf.reshape(table[tid:tid + 1, :], new_shape=(1, 1, H))",
    ]
    if var >= 4:
        body += [
            ind + "# Long context divides over a second axis: one worker per KV",
            ind + "# block stripe, combined by log-sum-exp at the end of the scan.",
            ind + "with Mesh((\"cta\",), layout=(HKV, WRK), names=(\"g\", \"w\")) as kv:",
        ]
        ind += "    "
    for i, kind in enumerate(KINDS):
        body.append("")
        body.append(ind + f"# ---- layer {i}: {kind}")
        for line in prenorm(i) + BODY[kind](i, var):
            body.append(ind + line)
        body.append(ind + f"h = h + l{i}_mix")
    body += [
        "",
        ind + "# ---- closing norm and head",
        ind + "ff = tf.cast(h, dtype=\"f32\")",
        ind + "fms = tf.reduce(tf.square(ff), axes=(-1,), keepdim=True, kind=\"mean\")",
        ind + "fn = tf.cast(ff * tf.rsqrt(fms + tf.full_like(fms, value=EPS)), dtype=_DT)",
        ind + "fh = tf.reshape(fn * tf.reshape(gamma_final, new_shape=(1, 1, H)),"
        " new_shape=(1, H))",
    ]
    body += [ind + line for line in proj(var, "logits0", "fh", "w_head", "V")]
    body += [ind + "logits = tf.cast(logits0, dtype=\"f32\")"]
    rets = ["logits"]
    for i, kind in enumerate(KINDS):
        rets += fresh(i, kind)
    body.append(ind + "return " + ", ".join(rets))
    out.append("\n".join(body) + "\n")
    out.append(TAIL)
    return "".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = build(a.variant)
    path = Path(a.out) if a.out else Path(f"model_v{a.variant}.py")
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({len(text.splitlines())} lines, variant {a.variant})")
