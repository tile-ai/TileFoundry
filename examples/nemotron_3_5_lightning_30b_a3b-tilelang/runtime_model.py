"""The runtime twin of `model.py`: the same decode step, as kernels.

`@runtime_module` checks at decoration time that this class's function and child
names are exactly the authored Module's, so the twin covers the whole model at
once -- one `decode_step`, because the authored side is one `decode_step`.

Two bodies live here and `NEMO_IMPL` (or `set_impl`) chooses between them:

``mega``   one launch. A persistent cooperative grid runs the whole step:
           52 layers, the closing norm and the head, with `T.sync_grid()` at
           exactly the points where `model.py` reshards a value back out of its
           mesh split. This is the implementation the numbers are about.
``ops``    the op-by-op path: one torch call per operation, so a step is
           hundreds of launches. It exists to be the other number criterion 3
           asks for, and to be a second opinion on the mega kernel's arithmetic.

Both read the same weights. `load` is overridden so the twin keeps them packed
by kind -- one tensor per weight class, layers stacked on the leading axis --
because a kernel that takes 401 separate pointers cannot be launched, while one
that takes 18 can address any of them by (layer, row).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import model
from model import Nemotron35Lightning30BA3B as SEM
from tilefoundry.runtime import runtime_func, runtime_module

H, V, EPS = model.H, model.V, model.EPS
MH, MD, SS, NG, MI, CONV, KER, WIN, PROJ = (model.MH, model.MD, model.SS, model.NG,
                                            model.MI, model.CONV, model.KER, model.WIN,
                                            model.PROJ)
GRP, HPG, DTMIN = model.GRP, model.HPG, model.DTMIN
HQ, HKV, DH, QP, KVP, GQA, QSCALE = (model.HQ, model.HKV, model.DH, model.QP,
                                     model.KVP, model.GQA, model.QSCALE)
E, K, I, IS, RSCALE = model.E, model.K, model.I, model.IS, model.RSCALE
KINDS = model.LAYER_KINDS
BF16 = torch.bfloat16

#: Which implementation `decode_step` runs. "mega" is the one launch.
IMPL = os.environ.get("NEMO_IMPL", "mega")


def set_impl(name: str) -> None:
    """Choose between the mega kernel and the op-by-op path."""
    global IMPL
    if name not in ("mega", "ops"):
        raise ValueError(f"impl must be 'mega' or 'ops', got {name!r}")
    IMPL = name


# ---------------------------------------------------------------------------
# Where each declared parameter sits in the positional call.
# ---------------------------------------------------------------------------

_PARAMS = tuple(p.name for p in SEM.entry_function().params)
_INDEX = {name: at for at, name in enumerate(_PARAMS)}

#: Layer index -> the first of its state parameters, for the layers that hold any.
_STATE_AT = {}
for _i, _k in enumerate(KINDS):
    if _k == "linear_attention":
        _STATE_AT[_i] = _INDEX[f"l{_i}_conv_state"]
    elif _k == "full_attention":
        _STATE_AT[_i] = _INDEX[f"l{_i}_k_cache"]


def _rms(x, gamma):
    """The published `NemotronHRMSNorm`, cast for cast.

    The normalised value lands in bf16 *before* the weight multiplies it, which
    is one rounding more than doing the whole thing in f32 would be. It is what
    the checkpoint's own implementation does, so it is what the reference does.
    """
    xf = x.float()
    nz = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(BF16)
    return nz * gamma.reshape(1, 1, -1)


def _attend(qg, blocks):
    """Online-softmax merge over the K/V blocks of one attention layer.

    The blocks are the two views one step is handed -- the whole ABLK-sized ones
    and the remainder -- and the merge is what makes reading them separately the
    same value as one softmax over their concatenation.
    """
    m = torch.full((1, HKV, GQA, 1), -float("inf"), device=qg.device, dtype=torch.float32)
    l = torch.zeros_like(m)
    acc = torch.zeros(1, HKV, GQA, DH, device=qg.device, dtype=torch.float32)
    for kb, vb in blocks:
        if kb.shape[1] == 0:
            continue
        kh = kb.transpose(1, 2)                       # (1, HKV, len, DH)
        vh = vb.transpose(1, 2)
        raw = torch.matmul(qg, kh.transpose(-1, -2)).float()
        sb = raw * QSCALE
        bm = sb.amax(dim=-1, keepdim=True)
        nm = torch.maximum(m, bm)
        cr = torch.exp(m - nm)
        pw = torch.exp(sb - nm)
        l = l * cr + pw.sum(dim=-1, keepdim=True)
        acc = acc * cr + torch.matmul(pw.to(BF16), vh).float()
        m = nm
    return (acc / l).to(BF16)


def _ops_step(w, acts, stop=None, attn_at=None):
    """One decode step, one torch call per operation.

    *stop* returns the running hidden row after that many layers instead of the
    logits, which is what a bisect against the mega kernel's scratch compares.
    *attn_at* returns that layer's five `attend` arguments instead, so a real
    step is where the standalone attention gets its inputs rather than five
    independent draws.
    """
    token_ids, cur_pos = acts["token_ids"], acts["cur_pos"]
    h = w["table"].index_select(0, token_ids).reshape(1, 1, H)
    fresh = []
    for i, kind in enumerate(KINDS):
        if stop is not None and i >= stop:
            return h.reshape(-1).float()
        h2 = _rms(h, w[f"l{i}_gamma"]).reshape(1, H)
        if kind == "linear_attention":
            conv_state, ssm_state = acts[f"l{i}_conv_state"], acts[f"l{i}_ssm_state"]
            w_in = w[f"l{i}_w_in"]
            gate = h2 @ w_in[0:MI].t()
            col0 = h2 @ w_in[MI:MI + CONV].t()
            dt = h2 @ w_in[MI + CONV:PROJ].t()
            win = torch.cat([conv_state, col0.reshape(1, CONV, 1)], dim=2)
            conv_out = win[:, :, 1:KER].contiguous()
            cs = (win * w[f"l{i}_conv_w"].reshape(1, CONV, KER)).sum(-1)
            xbc = F.silu(cs + w[f"l{i}_conv_b"].reshape(1, CONV))
            x = xbc[:, 0:MI].reshape(1, MH, MD)
            bg = xbc[:, MI:MI + NG * SS].reshape(1, NG, SS)
            cg = xbc[:, MI + NG * SS:CONV].reshape(1, NG, SS)
            b = bg.repeat_interleave(HPG, dim=1).reshape(1, MH, 1, SS)
            c = cg.repeat_interleave(HPG, dim=1).reshape(1, MH, SS, 1)
            dta = F.softplus(dt.reshape(1, MH, 1)
                             + w[f"l{i}_dt_bias"].reshape(1, MH, 1)).clamp(min=DTMIN)
            dte = dta.reshape(1, MH, 1, 1)
            an = (-torch.exp(w[f"l{i}_a_log"].float())).reshape(1, MH, 1, 1)
            da = torch.exp(dte.float() * an)
            dbx = ((dte * b) * x.reshape(1, MH, MD, 1)).float()
            ssm_out = ssm_state * da + dbx
            y = torch.matmul(ssm_out.to(BF16), c.to(BF16)).reshape(1, MH, MD)
            yd = y + x * w[f"l{i}_d_skip"].reshape(1, MH, 1)
            yf = yd.reshape(1, MI).float() * F.silu(gate.float())
            yg = yf.reshape(1, NG, GRP)
            yn = (yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + EPS)).reshape(1, MI)
            scan = w[f"l{i}_gamma_gdn"].reshape(1, MI) * yn.to(BF16)
            mix = (scan @ w[f"l{i}_w_out"].t()).reshape(1, 1, H)
            fresh += [conv_out, ssm_out]
        elif kind == "full_attention":
            k_cache, v_cache = acts[f"l{i}_k_cache"], acts[f"l{i}_v_cache"]
            k_tail, v_tail = acts[f"l{i}_k_tail"], acts[f"l{i}_v_tail"]
            q0 = h2 @ w[f"l{i}_w_q"].t()
            k_new = (h2 @ w[f"l{i}_w_k"].t()).reshape(1, 1, HKV, DH)
            v_new = (h2 @ w[f"l{i}_w_v"].t()).reshape(1, 1, HKV, DH)
            # `cache_update`, realised on the cache's own buffer.
            k_tail[:, cur_pos] = k_new[:, 0]
            v_tail[:, cur_pos] = v_new[:, 0]
            qg = q0.reshape(1, HKV, GQA, DH)
            if attn_at == i:
                return qg, k_cache, v_cache, k_tail, v_tail
            ct = _attend(qg, ((k_cache, v_cache), (k_tail, v_tail)))
            ctx = ct.reshape(1, QP)
            mix = (ctx @ w[f"l{i}_w_o"].t()).reshape(1, 1, H)
            fresh += [k_tail[:, cur_pos:cur_pos + 1].reshape(1, 1, HKV, DH),
                      v_tail[:, cur_pos:cur_pos + 1].reshape(1, 1, HKV, DH)]
        else:
            lg = h2.float() @ w[f"l{i}_w_router"].float().t()
            sig = torch.sigmoid(lg)
            ch = sig + w[f"l{i}_e_bias"].reshape(1, E)
            _tv, ti = torch.topk(ch, K, dim=-1, sorted=False)
            flat = ti.reshape(K)
            pick = sig.reshape(E)[flat].reshape(1, K)
            gw = (pick / (pick.sum(-1, keepdim=True) + 1e-20)) * RSCALE
            w_up, w_down = w[f"l{i}_w_up"], w[f"l{i}_w_down"]
            chosen = flat.tolist()
            total = None
            for j, e in enumerate(chosen):
                mid = torch.square(torch.relu(h2 @ w_up[e].t()))
                r = (mid @ w_down[e].t()).reshape(H).float() * gw[0, j]
                total = r if total is None else total + r
            smid = torch.square(torch.relu(h2 @ w[f"l{i}_w_sh_up"].t()))
            sh = (smid @ w[f"l{i}_w_sh_down"].t()).reshape(H)
            mix = (total.to(BF16) + sh).reshape(1, 1, H)
        h = h + mix
    if stop is not None:
        return h.reshape(-1).float()
    fh = _rms(h, w["gamma_final"]).reshape(1, H)
    logits = (fh @ w["w_head"].t()).float()
    return (logits, *fresh)


@runtime_module(SEM)
class Nemotron35Lightning30BA3BRuntime:
    """The published model's decode step, as one launch."""

    @runtime_func
    def attend_by_head(self, qg, k_cache, v_cache, k_tail, v_tail):
        """The short-context placement, on its own.

        The step does not call this -- it inlines the same body, which is what
        keeps one launch -- but the semantics names it, so the twin answers with
        the same code generated around a pair of parameters instead of around
        the step's own scratch.
        """
        import mega_kernel  # noqa: PLC0415 -- compiled on first use

        return mega_kernel.run_attn("head", qg, k_cache, v_cache, k_tail, v_tail)

    @runtime_func
    def attend_by_context(self, qg, k_cache, v_cache, k_tail, v_tail):
        """The long-context placement, on its own. See `attend_by_head`."""
        import mega_kernel  # noqa: PLC0415

        return mega_kernel.run_attn("context", qg, k_cache, v_cache, k_tail, v_tail)

    @runtime_func
    def attend(self, qg, k_cache, v_cache, k_tail, v_tail):
        """Whichever placement the whole-block context length asks for.

        The same number the semantics dispatches on and the same number the
        step's own branch reads: `ctx_full`, against the crossover.
        """
        import mega_kernel  # noqa: PLC0415

        return mega_kernel.run_attn("dispatch", qg, k_cache, v_cache, k_tail, v_tail)

    @runtime_func
    def decode_step(self, *args):
        acts = {name: args[at] for name, at in _INDEX.items()
                if not SEM.entry_function().params[at].is_const}
        # The declared parameter, not a shape it happens to agree with. It costs
        # nothing to read because the position table it views lives on the host.
        acts["cur_pos"] = int(args[_INDEX["cur_pos"]][0])
        acts["token_ids"] = args[_INDEX["token_ids"]]
        if IMPL == "ops":
            return _ops_step(self._bound, acts)
        return self._mega(args, acts)

    def _mega(self, args, acts):
        import mega_kernel  # noqa: PLC0415 -- compiled on first use

        return mega_kernel.run_step(self._run, args, acts["cur_pos"], _INDEX)

    def load(self, resource):
        """Read every declared weight straight into its slot in the packs.

        A kernel cannot take 401 pointers, so the twin holds one flat tensor per
        weight class and this reads each weight into its place and drops the
        original. `_bound` then keeps the *view* standing for that weight, so the
        op-by-op path and `check`'s weight report see exactly what was declared
        while the kernel sees eighteen buffers.
        """
        import mega_kernel  # noqa: PLC0415

        device = getattr(resource, "device", "cuda")
        self._run = mega_kernel.Runner(device)
        for name in self._ir.weights:
            value = resource.load(name)
            self._bound[name] = mega_kernel.pack_into(self._run.packed, name, value)
            del value
