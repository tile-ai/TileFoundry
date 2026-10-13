"""Run the published `transformers` model and keep what `check` compares against.

    CUDA_VISIBLE_DEVICES=1 python tools/hf_reference.py --lens 1024 --layers 0,3 --ckpt-out refdata/ckpt
    CUDA_VISIBLE_DEVICES=1 python tools/hf_reference.py --lens 1024,4096,8192 --logits-only

For each length it writes `refdata/n{N}/`:

  logits.pt            the last row's logits (bf16) and the greedy first token
  embed/               inputs and expected output of `Qwen3_8_27B.embed`
  final/               ... of `Qwen3_8_27B.final`
  v1/layer{L}/<func>/  ... of every function of layer L, one directory each:
                       `in{i}.pt` per activation parameter in declaration
                       order, `expected.pt` the output (a list for a tuple)

Each function's inputs are the published model's own activations at that
point, captured by hooks; each expected output is the published module's
own code run on them, and is checked against the hooked downstream value
where one exists, so the reference cannot drift from the model it names.

`--ckpt-out` also writes a weights directory keyed by this model's module
paths (`layer0.mixer.w_qkv`, ...), the form `check --weights ckpt:DIR` reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt import CKPT, DEFAULT_TEXT, prompt_ids  # noqa: E402


def save(dirpath: Path, inputs, expected):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(inputs):
        torch.save(t.contiguous().cpu(), dirpath / f"in{i}.pt")
    if isinstance(expected, (list, tuple)):
        torch.save([t.contiguous().cpu() for t in expected], dirpath / "expected.pt")
    else:
        torch.save(expected.contiguous().cpu(), dirpath / "expected.pt")


def same(a, b, what):
    if not torch.equal(a, b):
        d = (a.float() - b.float()).abs().max().item()
        raise SystemExit(f"reference self-check failed: {what} differs from the hooked value by {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="1024")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--ckpt-out", default=None)
    ap.add_argument("--logits-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    import model as tfm

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(CKPT)
    net = hf.Qwen3_5ForConditionalGeneration.from_pretrained(CKPT, dtype=torch.bfloat16).eval().to(dev)
    text = net.model.language_model
    cfg = text.config
    print(f"attn_implementation={cfg._attn_implementation}", flush=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else []
    out_root = Path(args.out)

    if args.ckpt_out:
        from safetensors.torch import save_file

        alias = tfm.hf_alias()
        sd = dict(net.named_parameters())
        tensors = {}
        root = tfm.Qwen3_8_27B
        for name in ("table", "gamma_final", "w_head"):
            tensors[name] = sd[alias[name]].detach().contiguous().cpu()
        for li in layers:
            lmod = root.modules[li]
            prefix = f"model.language_model.layers.{li}"
            for sub, hfsub in (("", ""), ("mixer.", alias[f"{prefix}.mixer"] + "."), ("mlp.", "mlp.")):
                node = lmod if not sub else getattr(lmod, sub[:-1])
                for wname in node.weights:
                    raw = alias[wname]
                    if wname in ("gamma_in", "gamma_post"):
                        key = f"{prefix}.{raw}"
                    else:
                        key = f"{prefix}.{hfsub}{raw}"
                    tensors[f"layer{li}.{sub}{wname}"] = sd[key].detach().contiguous().cpu()
        Path(args.ckpt_out).mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(Path(args.ckpt_out) / "model.safetensors"))
        print(f"wrote {len(tensors)} weights to {args.ckpt_out}", flush=True)

    for n in [int(x) for x in args.lens.split(",")]:
        ids = prompt_ids(tok, DEFAULT_TEXT, n).to(dev)
        d = out_root / f"n{n}"
        d.mkdir(parents=True, exist_ok=True)
        caught: dict[str, object] = {}
        hooks = []
        if not args.logits_only:
            def keep(name, which="out"):
                def fn(mod, inp, out):
                    if which == "in":
                        caught[name] = inp[0].detach()
                    else:
                        caught[name] = (out[0] if isinstance(out, tuple) else out).detach()
                return fn

            def keep_pre(name):
                def fn(mod, inp, kwargs):
                    caught[name] = (inp[0] if inp else kwargs["hidden_states"]).detach()
                return fn

            for li in layers:
                L = text.layers[li]
                hooks.append(L.register_forward_pre_hook(keep_pre(f"l{li}.in"), with_kwargs=True))
                hooks.append(L.input_layernorm.register_forward_hook(keep(f"l{li}.hn")))
                mixer = L.linear_attn if L.block_type == "linear_attention" else L.self_attn
                hooks.append(mixer.register_forward_hook(keep(f"l{li}.mixer")))
                hooks.append(L.post_attention_layernorm.register_forward_hook(keep(f"l{li}.attended", "in")))
                hooks.append(L.post_attention_layernorm.register_forward_hook(keep(f"l{li}.hn2")))
                hooks.append(L.mlp.register_forward_hook(keep(f"l{li}.mlp")))
                hooks.append(L.register_forward_hook(keep(f"l{li}.out")))
            hooks.append(text.embed_tokens.register_forward_hook(keep("embed")))
            hooks.append(text.norm.register_forward_hook(keep("final.in", "in")))

        with torch.no_grad():
            res = net(input_ids=ids[None], logits_to_keep=1, use_cache=False)
        logits = res.logits[0, -1].detach()
        first = int(torch.argmax(logits))
        torch.save({"logits": logits.cpu(), "first_token": first, "ids": ids.cpu()}, d / "logits.pt")
        print(f"n={n} first_token={first} {tok.decode([first])!r}", flush=True)
        for h in hooks:
            h.remove()
        if args.logits_only:
            continue

        last = torch.tensor([n - 1], dtype=torch.int64, device=dev)
        save(d / "embed", [ids], caught["embed"])
        save(d / "final", [caught["final.in"], last], logits[None])

        cos_hf, sin_hf = text.rotary_emb(caught["embed"], torch.arange(n, device=dev)[None].expand(3, 1, -1).contiguous())
        cos, sin = tfm.rope_tables(n, dev)
        same(cos, cos_hf[0], "rope cos table")
        same(sin, sin_hf[0], "rope sin table")

        with torch.no_grad():
            for li in layers:
                L = text.layers[li]
                ld = d / "v1" / f"layer{li}"
                hidden, hn = caught[f"l{li}.in"], caught[f"l{li}.hn"]
                attended, hn2 = caught[f"l{li}.attended"], caught[f"l{li}.hn2"]
                save(ld / "input_norm", [hidden], hn)
                save(ld / "post_norm", [attended], hn2)
                act = L.mlp.act_fn(L.mlp.gate_proj(hn2)) * L.mlp.up_proj(hn2)
                save(ld / "gate_up", [hn2], act)
                down = attended + L.mlp.down_proj(act)
                same(down, caught[f"l{li}.out"], "mlp + residual rerun")
                save(ld / "down", [act, attended], down)
                if L.block_type == "linear_attention":
                    m = L.linear_attn
                    qkv, z, b, a = m.in_proj_qkv(hn), m.in_proj_z(hn), m.in_proj_b(hn), m.in_proj_a(hn)
                    save(ld / "in_proj", [hn], [qkv, z, b, a])
                    mixed = F.silu(m.conv1d(qkv.transpose(1, 2))[:, :, :n]).transpose(1, 2)
                    q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
                    q = hf.l2norm(q.reshape(1, n, -1, m.head_k_dim), dim=-1, eps=1e-6)
                    k = hf.l2norm(k.reshape(1, n, -1, m.head_k_dim), dim=-1, eps=1e-6)
                    v = v.reshape(1, n, -1, m.head_v_dim)
                    beta = b.sigmoid()
                    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
                    save(ld / "prep", [qkv, b, a], [q, k, v, g, beta])
                    rep = m.num_v_heads // m.num_k_heads
                    core, _ = m.chunk_gated_delta_rule(
                        q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2), v, g=g, beta=beta,
                        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False)
                    save(ld / "delta_rule", [q, k, v, g, beta], core)
                    gated = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(1, n, -1)
                    save(ld / "gated_norm", [core, z], gated)
                    out = hidden + m.out_proj(gated)
                    same(out, attended, "gated delta net + residual rerun")
                    save(ld / "out_proj", [gated, hidden], out)
                else:
                    m = L.self_attn
                    shape = (1, n, -1, m.head_dim)
                    qp, gate = torch.chunk(m.q_proj(hn).view(1, n, -1, m.head_dim * 2), 2, dim=-1)
                    gate = gate.contiguous()  # (1, n, heads, head_dim): model.py keeps it per head
                    qs = m.q_norm(qp.view(shape)).transpose(1, 2)
                    ks = m.k_norm(m.k_proj(hn).view(shape)).transpose(1, 2)
                    vs = m.v_proj(hn).view(shape).transpose(1, 2)
                    qs, ks = hf.apply_rotary_pos_emb(qs, ks, cos_hf, sin_hf)
                    q4, k4, v4 = (t.transpose(1, 2).contiguous() for t in (qs, ks, vs))
                    save(ld / "qkv", [hn, cos, sin], [q4, gate, k4, v4])
                    iface = hf.ALL_ATTENTION_FUNCTIONS.get_interface(cfg._attn_implementation, hf.eager_attention_forward)
                    mask = hf.create_causal_mask(config=cfg, inputs_embeds=hn, attention_mask=None,
                                                 past_key_values=None, position_ids=torch.arange(n, device=dev)[None])
                    ao, _ = iface(m, qs, ks, vs, mask, dropout=0.0, scaling=m.scaling)
                    gated = ao.reshape(1, n, -1).contiguous() * torch.sigmoid(gate.reshape(1, n, -1))
                    save(ld / "attention", [q4, k4, v4, gate], gated)
                    out = hidden + m.o_proj(gated)
                    same(out, attended, "attention + residual rerun")
                    save(ld / "o_proj", [gated, hidden], out)
        print(f"n={n}: wrote {d}", flush=True)


if __name__ == "__main__":
    main()
