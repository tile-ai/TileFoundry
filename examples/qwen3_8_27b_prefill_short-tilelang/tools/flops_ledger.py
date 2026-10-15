#!/usr/bin/env python3
"""FLOP and byte ledger for one Qwen3.8-27B text prefill, read from the safetensors headers.

Nothing here trusts the model card. Every weight shape comes from the checkpoint
headers; every per-token cost is written out as a formula next to its number.

Output: FLOPS.md (the ledger, human readable) and reports/flops.json.
"""
from __future__ import annotations

import glob
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402

CKPT = str(paths.need("ckpt"))
LENGTHS = (128, 256, 512, 1024)
PEAK_BF16 = 989.5e12  # tilefoundry target show nvidia.h200_sxm: throughput.bf16
PEAK_F32 = 67e12      # throughput.f32
HBM_BW = 4.8e12       # memory.hbm.bandwidth
BYTES = {"BF16": 2, "F32": 4}


def headers():
    shapes = {}
    for path in sorted(glob.glob(f"{CKPT}/*.safetensors")):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for key, val in json.loads(fh.read(n)).items():
                if key != "__metadata__":
                    shapes[key] = (val["dtype"], tuple(val["shape"]))
    return shapes


def main():
    shapes = headers()
    cfg = json.loads(Path(f"{CKPT}/config.json").read_text())["text_config"]
    text = {k: v for k, v in shapes.items() if k.startswith("model.language_model.")}
    head = shapes["lm_head.weight"]
    skipped = {k for k in shapes if k not in text and k != "lm_head.weight"}

    # Every 2-D ".weight" under a decoder layer that is not a norm or the conv is an
    # nn.Linear: y = x @ W^T, 2 * T * out * in flops, W read once.
    gemm = defaultdict(int)       # category -> parameter count
    other = defaultdict(int)      # norms, conv, A_log, dt_bias -> bytes
    for key, (dt, shp) in text.items():
        if ".layers." not in key:
            continue
        leaf = key.split(".layers.")[1].split(".", 1)[1]
        numel = math.prod(shp)
        if len(shp) == 2 and leaf.endswith(".weight") and "norm" not in leaf:
            gemm[leaf] += numel
        else:
            other[leaf] += numel * BYTES[dt]
    embed_shape = text["model.language_model.embed_tokens.weight"][1]
    hidden = embed_shape[1]
    kinds = cfg["layer_types"]
    n_gdn = kinds.count("linear_attention")
    n_attn = kinds.count("full_attention")
    hq, hkv, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    hv, hk = cfg["linear_num_value_heads"], cfg["linear_num_key_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    conv_dim = 2 * hk * dk + hv * dv
    kernel = cfg["linear_conv_kernel_dim"]
    chunk = 64

    gemm_params = sum(gemm.values())
    head_params = math.prod(head[1])
    weight_bytes = 2 * gemm_params + sum(other.values()) + 2 * head_params

    rows = []
    for T in LENGTHS:
        f = {}
        f["gemm (all decoder nn.Linear)"] = 2 * T * gemm_params
        f["lm_head (last token only)"] = 2 * 1 * head_params
        # causal attention: QK^T and PV each 2 * D flops per (query, key<=query) pair, per q head
        f["attention core (causal QK^T + PV)"] = n_attn * hq * 2 * 2 * dh * (T * (T + 1) // 2)
        # chunked gated delta rule, C=64, per chunk per v-head:
        #   A = kb k^T: 2C^2 dk ; tri solve: ~2/3 C^3 ; u = A vb: 2C^2 dv ; w = A kb: 2C^2 dk
        #   q k^T: 2C^2 dk ; v' = w S: 2 C dk dv ; q S: 2 C dk dv ; attn v_new: 2C^2 dv ; k^T v_new: 2 C dk dv
        nc = math.ceil(T / chunk)
        per_chunk = 2 * chunk**2 * (3 * dk + 2 * dv) + (2 * chunk**3) // 3 + 6 * chunk * dk * dv
        f["gdn core (chunked WY, C=64)"] = n_gdn * hv * nc * per_chunk
        f["gdn conv1d (depthwise, k=4)"] = n_gdn * 2 * T * conv_dim * kernel
        flops = sum(f.values())
        # Compulsory HBM bytes: every weight once, the T embedding rows, the caches written.
        b = {}
        b["decoder weights (bf16, once)"] = 2 * gemm_params + sum(other.values())
        b["lm_head weight (bf16, once)"] = 2 * head_params
        b["embedding rows gathered"] = T * hidden * 2
        b["final norm weight"] = hidden * 2
        b["kv cache written (16 layers, bf16)"] = n_attn * 2 * T * hkv * dh * 2
        b["gdn state written (48 layers, f32 + conv bf16)"] = n_gdn * (hv * dk * dv * 4 + conv_dim * (kernel - 1) * 2)
        nbytes = sum(b.values())
        t_c = flops / PEAK_BF16
        t_m = nbytes / HBM_BW
        # per-GEMM serial bound: each linear is its own kernel, max(compute, weight read)
        serial = 0.0
        for leaf, p in gemm.items():
            serial += max(2 * T * p / PEAK_BF16, 2 * p / HBM_BW)
        serial += max(2 * head_params / PEAK_BF16, 2 * head_params / HBM_BW)
        serial += (f["attention core (causal QK^T + PV)"] + f["gdn core (chunked WY, C=64)"]) / PEAK_BF16
        rows.append(dict(T=T, flops=f, bytes=b, total_flops=flops, total_bytes=nbytes,
                         compute_ms=t_c * 1e3, memory_ms=t_m * 1e3,
                         roofline_ms=max(t_c, t_m) * 1e3, serial_gemm_roofline_ms=serial * 1e3,
                         bound="compute" if t_c > t_m else "memory"))

    out = dict(gemm_params_by_leaf=dict(gemm), gemm_params=gemm_params, lm_head_params=head_params,
               other_bytes=dict(other), weight_bytes=weight_bytes, skipped_tensors=len(skipped),
               crossover_tokens=PEAK_BF16 / HBM_BW, rows=rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/flops.json").write_text(json.dumps(out, indent=1))

    md = ["# FLOP / byte ledger (from safetensors headers)", "",
          f"Checkpoint: the published `Qwen3.8-27B` (`$QWEN38_CKPT`). Generated by `tools/flops_ledger.py`.",
          f"Rates: bf16 {PEAK_BF16/1e12:.1f} TFLOP/s, HBM {HBM_BW/1e12:.1f} TB/s "
          "(`tilefoundry target show nvidia.h200_sxm`).", "",
          "## Weights counted", "",
          "| leaf (summed over the layers that have it) | params | GEMM flops per token |", "|---|---:|---:|"]
    for leaf, p in sorted(gemm.items()):
        md.append(f"| `{leaf}` | {p:,} | {2*p:,} |")
    md += [f"| **all decoder nn.Linear** | **{gemm_params:,}** | **{2*gemm_params:,}** |",
           f"| `lm_head.weight` {head[1]} (last token only) | {head_params:,} | {2*head_params:,} once |",
           f"| `embed_tokens.weight` {embed_shape} | {math.prod(embed_shape):,} | 0 (row gather) |", "",
           f"Not counted (not on the text prefill path): {len(skipped)} tensors "
           "(`model.visual.*` vision tower, `mtp.*` multi-token-prediction head).", "",
           "Non-GEMM decoder tensors (norms, conv1d, A_log, dt_bias): "
           + ", ".join(f"`{k}` {v:,} B" for k, v in sorted(other.items())), "",
           f"Arithmetic intensity of a bf16 GEMM with M=T rows is T flop/byte of weight; the ridge "
           f"point is {PEAK_BF16/HBM_BW:.0f} tokens: below it the linears are weight-read bound, above it "
           "they are tensor-core bound.", "",
           "## Per length", "",
           "| T | GEMM GFLOP | lm_head GFLOP | attn core GFLOP | GDN core GFLOP | conv GFLOP | total GFLOP | compulsory bytes (GB) | compute ms @989.5 | memory ms @4.8TB/s | roofline lower bound ms | per-linear serial bound ms | bound |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        f = r["flops"]
        vals = [f[k] / 1e9 for k in f]
        md.append(f"| {r['T']} | " + " | ".join(f"{v:,.2f}" for v in vals)
                  + f" | {r['total_flops']/1e9:,.1f} | {r['total_bytes']/1e9:.3f} | {r['compute_ms']:.3f} | "
                  f"{r['memory_ms']:.3f} | **{r['roofline_ms']:.3f}** | {r['serial_gemm_roofline_ms']:.3f} | {r['bound']} |")
    md += ["", "Formulas:", "",
           "- GEMM: `2 * T * params` for every decoder `nn.Linear` (q/k/v/o, in_proj_qkv/z/a/b, out_proj, gate/up/down).",
           "- lm_head: `2 * 1 * 5120 * 248320`: a prefill emits one token, so only the last row is projected (SGLang does the same).",
           f"- attention core: `16 layers * {hq} heads * 2 GEMMs * 2 * {dh} * T(T+1)/2` (causal pairs only).",
           f"- GDN core (chunk C=64, per chunk per v-head): `2C^2(3dk+2dv) + 2C^3/3 + 6 C dk dv`, `x {n_gdn} layers x {hv} heads x ceil(T/64)`.",
           "- compulsory bytes: all decoder weights + lm_head once, T embedding rows, KV cache and GDN state written once.",
           "- roofline lower bound = max(total flops / 989.5 TF, compulsory bytes / 4.8 TB/s) (perfect overlap of the whole forward).",
           "- per-linear serial bound = sum over linears of max(own flops / 989.5 TF, own weight bytes / 4.8 TB/s) + cores at peak: "
           "what one-kernel-per-linear can reach at best."]
    (ROOT / "FLOPS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
