#!/usr/bin/env python3
"""Print the artifact chain as a Markdown table: for each kernel, which authored
function of model.py it implements, its scheduled HIR, the finalized TIR, the
runtime (tilelang) kernel, and which prompt lengths it serves.

A row exists only if both the scheduled HIR file and its finalized TIR exist, so the
table cannot claim an artifact that is not on disk.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import plan  # noqa: E402

LENGTHS = (128, 256, 512, 1024)
GEMM_FN = {
    "gdn_in": ("`Qwen3_8DeltaNetLayer.gated_delta_net`: x @ [w_qkv|w_z|w_b|w_a]^T", "gemm_kernel(..., 'store')"),
    "att_in": ("`Qwen3_8AttentionLayer.gated_attention`: x @ [w_q|w_k|w_v]^T", "gemm_kernel(..., 'store')"),
    "gate_up": ("`*.swiglu`: silu(x @ w_gate^T) * (x @ w_up^T)", "gemm_kernel(..., 'swiglu2')"),
    "o_proj": ("`*.layer`: hidden + mixer @ w_out^T / w_o^T", "gemm_kernel(..., 'residual' or 'splitk')"),
    "down": ("`*.layer`: attended + act @ w_down^T", "gemm_kernel(..., 'residual' or 'splitk')"),
}
SMALL = [
    ("rmsnorm_T{T}", "`*.layer`: tf.rms_norm(h, 1 + gamma_in / gamma_post)", "small.rmsnorm_kernel(T, 5120, plan.norm_rows)"),
    ("embed_T{T}", "`Qwen3_8_27B.embed`", "small.embed_kernel(T, ...)"),
    ("reduce_sk2_T{T}", "`*.layer`: residual add finishing a split-K o_proj/down", "gemm.splitk_reduce_kernel(T, 5120, 2, True)"),
]
ONCE = [
    ("rmsnorm_last", "`Qwen3_8_27B.final_norm` (last row only)", "small.rmsnorm_kernel(1, 5120, 1)", "all"),
    ("lm_head", "`Qwen3_8_27B.lm_head`", "small.gemv_kernel(248320, 5120, 16, 256)", "all"),
]


def exists(kind, name):
    return (ROOT / f"scheduled/{kind}/{name}.py").exists() and (ROOT / f"tir/{kind}/{name}.py").exists()


def rows():
    out = []
    for linear, (fn, rt) in GEMM_FN.items():
        for T in LENGTHS:
            name = f"{linear}_T{T}"
            if exists("gemm", name):
                cfg = plan.for_length(T).gemm[linear]
                out.append((fn, f"`scheduled/gemm/{name}.py`", f"`tir/gemm/{name}.py`", f"`{rt}` {cfg.tag()}", str(T)))
    for pattern, fn, rt in SMALL:
        for T in LENGTHS:
            name = pattern.format(T=T)
            if exists("small", name):
                out.append((fn, f"`scheduled/small/{name}.py`", f"`tir/small/{name}.py`", f"`{rt}`", str(T)))
    for name, fn, rt, lengths in ONCE:
        if exists("small", name):
            out.append((fn, f"`scheduled/small/{name}.py`", f"`tir/small/{name}.py`", f"`{rt}`", lengths))
    extra = ROOT / "tools/chain_rows_extra.py"
    if extra.exists():
        ns = {}
        exec(extra.read_text(), ns)
        for r in ns["ROWS"]:
            if (ROOT / r[1].strip("`")).exists() and (ROOT / r[2].strip("`")).exists():
                out.append(r)
    return out


def main():
    print("| authored function (model.py) | scheduled HIR | finalized TIR | runtime kernel (runtime/kernels) | serves T |")
    print("|---|---|---|---|---|")
    for r in rows():
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
