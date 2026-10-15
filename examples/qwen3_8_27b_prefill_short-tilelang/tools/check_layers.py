#!/usr/bin/env python3
"""Run `tilefoundry check` on every decoder layer, teacher-forced.

For layer i and length T, the layer's input is the hidden state transformers fed
that layer (reference/T{T}/layers/layer{i}_hidden.pt), and the expected outputs are
what transformers' layer returned and wrote into its cache. One layer is one
`check` invocation (selection `<source>:<root>.layer{i}`), so the verdict is check's.

Bounds: only check --help's own rule, the same as the whole-prefill check
(tools/check_hir.sh): count the serial f32->bf16 landings of the longest chain and take
rtol = 2^-9 * sqrt(n). Inside one layer n is about 21 (rmsnorm, projections, conv,
silu, l2norm, ..., o_proj, residual, rmsnorm, gate/up, SwiGLU, down, residual), so
rtol = 2^-9 * sqrt(21) = 0.009 and every output is judged by
    rel_l2 <= 0.009,  cosine >= 1 - 0.009^2 / 2 = 0.99996,  nan_inf.
No per-element allclose: it needs an atol, and the rule gives none ("atol covers the
near-zero elements" names no number), so any atol would be a tolerance of my own.
(History, WORKLOG: earlier runs used allclose with atol = 0.009 x RMS for caches and
one bf16 ulp at the peak for the hidden output and K; those runs are archived under
reports/check/archive/.)

usage: check_layers.py SOURCE ROOT T [T ...] [--jobs N] [--tag NAME] [--vs-hir]
  SOURCE: model.py (authored HIR vs transformers) or runtime/twin.py (the twin vs
  transformers, or with --vs-hir the twin vs the evaluator running model.py)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

#: the `tilefoundry` CLI: $TILEFOUNDRY, or whatever is on PATH (this example is not
#: installed and does not carry a virtualenv of its own).
TILEFOUNDRY = os.environ.get("TILEFOUNDRY") or shutil.which("tilefoundry") or "tilefoundry"
RTOL = 0.009
COS = 0.99996


def command(source, root, T, i, kind, out_json, vs_hir=False):
    R = f"reference/T{T}"
    hidden = f"{R}/layers/layer{i}_hidden.pt"
    inputs = hidden if kind == "linear_attention" else f"{hidden},{R}/cos.pt,{R}/sin.pt"
    n_out = len(torch.load(ROOT / f"{R}/layers/layer{i}_expected.pt"))
    cmd = [TILEFOUNDRY, "check", f"{source}:{root}.layer{i}",
           "--inputs", f"files:{inputs}", "--weights", "ckpt:ckpt",
           "--dim", f"seq_len={T}", "--json", out_json]
    if not vs_hir:
        cmd += ["--expected", f"{R}/layers/layer{i}_expected.pt"]
    for j in range(n_out):
        cmd += ["--out", f"output[{j}]", "--fn", "rel_l2", "--max", str(RTOL), "--fn", "cosine", "--min", str(COS),
                "--fn", "nan_inf"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("root")
    ap.add_argument("lengths", nargs="+", type=int)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tag", default="hir_vs_hf")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--vs-hir", action="store_true",
                    help="reference = the evaluator running model.py (no --expected); for a runtime twin")
    args = ap.parse_args()
    config = json.loads((ROOT / "config.json").read_text())["text_config"]
    kinds = config["layer_types"]
    layers = range(64) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    outdir = ROOT / "reports/check" / f"{args.tag}_layers"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for T in args.lengths:
        for i in layers:
            out_json = str(outdir / f"T{T}_layer{i}.json")
            jobs.append((T, i, command(args.source, args.root, T, i, kinds[i], out_json, args.vs_hir),
                         out_json))

    def run(job):
        T, i, cmd, out_json = job
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        try:
            rep = json.loads(Path(out_json).read_text())
            passed = all(r["passed"] for r in rep["runs"])
            worst = {o["path"]: {f["fn"]: f.get("max_violation", f.get("rel_l2", f.get("cosine"))) for f in o["fns"]}
                     for o in rep["runs"][0]["outputs"]}
        except Exception as exc:  # noqa: BLE001
            passed, worst = False, {"error": (proc.stderr or str(exc))[-400:]}
        return dict(T=T, layer=i, kind=kinds[i], passed=passed, rc=proc.returncode, worst=worst)

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, jobs))
    summary = outdir / f"summary_{'_'.join(map(str, args.lengths))}.json"
    summary.write_text(json.dumps(results, indent=1))
    for T in args.lengths:
        rows = [r for r in results if r["T"] == T]
        print(f"T={T}: {sum(r['passed'] for r in rows)}/{len(rows)} layers PASS (check exit 0: "
              f"{sum(r['rc'] == 0 for r in rows)})")
        for r in rows:
            if not r["passed"]:
                print("  FAIL", r["layer"], r["kind"], json.dumps(r["worst"])[:300])
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
