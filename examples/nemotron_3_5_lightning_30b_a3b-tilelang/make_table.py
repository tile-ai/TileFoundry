"""The three-row table: this kernel, SGLang, and the roofline lower bound.

The roofline row is per-token traffic divided by HBM bandwidth. The traffic is
`analyze`'s own figure for everything that does not depend on the context, plus
a hand count of the K/V read, because the shipped HIR's traffic model
undercounts that one term -- it reports 0.10 GB at 262144 where the cache really
is 1.61 GB, since the sliced read is priced at the slice rather than at what the
context makes it. Substituting the term that is wrong is stated here rather than
hidden inside a number.
"""
from __future__ import annotations

import argparse
import json
import re

#: `analyze model.py --memory --dim ctx_full=0 --dim ctx_tail=1`, gmem read+write
#: (reports/final_tc_ctx1.txt).
BASE_BYTES = 6_752_466_852 + 244_989_744
#: 6 full-attention layers x (K + V) x 2 KV heads x 128 dims x 2 bytes.
KV_PER_POS = 6 * 2 * 2 * 128 * 2
#: H200 SXM spec HBM3e bandwidth. A *lower* bound on time uses the peak.
HBM = 4.8e12
#: What a pure stream actually reaches on this card (`kbench/bw3.py`): 95.3% of
#: spec. Nothing on this card has read faster, so this is the floor an
#: implementation can be held to, where HBM is the floor it cannot go under.
HBM_REAL = 4.576e12


def roofline(ctx):
    b = BASE_BYTES + KV_PER_POS * ctx
    return b, b / HBM


def load_rows(path):
    """The sweep's JSON, or its log if the sweep has not finished writing one."""
    if path.endswith(".json"):
        try:
            return json.load(open(path))["rows"]
        except FileNotFoundError:
            path = path.replace(".json", ".log").replace("mine_mega", "bench_mine")
    rows = []
    for line in open(path):
        m = re.match(r"ctx\s+(\d+)\s+([\d.]+) tok/s\s+([\d.]+) ms", line)
        if m:
            rows.append({"ctx": int(m[1]), "tok_s": float(m[2]),
                         "ms_per_token": float(m[3])})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine", default="reports/mine_mega.json")
    ap.add_argument("--sglang", default="sglang_baseline.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mine = {r["ctx"]: r for r in load_rows(a.mine)}
    sg = {r["ctx"]: r for r in json.load(open(a.sglang))["rows"]}
    ctxs = sorted(set(mine) | set(sg))
    head = ["| context | " + " | ".join(f"{c}" for c in ctxs) + " |",
            "|---|" + "---|" * len(ctxs)]
    rows = [
        ("**this, tok/s**", [f"**{mine[c]['tok_s']:.1f}**" if c in mine else "—" for c in ctxs]),
        ("SGLang, tok/s", [f"{sg[c]['decode_tps']:.1f}" if c in sg else "—" for c in ctxs]),
        ("roofline floor, tok/s", [f"{1 / roofline(c)[1]:.0f}" for c in ctxs]),
        ("this, ms/token", [f"{mine[c]['ms_per_token']:.3f}" if c in mine else "—" for c in ctxs]),
        ("SGLang, ms/token", [f"{sg[c]['ms_per_token']:.3f}" if c in sg else "—" for c in ctxs]),
        ("roofline, ms/token", [f"{roofline(c)[1] * 1e3:.3f}" for c in ctxs]),
        ("bytes a token, GB", [f"{roofline(c)[0] / 1e9:.3f}" for c in ctxs]),
        ("floor at measured bandwidth, ms/token",
         [f"{roofline(c)[0] / HBM_REAL * 1e3:.3f}" for c in ctxs]),
        ("**off the spec floor**", [f"{mine[c]['ms_per_token'] / (roofline(c)[1] * 1e3):.2f}x"
                                    if c in mine else "—" for c in ctxs]),
    ]
    text = "\n".join(head + [f"| {n} | " + " | ".join(v) + " |" for n, v in rows])
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
