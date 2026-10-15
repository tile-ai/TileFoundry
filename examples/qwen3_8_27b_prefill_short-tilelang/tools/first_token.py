#!/usr/bin/env python3
"""First token and logits error of the tilelang prefill vs transformers, per length.

Runs the runtime twin's prefill on reference/T{T}/token_ids.pt (the same ids the
transformers reference used) and compares with reference/T{T}/expected.pt[0] (the
transformers logits of the last position). Writes reports/first_token.json.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prompting  # noqa: E402
from runtime import engine as eng  # noqa: E402
from runtime.twin import Qwen3_8_27BTwin  # noqa: E402
from tilefoundry.runtime import SafetensorsResource  # noqa: E402


def main():
    lengths = [int(a) for a in sys.argv[1:]] or list(prompting.LENGTHS)
    tw = Qwen3_8_27BTwin()
    tw.load(SafetensorsResource(str(ROOT / "ckpt"), device="cuda"))
    rows = []
    for T in lengths:
        ids = torch.load(ROOT / f"reference/T{T}/token_ids.pt").cuda()
        cos, sin = prompting.rope_tables(T, eng.CFG, "cuda")
        logits = tw(ids, cos, sin)[0].float().clone()
        ref = torch.load(ROOT / f"reference/T{T}/expected.pt")[0].cuda().float()
        meta = json.loads((ROOT / f"reference/T{T}/meta.json").read_text())
        mine, theirs = int(logits.argmax()), int(ref.argmax())
        top2 = torch.topk(logits, 2).values[0]
        row = dict(T=T, first_token_mine=mine, first_token_transformers=theirs, equal=mine == theirs,
                   transformers_top2_margin=meta["top2_margin"], mine_top2_margin=float(top2[0] - top2[1]),
                   logits_max_abs_err=float((logits - ref).abs().max()),
                   logits_rel_l2=float((logits - ref).norm() / ref.norm()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/first_token.json").write_text(json.dumps(rows, indent=1))
    sys.exit(0 if all(r["equal"] for r in rows) else 1)


if __name__ == "__main__":
    main()
