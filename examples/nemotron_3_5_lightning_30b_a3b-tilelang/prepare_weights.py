"""Write the prepared checkpoint the two twins load from.

`Module.prepare` reads the published checkpoint through `hf_alias`, assembles
each MoE layer's 128 per-expert tensors into one stacked tensor, applies the
layout renames, validates every result against its declaration, and writes one
safetensors directory keyed by the canonical names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tilefoundry.runtime import SafetensorsResource

import model
from hf_alias import hf_alias
import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="published checkpoint (or $NEMOTRON35_CKPT)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = SafetensorsResource(str(paths.need("ckpt", args.ckpt)), device="cpu",
                              alias=hf_alias(model.config))
    model.Nemotron35Lightning30BA3B.prepare(raw, args.out, device="cpu")
    print("prepared ->", args.out)
    total = sum(p.stat().st_size for p in Path(args.out).glob("*.safetensors"))
    print(f"{total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
