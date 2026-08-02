"""Turn the published MiniCPM3-4B checkpoint into the canonical weights the
authored Module declares.

The published file is a single ``pytorch_model.bin`` and its head is *tied*: the
checkpoint has no ``lm_head.weight`` at all, so the alias table the shipped model
ships with -- which names one -- gets that one entry pointed at the embedding
table instead. Everything else is `hf_alias` verbatim.

    .venv/bin/python tools/prep_ckpt.py [--out prepared]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

def main() -> None:
    ap = argparse.ArgumentParser()
    # No default: the published checkpoint's location is the caller's to state.
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(HERE / "prepared"))
    args = ap.parse_args()

    from tilefoundry.runtime.resource import DictResource

    from hf_alias import hf_alias
    from model import MiniCPM3_4B, config

    raw = torch.load(
        str(Path(args.ckpt) / "pytorch_model.bin"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    print(f"raw tensors: {len(raw)}")

    alias = dict(hf_alias(config))
    # Tied head: the published checkpoint stores no `lm_head.weight`.
    assert "lm_head.weight" not in raw, "checkpoint has its own head; drop the tie"
    alias["head_weight_raw"] = "model.embed_tokens.weight"
    alias.pop("w_head", None)  # a converter owns it; the Preprocessed entry would shadow nothing

    resource = DictResource(raw, alias=alias)
    out = Path(args.out)
    print(f"preparing into {out} ...")
    MiniCPM3_4B.prepare(resource, str(out), device="cpu")
    total = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"done: {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
