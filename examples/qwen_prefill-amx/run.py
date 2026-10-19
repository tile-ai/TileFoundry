"""Load the original CPU/AMX runtime and execute one text prefill."""

import argparse
from pathlib import Path
import torch
from config import config
from runtime_model import QwenPrefill
from weights import PreparedModelResource


def run(model, checkpoint, sequence=128, prepared=None):
    torch.set_num_threads(8)
    source = Path(prepared) if prepared else Path(__file__).parent / "prepared" / model
    resource = PreparedModelResource(checkpoint, source)
    with QwenPrefill(config(model), sequence=sequence) as runtime:
        runtime.load(resource)
        tokens = (torch.arange(sequence, dtype=torch.int32) * 17 + 100) % min(
            30000, runtime.vocab
        )
        logits = runtime(tokens)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Prefill produced non-finite logits")
        print(
            f"{model}: logits {tuple(logits.shape)}, next token {int(logits.argmax())}"
        )
        return logits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("1.7b", "4b"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--prepared")
    a = p.parse_args()
    run(a.model, a.checkpoint, a.sequence, a.prepared)
