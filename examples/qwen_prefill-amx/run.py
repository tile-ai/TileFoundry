"""Load Qwen weights and execute one text prefill through its runtime module."""

import argparse
from pathlib import Path
import torch
from tilefoundry.runtime import SafetensorsResource
from config import config
from runtime_model import make_runtime, inputs


def run(model, sequence=128, prepared=None, checkpoint=None):
    torch.set_num_threads(8)
    c = config(model)
    runtime = make_runtime(c, sequence)()
    source = Path(prepared) if prepared else Path(__file__).parent / "prepared" / model
    if not source.exists():
        if checkpoint is None:
            raise ValueError(
                "Supply --checkpoint to prepare weights, or --prepared for existing weights"
            )
        from weights import aliases, GGUFResource

        runtime.module.prepare(
            GGUFResource(checkpoint, aliases(c)), str(source), device="cpu"
        )
    runtime.load(SafetensorsResource(str(source), device="cpu"))
    tokens = (torch.arange(sequence) * 17 + 100) % min(30000, c["vocab_size"])
    with torch.inference_mode():
        logits = runtime(*inputs(c, sequence, tokens))
    if not torch.isfinite(logits).all():
        raise RuntimeError("Prefill produced non-finite logits")
    token = int(logits.argmax())
    print(f"{model}: logits {tuple(logits.shape)}, next token {token}")
    return logits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("1.7b", "4b"), required=True)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--prepared")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    run(a.model, a.sequence, a.prepared, a.checkpoint)
