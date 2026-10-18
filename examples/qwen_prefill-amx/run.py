"""One Prefill smoke run; no warmups, repetitions or performance comparison."""

import argparse, json
from pathlib import Path
import torch
from tilefoundry.runtime import SafetensorsResource
from config import config, ROOT
from scheduled import make_scheduled
from runtime_model import make_runtime, inputs


def run(model, sequence=128, prepared=None):
    torch.set_num_threads(8)
    c = config(model)
    semantic = make_scheduled(c, sequence)
    runtime = make_runtime(semantic, c)()
    source = Path(prepared) if prepared else ROOT / "artifacts" / ("prepared-" + model)
    runtime.load(SafetensorsResource(str(source), device="cpu"))
    tokens = (torch.arange(sequence) * 17 + 100) % min(30000, c["vocab_size"])
    logits = runtime(*inputs(c, sequence, tokens))
    assert tuple(logits.shape) == (1, c["vocab_size"]) and torch.isfinite(logits).all()
    report = dict(
        model=model,
        sequence=sequence,
        shape=list(logits.shape),
        argmax=int(logits.argmax()),
        finite=True,
        stats=runtime.stats,
        smoke_only=True,
        codegen=False,
    )
    (source / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("1.7b", "4b"), required=True)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--prepared")
    a = p.parse_args()
    run(a.model, a.sequence, a.prepared)
