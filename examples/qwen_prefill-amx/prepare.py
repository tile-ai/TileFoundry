"""Canonical GGUF weights and offline layout conversion via Module.prepare."""

from pathlib import Path
import argparse, json
from scheduled import make_scheduled
from config import config, ROOT
from weights import aliases, GGUFResource


def prepare(model, checkpoint, output=None, sequence=128):
    c = config(model)
    program = make_scheduled(c, sequence)
    out = Path(output) if output else ROOT / "artifacts" / ("prepared-" + model)
    program.prepare(GGUFResource(Path(checkpoint), aliases(c)), str(out), device="cpu")
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                model=model,
                sequence=sequence,
                checkpoint=str(Path(checkpoint).resolve()),
                format="module-list-v1",
            ),
            indent=2,
        )
        + "\n"
    )
    print(out, flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("1.7b", "4b"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output")
    p.add_argument("--sequence", type=int, default=128)
    a = p.parse_args()
    prepare(a.model, a.checkpoint, a.output, a.sequence)
