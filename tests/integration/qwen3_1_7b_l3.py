"""Decode the published Qwen3-1.7B checkpoint with the source the wheel ships.

    <venv>/bin/python tests/integration/qwen3_1_7b_l3.py --venv <venv> --work <dir>

Sixteen greedy continuation steps against Hugging Face, compared as token IDs. The
IR side comes from the directory `tilefoundry models qwen3_1_7b --source` names in
an installation, copied and executed here rather than imported from this checkout.
Run by the installation's own interpreter, so the `tilefoundry` it imports is the
one tested.

Prefill is Hugging Face's: these kernels express one step at a time and not the
prompt pass before it, so what is compared is continuation decode.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tilefoundry
from tilefoundry.runtime import SafetensorsResource

CKPT_ENV = "TILEFOUNDRY_QWEN3_1_7B_CKPT"
MODEL = "qwen3_1_7b"

PROMPT = "Explain why the sky appears blue in one sentence."
STEPS = 16
DEV = "cuda"

#: The dtype the checkpoint publishes, and so the dtype both sides decode at.
DTYPE = torch.bfloat16

# The published keys this run reads off the mounted checkpoint.
_PUBLISHED = (
    "hidden_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "num_hidden_layers",
)

def _installed(venv: Path) -> None:
    """Refuse to run unless the `tilefoundry` imported here came out of *venv*."""
    where = Path(tilefoundry.__file__).resolve()
    if not where.is_relative_to(venv.resolve()):
        raise SystemExit(f"tilefoundry resolved to {where}, outside {venv}")
    print(f"tilefoundry: {where}")


def _published(mount: Path) -> dict:
    """The checkpoint's own `config.json`, refused unless it is this model.

    Any readable directory would otherwise satisfy the mount, and a mismatched one
    would report agreement about a model nobody asked for.
    """
    stated = mount / "config.json"
    if not stated.is_file():
        raise SystemExit(f"no config.json in {mount}")
    read = json.loads(stated.read_text(encoding="utf-8"))
    if read.get("model_type") != "qwen3":
        raise SystemExit(f"{mount} publishes model_type {read.get('model_type')!r}")
    missing = sorted(key for key in _PUBLISHED if key not in read)
    if missing:
        raise SystemExit(f"{stated} states no {', '.join(missing)}")
    if not sorted(mount.glob("model*.safetensors*")):
        raise SystemExit(f"no safetensors shard or index in {mount}")
    return read


def _source_directory(venv: Path, *command: str, outside: Path) -> Path:
    """The first directory a source-listing command from *venv* names."""
    done = subprocess.run(
        [str(venv / "bin" / "tilefoundry"), *command],
        cwd=str(outside), capture_output=True, text=True, check=False,
    )
    if done.returncode != 0:
        raise SystemExit(
            f"`tilefoundry {' '.join(command)}` exited {done.returncode}\n"
            f"{done.stderr.strip()}"
        )
    lines = done.stdout.splitlines()
    if not lines:
        raise SystemExit(f"`tilefoundry {' '.join(command)}` printed no directory")
    shipped = Path(lines[0])
    if not shipped.is_dir():
        raise SystemExit(f"`tilefoundry {' '.join(command)}` named no directory: {shipped}")
    return shipped


def _emitted(venv: Path, work: Path, outside: Path) -> tuple[Path, Path]:
    """Copy the shipped model, and locate the shipped decode driver."""
    shipped = _source_directory(venv, "models", MODEL, "--source", outside=outside)
    copied = work / shipped.name
    shutil.copytree(shipped, copied)
    model_source = copied / "model.py"
    alias_source = copied / "hf_alias.py"
    if not model_source.is_file() or not alias_source.is_file():
        raise SystemExit(f"copied model directory has no model.py: {copied}")
    print(f"copied {MODEL} source directory to {copied}")
    driver = _source_directory(venv, "tutorial", "orchestrator", "causal_lm", outside=outside)
    generation_source = driver / "generation.py"
    if not generation_source.is_file():
        raise SystemExit(f"shipped causal_lm source has no generation.py: {driver}")
    return model_source, generation_source


def _oracle(mount: Path):
    """The published model and tokenizer, at the dtype the checkpoint publishes."""
    tokenizer = AutoTokenizer.from_pretrained(str(mount))
    model = AutoModelForCausalLM.from_pretrained(str(mount), dtype=DTYPE)
    return tokenizer, model.to(DEV).eval()


def _loaded(source: Path, mount: Path):
    """The emitted decoder with its published weights bound directly."""
    namespace = runpy.run_path(str(source))
    root = namespace["Qwen3_1_7B"].cloned()
    alias = runpy.run_path(str(source.with_name("hf_alias.py")))["hf_alias"](
        namespace["config"]
    )
    return root.load(SafetensorsResource(str(mount), device=DEV, alias=alias))


def _hf_greedy(model, prompt_ids):
    """Hugging Face's own 16 greedy continuations.

    Stepped by hand rather than through `generate`, so both sides run 16 steps
    whatever the tokens are and no `generation_config` field decides what greedy
    means here.
    """
    with torch.no_grad():
        prefill = model(prompt_ids[:, :-1], use_cache=True)
        cache, token, tokens = prefill.past_key_values, prompt_ids[:, -1:], []
        for _ in range(STEPS):
            step = model(token, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            token = torch.argmax(step.logits[:, -1], dim=-1, keepdim=True)
            tokens.append(int(token))
    return tokens


def _recorded_greedy(generation: dict[str, object], tokens: list[int]):
    """Return greedy sampling that skips decode's unmeasured warm sample."""
    warmed = False

    def sample(logits) -> int:
        nonlocal warmed

        token = generation["greedy"](logits)
        if warmed:
            tokens.append(token)
        warmed = True
        return token

    return sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--venv", required=True, type=Path, help="the installation under test"
    )
    parser.add_argument(
        "--work", required=True, type=Path, help="scratch directory for copied source"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get(CKPT_ENV),
        help=f"the published checkpoint mount (default: ${CKPT_ENV})",
    )
    parser.add_argument(
        "--outside",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="a directory that is not a checkout, to read the source from",
    )
    args = parser.parse_args(argv)

    if args.checkpoint is None:
        raise SystemExit(f"no --checkpoint and no {CKPT_ENV}")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"--checkpoint {args.checkpoint} is not a directory")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this decodes two copies of a 1.7B model")

    _installed(args.venv)
    _published(args.checkpoint)
    args.work.mkdir(parents=True, exist_ok=True)
    args.outside.mkdir(parents=True, exist_ok=True)

    source, generation_source = _emitted(args.venv, args.work, args.outside)

    tokenizer, model = _oracle(args.checkpoint)
    loaded = _loaded(source, args.checkpoint)
    generation = runpy.run_path(str(generation_source))

    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEV)
    if prompt_ids.shape[1] < 2:
        raise SystemExit("a continuation needs a token of context")

    want = _hf_greedy(model, prompt_ids)
    got: list[int] = []
    done = generation["decode"](
        loaded,
        tokenizer,
        PROMPT,
        max_new=STEPS,
        sampler=_recorded_greedy(generation, got),
        device=DEV,
    )

    if got != want:
        step = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise SystemExit(
            f"diverged at step {step}: {tokenizer.decode(got)!r} "
            f"against {tokenizer.decode(want)!r}"
        )
    print(
        f"{done.tokens} greedy steps agree at "
        f"{done.tokens / done.seconds:.1f} tok/s: {tokenizer.decode(got)!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
