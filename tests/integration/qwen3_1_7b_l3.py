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
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

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

# Canonical name -> published key. The seven projections and the head resolve to
# the keys their converters read; `layer{i}` is a subtree segment, not a leaf.
_ALIAS = {
    "w_embed": "model.embed_tokens.weight",
    "gamma_final": "model.norm.weight",
    "head_weight_raw": "lm_head.weight",
    "gamma_in": "input_layernorm.weight",
    "gamma_post": "post_attention_layernorm.weight",
    "gamma_q": "self_attn.q_norm.weight",
    "gamma_k": "self_attn.k_norm.weight",
    "q_proj_weight": "self_attn.q_proj.weight",
    "k_proj_weight": "self_attn.k_proj.weight",
    "v_proj_weight": "self_attn.v_proj.weight",
    "o_proj_weight": "self_attn.o_proj.weight",
    "gate_proj_weight": "mlp.gate_proj.weight",
    "up_proj_weight": "mlp.up_proj.weight",
    "down_proj_weight": "mlp.down_proj.weight",
}


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


def _emitted(venv: Path, work: Path, outside: Path) -> Path:
    """Ask the installation for its model directory and copy it where it will run."""
    done = subprocess.run(
        [str(venv / "bin" / "tilefoundry"), "models", MODEL, "--source"],
        cwd=str(outside), capture_output=True, text=True, check=False,
    )
    if done.returncode != 0:
        raise SystemExit(
            f"`tilefoundry models {MODEL} --source` exited {done.returncode}\n"
            f"{done.stderr.strip()}"
        )
    lines = done.stdout.splitlines()
    if not lines:
        raise SystemExit(f"`tilefoundry models {MODEL} --source` printed no directory")
    shipped = Path(lines[0])
    if not shipped.is_dir():
        raise SystemExit(f"`tilefoundry models {MODEL} --source` named no directory: {shipped}")
    copied = work / shipped.name
    shutil.copytree(shipped, copied)
    source = copied / "model.py"
    if not source.is_file():
        raise SystemExit(f"copied model directory has no model.py: {copied}")
    print(f"copied {MODEL} source directory to {copied}")
    return source


def _oracle(mount: Path):
    """The published model and tokenizer, at the dtype the checkpoint publishes."""
    tokenizer = AutoTokenizer.from_pretrained(str(mount))
    model = AutoModelForCausalLM.from_pretrained(str(mount), dtype=DTYPE)
    return tokenizer, model.to(DEV).eval()


def _rope_caches(config, rows: int):
    """Full cos / sin caches `[rows, head_dim]` from the model's own rotary class.

    Row `p` is the embedding for absolute position `p`, so gathering by `pos_ids`
    reproduces the cos / sin the published attention applies.
    """
    rotary = Qwen3RotaryEmbedding(config).to(DEV)
    reference = torch.zeros(1, rows, config.hidden_size, device=DEV, dtype=DTYPE)
    cos, sin = rotary(reference, torch.arange(rows, device=DEV).unsqueeze(0))
    return cos[0].to(DTYPE), sin[0].to(DTYPE)


def _loaded(source: Path, work: Path, mount: Path, n_layers: int):
    """The emitted decoder with the published weights bound, through `prepare`."""
    root = runpy.run_path(str(source))["Qwen3_1_7B_Decoder"].cloned()
    alias = {**_ALIAS, **{f"layer{i}": f"model.layers.{i}" for i in range(n_layers)}}
    raw = SafetensorsResource(str(mount), device="cpu", alias=alias, dtype=DTYPE)
    prepared = work / "prepared"
    root.prepare(raw, str(prepared), device="cpu")
    return root.load(SafetensorsResource(str(prepared), device=DEV))


def _hf_greedy(model, prompt_ids):
    """Hugging Face's own 16 greedy continuations, and the prompt's cache.

    Stepped by hand rather than through `generate`, so both sides run 16 steps
    whatever the tokens are and no `generation_config` field decides what greedy
    means here.
    """
    with torch.no_grad():
        prefill = model(prompt_ids[:, :-1], use_cache=True)
        context = tuple(
            (
                layer.keys.transpose(1, 2).contiguous(),
                layer.values.transpose(1, 2).contiguous(),
            )
            for layer in prefill.past_key_values.layers
        )

        cache, token, tokens = prefill.past_key_values, prompt_ids[:, -1:], []
        for _ in range(STEPS):
            step = model(token, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            token = torch.argmax(step.logits[:, -1], dim=-1, keepdim=True)
            tokens.append(int(token))
    return context, tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--venv", required=True, type=Path, help="the installation under test"
    )
    parser.add_argument(
        "--work", required=True, type=Path, help="scratch directory for the store"
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
    published = _published(args.checkpoint)
    args.work.mkdir(parents=True, exist_ok=True)

    source = _emitted(args.venv, args.work, args.outside)

    tokenizer, model = _oracle(args.checkpoint)
    loaded = _loaded(source, args.work, args.checkpoint, published["num_hidden_layers"])

    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEV)
    if prompt_ids.shape[1] < 2:
        raise SystemExit("a continuation needs a token of context")

    context, want = _hf_greedy(model, prompt_ids)
    cos, sin = _rope_caches(model.config, published["max_position_embeddings"])
    scale = torch.full(
        (1, 1, 1, 1), model.model.layers[0].self_attn.scaling, device=DEV, dtype=DTYPE
    )

    caches, token, got = context, prompt_ids[0, -1:], []
    for offset in range(STEPS):
        pos_ids = torch.tensor(
            [prompt_ids.shape[1] - 1 + offset], device=DEV, dtype=torch.int32
        )
        logits, entries = loaded.forward(token, cos, sin, pos_ids, scale, caches)
        caches = loaded.append_cache(caches, entries)
        token = torch.argmax(logits[0]).reshape(1).to(torch.int64)
        got.append(int(token))

    if got != want:
        step = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise SystemExit(
            f"diverged at step {step}: {tokenizer.decode(got)!r} "
            f"against {tokenizer.decode(want)!r}"
        )
    print(f"{STEPS} greedy steps agree: {tokenizer.decode(got)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
