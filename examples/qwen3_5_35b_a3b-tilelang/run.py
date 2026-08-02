#!/usr/bin/env python
"""Decode real tokens from Qwen3.5-35B-A3B and report how fast.

    python run.py --prompt "The capital of France is" --max-new-tokens 32

Two more switches, both for making a run comparable to another run:

    --seed N     seeds the sampler
    --greedy     turns sampling off entirely (argmax), for diffing token streams

What the printed rate means
---------------------------
`tok/s` is **decode** throughput: new tokens divided by the time to produce
them, measured after the prompt is consumed. It excludes weight loading and
kernel compilation, which happen once, and it excludes prefill, which is
reported separately. This is one token per step, batch one -- the regime
`model.py` declares (`S = 1`) -- so it is a latency number, not a throughput
number: the floor is one pass over 2.95 G active parameters (~5.9 GB at bf16)
per token, whatever the batch would have allowed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from config import CKPT, REAL, from_checkpoint, truncated  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=None, help="seed the sampler")
    p.add_argument(
        "--greedy", action="store_true", help="argmax instead of sampling"
    )
    # No default: where the weights live is a fact about the machine. `$QWEN35_CKPT`
    # supplies it when set, otherwise this has to be said.
    p.add_argument("--ckpt", default=str(CKPT) if CKPT else None, required=CKPT is None,
                   help="the published checkpoint directory (Qwen/Qwen3.5-35B-A3B)")
    # Everything below is for working on this, not for using it.
    p.add_argument(
        "--layers", type=int, default=None,
        help="run only the first N layers (a fast loop, not the published model)",
    )
    p.add_argument(
        "--graph", dest="graph", action="store_true", default=True,
        help="capture the decode step into a CUDA graph (default)",
    )
    p.add_argument("--no-graph", dest="graph", action="store_false")
    p.add_argument(
        "--capacity", type=int, default=None,
        help="KV cache capacity for the graphed path (default: what the run needs)",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--impl", default=None,
        help="TF_IMPL override, e.g. 'torch' or 'linear_attention:torch'",
    )
    p.add_argument(
        "--verify-config", action="store_true",
        help="check config.REAL against the checkpoint's own config.json and exit",
    )
    p.add_argument(
        "--compare", metavar="PATH", default=None,
        help="diff the greedy token ids against a tools/hf_greedy.json",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def verify_config(ckpt) -> int:
    """`config.REAL` against the published file, field by field.

    `REAL` is reconstructed by hand (see `config.py`'s docstring: the fixture's
    own config module does not ship). A reconstruction that is wrong in a way
    every shape check accepts is the failure mode that matters, so this exists.
    """
    published = from_checkpoint(ckpt, dt=REAL.dt, max_ctx=REAL.max_ctx)
    bad = []
    for field in REAL.__dataclass_fields__:
        mine, theirs = getattr(REAL, field), getattr(published, field)
        mark = "ok " if mine == theirs else "DIFF"
        if mine != theirs:
            bad.append(field)
        shown = (
            f"{len(mine)} entries, cycle {mine[:4]}"
            if field == "layer_types" else mine
        )
        print(f"  {mark} {field:22s} {shown}")
    print()
    for name in ("pass_dim", "gqa_group", "gdn_key_dim", "gdn_value_dim",
                 "gdn_conv_dim", "gdn_conv_context", "gdn_v_per_k", "n_layers"):
        print(f"  --  {name:22s} {getattr(REAL, name)}   (derived)")
    if bad:
        print(f"\nFAIL: {bad} disagree with {ckpt}/config.json")
        return 1
    print("\nPASS: every reconstructed field matches the published config.")
    print("      dt and max_ctx are the fixture's own choices, read back out of")
    print("      `tilefoundry models qwen3_5_35b_a3b`, and are not in that file.")
    return 0


def active_bytes(cfg, *, bytes_per_weight=2):
    """Weights read for one token: dense everything, plus `top_k` of `n_experts`.

    The floor this sets is the honest one to compare a decode rate against. It is
    not the parameter count: a 35 G-parameter MoE reads ~2.95 G of them per token,
    because 248 of 256 experts per layer are not selected.
    """
    h, e = cfg.hidden, cfg.n_experts  # noqa: F841 -- e documents the ratio below
    per_layer = {
        "linear_attention": (
            h * cfg.gdn_conv_dim + h * cfg.gdn_value_dim + cfg.gdn_value_dim * h
            + 2 * h * cfg.gdn_n_v_heads
        ),
        "full_attention": (
            h * cfg.n_q_heads * cfg.head_dim * 2 + 2 * h * cfg.n_kv_heads * cfg.head_dim
            + cfg.n_q_heads * cfg.head_dim * h
        ),
    }
    moe = (
        h * cfg.n_experts                                   # router
        + cfg.top_k * 3 * cfg.moe_intermediate * h          # the selected experts
        + 3 * cfg.shared_intermediate * h + h               # the shared expert
    )
    total = sum(per_layer[kind] + moe for kind in cfg.layer_types)
    total += cfg.hidden * cfg.vocab                         # lm_head
    return total * bytes_per_weight


def sample(logits, *, greedy, temperature, top_k, top_p, generator):
    """One token from one row of logits.

    The published `generation_config.json` is `do_sample: true`, temperature 1.0,
    top_k 20, top_p 0.95; those are the defaults here. `--greedy` is argmax, and
    is what two implementations get compared through: a sampled stream diverges
    on the first differing draw even when both are right.

    Everything below the top-k happens on **20** elements, not 248320. The
    obvious order -- mask to the top k, softmax the whole vocabulary, then sort
    it for the nucleus -- costs 8.9 ms a token here, against a 3.36 ms decode
    step: the `sort` alone is 248320 elements of eager torch, and the step it is
    deciding for is a single CUDA graph replay. Taking the top k *first* is not
    an approximation, because top-p is applied inside the top-k survivors either
    way; it just declines to sort 248300 values that are already excluded.
    """
    row = logits.reshape(-1)
    if greedy:
        return int(torch.argmax(row).item())

    k = min(top_k or row.numel(), row.numel())
    values, ids = torch.topk(row.float(), k)
    if temperature != 1.0:
        values = values / max(temperature, 1e-6)
    probs = torch.softmax(values, dim=-1)
    if top_p and top_p < 1.0:
        # `topk` already returned them descending. Keep the tokens up to and
        # including the one that crosses top_p, and never fewer than one.
        keep = (torch.cumsum(probs, dim=-1) - probs) < top_p
        keep[0] = True
        probs = probs * keep
    probs = probs / probs.sum()
    return int(ids[torch.multinomial(probs, 1, generator=generator)].item())


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verify_config:
        return verify_config(args.ckpt)

    if args.impl is not None:
        os.environ["TF_IMPL"] = args.impl
    # Imported after TF_IMPL is set: the twins bind their implementations at
    # import time so a per-call dispatch never appears in the step.
    import runtime_model as rt  # noqa: PLC0415
    import weights as wt  # noqa: PLC0415

    cfg = REAL if args.layers is None else truncated(args.layers)
    if args.layers is not None:
        print(
            f"note: --layers {args.layers} runs a {args.layers}-layer prefix of the "
            f"published {REAL.n_layers}-layer stack. The tokens are not the model's."
        )

    tok = wt.tokenizer(args.ckpt)
    prompt_ids = tok.encode(args.prompt).ids
    if not prompt_ids:
        print("empty prompt after tokenisation", file=sys.stderr)
        return 2

    need = len(prompt_ids) + args.max_new_tokens + 1
    capacity = args.capacity or min(max(need, 64), cfg.max_ctx)
    if need > cfg.max_ctx:
        print(
            f"prompt + new tokens is {need}, and the reference declares "
            f"ctx_len < {cfg.max_ctx}",
            file=sys.stderr,
        )
        return 2

    print(f"loading {args.ckpt} ...", flush=True)
    session = rt.Session(
        cfg, ckpt=args.ckpt, capacity=capacity, verbose=args.verbose
    )
    print(
        f"loaded {session.loaded_bytes / 1e9:.1f} GB in {session.load_seconds:.1f}s"
        f"  ({session.loaded_bytes / 1e9 / session.load_seconds:.2f} GB/s)",
        flush=True,
    )

    generator = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    driver = session.graphed() if args.graph else session
    if args.graph:
        t0 = time.perf_counter()
        driver.build()
        print(f"captured the decode step in {time.perf_counter() - t0:.1f}s", flush=True)

    eos = {248046, 248044}
    picks = dict(
        greedy=args.greedy, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, generator=generator,
    )

    # -- prefill: one step per prompt token, because the step is one token ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = None
    for token in prompt_ids:
        logits = driver.step(token)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # One untimed draw. The first `torch.multinomial` / `torch.topk` of a process
    # loads its CUDA kernels lazily, which costs 200-370 ms once; over 32 tokens
    # that is 6-12 ms/token of pure start-up landing on the reported rate. It is
    # the same category as weight loading and kernel compilation, which this
    # measurement already excludes. Measured back to back in one process: first
    # sampled pass 8.93 ms/token, second 3.52, against 3.19 greedy -- so the real
    # cost of sampling is 0.33 ms/token and the rest was start-up.
    sample(logits, **picks)
    torch.cuda.synchronize()

    # -- decode --------------------------------------------------------------
    produced: list[int] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.max_new_tokens):
        nxt = sample(logits, **picks)
        produced.append(nxt)
        if nxt in eos:
            break
        logits = driver.step(nxt)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    text = tok.decode(produced)
    print()
    print("=" * 72)
    print(args.prompt + text)
    print("=" * 72)
    print()
    print(f"prompt        {len(prompt_ids)} tokens   {prefill_s * 1e3:.1f} ms"
          f"   ({len(prompt_ids) / prefill_s:.1f} tok/s)")
    print(f"generated     {len(produced)} tokens   {decode_s * 1e3:.1f} ms")
    print(f"              {decode_s / len(produced) * 1e3:.2f} ms/token")
    print()
    print(f"  {len(produced) / decode_s:.1f} tok/s")
    print()
    active_gb = active_bytes(cfg) / 1e9
    ms = decode_s / len(produced) * 1e3
    print(f"note: {active_gb:.2f} GB of weights are read per token"
          f"{' (this truncated stack)' if args.layers else ''}, so 4.8 TB/s of HBM"
          f" is a {active_gb / 4.8:.2f} ms/token floor"
          f"  ({4.8 / active_gb * 1e3:.0f} tok/s).")
    print(f"      this run moves them at {active_gb / (ms / 1e3) / 1e3:.2f} TB/s effective"
          f"  ({active_gb / (ms / 1e3) / 1e3 / 4.8 * 100:.0f}% of peak).")

    if args.compare:
        with open(args.compare) as fh:
            oracle = json.load(fh)
        theirs = oracle["greedy_ids"][: len(produced)]
        same = sum(1 for a, b in zip(produced, theirs) if a == b)
        first = next(
            (i for i, (a, b) in enumerate(zip(produced, theirs)) if a != b), None
        )
        print()
        print(f"vs {args.compare}: {same}/{len(theirs)} ids equal"
              + ("" if first is None else f", first difference at index {first}"))
        print(f"  mine   {produced[:16]}")
        print(f"  theirs {theirs[:16]}")
        return 0 if first is None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
