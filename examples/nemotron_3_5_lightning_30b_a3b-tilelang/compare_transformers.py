"""Greedy, token for token, against the published `transformers` implementation.

Both sides start from the same prompt and take the largest logit every step. The
two run on different GPUs so neither has to share memory with the other; the
comparison is on the token ids, which is what a decode step is finally judged by.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import paths


def ours(prompt_ids, steps, prepared, device, impl, forced=None):
    """Greedy through the twin. With *forced*, the context is the reference's own
    tokens, so every step is asked the same question the reference was asked and
    a disagreement is this step's arithmetic rather than an earlier step's."""
    os.environ["NEMO_IMPL"] = impl
    from tilefoundry.runtime import SafetensorsResource  # noqa: PLC0415
    import runtime_model  # noqa: PLC0415

    twin = runtime_model.Nemotron35Lightning30BA3BRuntime()
    twin.load(SafetensorsResource(prepared, device=device))
    caches = twin.init_caches(device=device, capacity=len(prompt_ids) + steps + 8)
    ids = torch.empty(len(prompt_ids) + steps, dtype=torch.int64, device=device)
    ids[: len(prompt_ids)] = torch.tensor(prompt_ids, device=device)
    got = []
    for step in range(len(prompt_ids) + steps - 1):
        args = twin.prepare_inputs_for_generation(ids[: step + 1], step, caches, device=device)
        logits, fresh = twin.forward(*args)
        caches = twin.append_cache(caches, fresh)
        if step >= len(prompt_ids) - 1:
            row = logits.reshape(-1).float()
            # `argmax`, not `topk`: on a tie they pick different entries, and the
            # reference takes the lowest index. `topk` is only for the margin.
            token = int(row.argmax())
            top = torch.topk(row, 2).values
            got.append((token, float(top[0] - top[1])))
            at = step + 1 - len(prompt_ids)
            ids[step + 1] = forced[at] if forced is not None else token
    return got


def theirs(prompt_ids, steps, ckpt, device):
    """Greedy through the published implementation, one token at a time.

    Not `generate`: that consumes the prompt in one batched forward, and this
    model's batched Mamba path is a chunked scan -- a different arithmetic from
    the recurrence a decode step runs. Driving it token by token compares two
    implementations of one algorithm instead of two algorithms.
    """
    from transformers import AutoModelForCausalLM  # noqa: PLC0415
    from transformers.cache_utils import DynamicCache  # noqa: PLC0415

    hf = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    cache = DynamicCache(config=hf.config)
    got, token = [], None
    with torch.no_grad():
        for step in range(len(prompt_ids) + steps - 1):
            token = prompt_ids[step] if step < len(prompt_ids) else token
            out = hf(torch.tensor([[token]], device=device),
                     past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if step >= len(prompt_ids) - 1:
                token = int(out.logits[0, -1].argmax().item())
                got.append(token)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--ckpt", default=None, help="or $NEMOTRON35_CKPT")
    ap.add_argument("--prepared", default=None, help="or $NEMOTRON35_PREPARED")
    ap.add_argument("--our-device", default="cuda:0")
    ap.add_argument("--their-device", default="cuda:1")
    ap.add_argument("--impl", default="ops")
    ap.add_argument("--side", default="both", choices=["both", "ours", "theirs"])
    ap.add_argument("--save", default=None)
    ap.add_argument("--teacher", action="store_true",
                    help="feed the reference's tokens back, so each step is asked"
                         " the same question the reference was asked")
    ap.add_argument("--load", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415

    ckpt = str(paths.need("ckpt", a.ckpt))
    prepared = str(paths.need("prepared", a.prepared))
    tok = AutoTokenizer.from_pretrained(ckpt)
    prompt_ids = tok.encode(a.prompt)
    print(f"prompt {len(prompt_ids)} tokens: {prompt_ids}")

    saved = json.load(open(a.load)) if a.load else {}
    if a.side in ("ours", "both"):
        forced = saved.get("theirs") if a.teacher else None
        pairs = ours(prompt_ids, a.steps, prepared, a.our_device, a.impl, forced)
        mine = [t for t, _ in pairs]
        saved["ours"] = mine
        saved["margin"] = [m for _, m in pairs]
        print("ours  ", mine)
        print("       ", repr(tok.decode(mine)))
    if a.side in ("theirs", "both"):
        ref = theirs(prompt_ids, a.steps, ckpt, a.their_device)
        saved["theirs"] = ref
        print("theirs", ref)
        print("       ", repr(tok.decode(ref)))
    if a.save:
        json.dump(saved, open(a.save, "w"))
    if "ours" in saved and "theirs" in saved:
        mine, ref = saved["ours"], saved["theirs"]
        n = min(len(mine), len(ref))
        same = next((i for i in range(n) if mine[i] != ref[i]), n)
        bad = [i for i in range(n) if mine[i] != ref[i]]
        print(f"\nidentical for {same} of {n} steps"
              + (f"; {len(bad)} of {n} differ in all" if a.teacher else ""))
        margin = saved.get("margin")
        if margin and bad:
            print("step  margin between the reference's top two logits")
            for i in bad:
                print(f"{i:4d}  {margin[i]:.4f}")
        print("MATCH" if same == n else f"DIVERGES at step {same}: {mine[same]} vs {ref[same]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
