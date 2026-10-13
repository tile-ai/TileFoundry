"""Prefill Qwen3.8-27B on the tilelang runtime twin and print the first token.

    python run.py --prompt "..." [--prompt-tokens 8192]

The prompt is tokenized; with --prompt-tokens it is repeated and cut to
exactly that many tokens. Prompts whose length is not a multiple of 128
(the row tile of the attention and GEMM kernels; the delta rule's chunk, 64,
divides it) are right-padded: every layer is causal, so the rows before the
padding are the unpadded prompt's rows, and the logits are read at the last
real row.

Timing covers one prefill as a CUDA graph replay -- token ids on the device
to the first token on the device: embedding, 64 layers, the closing norm,
the lm_head row, the argmax. It excludes weight loading, kernel
compilation, graph capture and tokenization (all done before, untimed),
and the 2 untimed warmup replays tools/bench.py also makes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PAD = 128


class PinnedResource:
    """A SafetensorsResource view that keeps every tensor it hands out alive.

    The runtime twin reads weights by name on each call; the safetensors
    resource caches them only weakly. A CUDA graph replays raw pointers, so
    the weights it captured must not be freed and re-read.
    """

    def __init__(self, inner, held=None):
        self.inner = inner
        self.held = {} if held is None else held

    def load(self, name):
        key = (id(self.inner), name)
        if key not in self.held:
            self.held[key] = (self.inner, self.inner.load(name))
        return self.held[key][1]

    def load_group(self, name):
        return self.inner.load_group(name)

    def subtree(self, seg):
        return PinnedResource(self.inner.subtree(seg), self.held)


def load_twin(ckpt, device="cuda"):
    import model as M
    import runtime_model as R
    from tilefoundry.runtime import SafetensorsResource

    twin = R.Qwen3_8_27BTwin()
    twin.load(PinnedResource(SafetensorsResource(str(ckpt), device=device, alias=M.hf_alias())))
    return twin


class Prefill:
    """One captured prefill per padded length: static ids in, first token out."""

    def __init__(self, twin, n_pad: int, device="cuda"):
        self.twin = twin
        self.n = n_pad
        self.ids = torch.zeros(n_pad, dtype=torch.int64, device=device)
        self.last = torch.zeros(1, dtype=torch.int64, device=device)
        # Eager run first: compiles every kernel and rearranges the weights.
        self._step()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._step()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=side):
                self.logits, self.token = self._step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

    def _step(self):
        logits = self.twin.forward(self.ids, self.last)
        return logits, torch.argmax(logits[0])

    def __call__(self, ids: torch.Tensor, last: int):
        self.ids.zero_()
        self.ids[: ids.numel()].copy_(ids)
        self.last.fill_(last)
        self.graph.replay()
        return self.logits, self.token


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="the published checkpoint; or set $QWEN38_CKPT")
    args = ap.parse_args(argv)

    import paths
    from transformers import AutoTokenizer

    from prompt import prompt_ids

    ckpt = paths.need("ckpt", args.ckpt)
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    ids = prompt_ids(tok, args.prompt, args.prompt_tokens).cuda()
    n = ids.numel()
    n_pad = -(-n // PAD) * PAD
    twin = load_twin(ckpt)
    pre = Prefill(twin, n_pad)
    for _ in range(2):
        pre(ids, n - 1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    _, token = pre(ids, n - 1)
    first = int(token)
    ms = (time.perf_counter() - t) * 1e3
    print(f"first token: {first} {tok.decode([first])!r}")
    print(f"prefill: {n} tokens (padded to {n_pad}) in {ms:.2f} ms, {n / ms * 1e3:.0f} tok/s")


if __name__ == "__main__":
    main()
