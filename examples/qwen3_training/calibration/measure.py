"""Measure the kernels that have a ready-made implementation, on one GPU, in BF16.

Run in the FA venv (``requirements-fa.txt``), not the project's: this file imports
torch and FlashAttention 4, never tilefoundry. It writes ``compute.json``, which
``export.py`` reads to give a kernel node its time (D59, D62, D69):

- ``FlashAttention``: FA4, causal GQA, one sequence; with ``cp > 1`` each cp position
  attends its own queries to the keys before them, so the key length grows with the
  position.
- the GEMMs of AttnIn, AttnOut, Router, Experts, MoE, Head and NewtonSchulz: cuBLAS
  through ``torch.matmul`` (batched for the experts and the stacked layers). A
  backward is two GEMMs of each forward one; a kernel the stage recomputes (Head) runs
  its forward GEMMs again in the backward.

The shapes are one rank's, from ``configs.py``. Times are medians of CUDA-event
timings, in nanoseconds.

    CUDA_VISIBLE_DEVICES=7 python -m examples.qwen3_training.calibration.measure --out compute.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import flash_attn
import torch
from flash_attn.cute import flash_attn_func

from ..configs import CONFIGS, FULL, Config, Size

DTYPE = torch.bfloat16
WARMUP, REPEAT = 3, 10


def timed(run) -> int:
    """The median time of *run* on the GPU, in nanoseconds."""
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEAT):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return int(statistics.median(samples) * 1e6)


def gemm_ns(batch: int, m: int, n: int, k: int) -> int:
    """One (batched) GEMM ``[batch, m, k] @ [batch, k, n]``."""
    a = torch.randn(batch, m, k, device="cuda", dtype=DTYPE)
    b = torch.randn(batch, k, n, device="cuda", dtype=DTYPE)
    ns = timed(lambda: torch.matmul(a, b))
    del a, b
    return ns


def dims(config: Config, size: Size) -> dict:
    tokens = size.tokens // config.cp
    capacity = tokens * size.top_k // size.experts
    return dict(
        N=tokens,
        C=size.hidden,
        H=size.q_heads,
        KV=size.kv_heads,
        D=size.head_dim,
        E=size.experts,
        F=size.expert_hidden,
        V=size.vocab,
        L=size.layers // config.pp,
        EL=size.experts // config.ep,
        ROWS=config.ep * capacity,
    )


def gemms(config: Config, size: Size) -> dict[str, list[tuple[int, int, int, int]]]:
    """Each kernel's forward GEMMs on one rank, as ``(batch, m, n, k)``."""
    d = dims(config, size)
    N, C, H, KV, D, E, F, V, L, EL, ROWS = (
        d[k] for k in ("N", "C", "H", "KV", "D", "E", "F", "V", "L", "EL", "ROWS")
    )
    experts = [(EL, ROWS, F, C), (EL, ROWS, F, C), (EL, ROWS, C, F)]
    router = [(1, N, E, C)]
    table = {
        "AttnIn": [(1, N, H * D, C), (1, N, KV * D, C), (1, N, KV * D, C)],
        "AttnOut": [(1, N, C, H * D)],
        "Router": router,
        "Experts": experts,
        "MoE": router + experts,
        "Head": [(1, N, V, C)],
    }
    for name, batch, rows, cols in (
        ("Wq", L, H * D, C),
        ("Wk", L, KV * D, C),
        ("Wo", L, C, H * D),
        ("Wg", L * E, F, C),
        ("Wd", L * E, C, F),
    ):
        small, large = min(rows, cols), max(rows, cols)
        # Gram of the short side, its square, and the product back to the matrix; five times.
        table[f"NewtonSchulz{name}"] = [
            (batch, small, small, large),
            (batch, small, small, small),
            (batch, large, small, small),
        ] * 5
    return table


def backward(forward: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """The two GEMMs of each forward one: the input's gradient and the weight's."""
    out = []
    for batch, m, n, k in forward:
        out += [(batch, m, k, n), (batch, n, k, m)]
    return out


def attention_ns(config: Config, size: Size, position: int) -> tuple[int, int]:
    """FA4 forward and backward for the queries of cp position *position*."""
    d = dims(config, size)
    keys = (position + 1) * d["N"]
    q = torch.randn(1, d["N"], d["H"], d["D"], device="cuda", dtype=DTYPE, requires_grad=True)
    k = torch.randn(1, keys, d["KV"], d["D"], device="cuda", dtype=DTYPE, requires_grad=True)
    v = torch.randn(1, keys, d["KV"], d["D"], device="cuda", dtype=DTYPE, requires_grad=True)
    forward = timed(lambda: flash_attn_func(q, k, v, causal=True))
    out = flash_attn_func(q, k, v, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    grad = torch.randn_like(out)
    backward_ns = timed(lambda: torch.autograd.grad(out, (q, k, v), grad, retain_graph=True))
    return forward, backward_ns


def measure(size: Size) -> dict:
    entries: dict[str, dict[str, int]] = {}
    cache: dict[tuple, int] = {}

    def total(shapes):
        ns = 0
        for shape in shapes:
            if shape not in cache:
                cache[shape] = gemm_ns(*shape)
            ns += cache[shape]
        return ns

    for config in CONFIGS.values():
        row = {}
        for kernel, forward in gemms(config, size).items():
            row[f"{kernel}/f"] = total(forward)
            if not kernel.startswith("NewtonSchulz"):
                recomputed = forward if kernel == "Head" else []
                row[f"{kernel}/b"] = total(recomputed + backward(forward))
        for position in range(config.cp):
            forward_ns, backward_ns = attention_ns(config, size, position)
            row[f"FlashAttention/f/cp{position}"] = forward_ns
            row[f"FlashAttention/b/cp{position}"] = backward_ns
        entries[config.name] = row
        print(config.name, json.dumps(row), flush=True)
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    entries = measure(FULL)
    record = {
        "source": "measured",
        "device": torch.cuda.get_device_name(),
        "dtype": "bf16",
        "torch": torch.__version__,
        "flash_attn": getattr(flash_attn, "__version__", "4.0.0b39"),
        "timing": f"median of {REPEAT} CUDA-event timings after {WARMUP} warmups, nanoseconds",
        "entries": entries,
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
