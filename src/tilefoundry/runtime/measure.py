"""``check`` / ``bench`` — numerical-parity and latency measurement helpers
over any two (or one) plain callables — a ``RuntimeModule`` bound method, a
raw torch function, an evaluator closure, ... Neither helper depends on
``RuntimeModule`` or the IR: both operate on whatever *inputs* the given
callable(s) accept.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from tilefoundry.target.base import Device


def _torch_device_str(device: "Device | None") -> str:
    """Map a ``Device`` to a torch device string for timing dispatch: a device
    whose class lives under ``tilefoundry.target.cuda`` is ``"cuda"``, every
    other concrete ``Device`` is ``"cpu"``."""
    if device is None:
        raise ValueError("bench: a device is required, got None")
    if type(device).__module__.startswith("tilefoundry.target.cuda"):
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class Gate:
    """Pass/fail thresholds for ``check()``."""
    rel_l2_max: float = 1e-3
    cosine_min: float = 0.999


@dataclass(frozen=True)
class Report:
    """Result of ``check()`` / ``bench()`` — a named metrics bag plus an
    optional pass/fail verdict (``bench()`` leaves it ``None``)."""
    metrics: Mapping[str, float]
    passed: bool | None = None


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def check(
    candidate: Callable, reference: Callable, inputs: tuple, gate: Gate = Gate(),
) -> Report:
    """Run *candidate* and *reference* on the same *inputs*; gate their
    rel_l2 / cosine similarity."""
    current = candidate(*inputs)
    expected = reference(*inputs)
    rel_l2 = _rel_l2(current, expected)
    cosine = _cosine(current, expected)
    passed = rel_l2 <= gate.rel_l2_max and cosine >= gate.cosine_min
    return Report(metrics={"rel_l2": rel_l2, "cosine": cosine}, passed=passed)


def bench(fn: Callable, inputs: tuple, iters: int = 100, *, device: Device) -> Report:
    """Mean per-call latency of *fn* over *iters* calls, after a few untimed
    warmup calls. A ``device`` mapping to ``"cuda"`` (§ ``_torch_device_str``)
    times with ``torch.cuda.Event``\\ s; any other device times with
    ``time.perf_counter()``. ``passed`` is always ``None`` — no gate."""
    warmups = min(3, iters)
    if _torch_device_str(device) == "cuda":
        for _ in range(warmups):
            fn(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        mean_ms = start.elapsed_time(end) / iters
    else:
        for _ in range(warmups):
            fn(*inputs)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*inputs)
        mean_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return Report(metrics={"mean_ms": mean_ms, "iters": float(iters)})


__all__ = ["Gate", "Report", "bench", "check"]
