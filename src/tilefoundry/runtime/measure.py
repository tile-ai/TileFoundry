"""``check`` / ``bench`` — numerical-parity and latency measurement helpers
over any two (or one) plain callables (e.g. a ``RuntimeModule`` bound method,
a raw torch function). ``check`` compares a bare tensor or an arbitrarily
nested tuple of tensors."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from tilefoundry.target.base import Device


def _torch_device_str(device: "Device | None") -> str:
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


def _flatten(x, path: str = "output") -> list[tuple[str, torch.Tensor]]:
    """Flatten a tensor or nested tuple-of-tensors into ``[(path, tensor), ...]``;
    the path list doubles as a structural signature for comparing outputs."""
    if isinstance(x, torch.Tensor):
        return [(path, x)]
    if isinstance(x, tuple):
        leaves: list[tuple[str, torch.Tensor]] = []
        for i, item in enumerate(x):
            leaves.extend(_flatten(item, f"{path}[{i}]"))
        return leaves
    raise TypeError(f"check: {path} must be a torch.Tensor or tuple thereof, got {type(x).__name__}")


def check(
    candidate: Callable, reference: Callable, inputs: tuple, gate: Gate = Gate(),
) -> Report:
    """Run *candidate* and *reference* on the same *inputs*. Each may return a
    bare tensor or an arbitrarily nested tuple of tensors; flattens both,
    requires identical structure/shape/dtype, and gates on the worst
    per-element rel_l2 / cosine (a single tensor is the one-element case)."""
    current = _flatten(candidate(*inputs))
    expected = _flatten(reference(*inputs))
    current_paths = [p for p, _ in current]
    expected_paths = [p for p, _ in expected]
    if current_paths != expected_paths:
        raise ValueError(
            f"check: candidate/reference output structures differ: "
            f"{current_paths} vs {expected_paths}"
        )
    # same path at the same index (just verified) => positional zip is safe
    pairs = [(c, r) for (_, c), (_, r) in zip(current, expected)]
    for path, (c, r) in zip(current_paths, pairs):
        if c.shape != r.shape or c.dtype != r.dtype:
            raise ValueError(
                f"check: {path} shape/dtype mismatch: "
                f"candidate {tuple(c.shape)}/{c.dtype} vs reference {tuple(r.shape)}/{r.dtype}"
            )
    if not pairs:
        return Report(metrics={"rel_l2": 0.0, "cosine": 1.0}, passed=True)
    rel_l2 = max(_rel_l2(c, r) for c, r in pairs)
    cosine = min(_cosine(c, r) for c, r in pairs)
    passed = rel_l2 <= gate.rel_l2_max and cosine >= gate.cosine_min
    return Report(metrics={"rel_l2": rel_l2, "cosine": cosine}, passed=passed)


def bench(fn: Callable, inputs: tuple, iters: int = 100, *, device: Device) -> Report:
    """Mean per-call latency of *fn* over *iters* calls, after a few untimed
    warmup calls. A CUDA device times with ``torch.cuda.Event``\\ s; any other
    device times with ``time.perf_counter()``. ``passed`` is always ``None`` —
    no gate."""
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
