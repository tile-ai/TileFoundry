"""``RuntimeResource`` — checkpoint access surface a ``RuntimeModule`` needs:
load one tensor by its own (unprefixed) name, and scope down to a child
namespace. ``DictResource`` is an in-memory / test double; ``SafetensorsResource``
reads a repacked (N-shard + ``index.json``) safetensors checkpoint directory.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol

import torch


class RuntimeResource(Protocol):
    """Checkpoint access surface: load one tensor by name, scope to a subtree."""

    def load(self, name: str) -> torch.Tensor: ...

    def subtree(self, prefix: str) -> "RuntimeResource": ...


class DictResource:
    """dict-backed ``RuntimeResource`` — test / in-memory fallback.

    *data* is a flat, dot-prefixed ``{"layer0.attention.w": tensor, ...}``
    dict shared by every scoped view; ``subtree`` only extends the prefix
    each ``load`` name is joined onto.
    """

    def __init__(self, data: Mapping[str, torch.Tensor], prefix: str = "") -> None:
        self._data = data
        self._prefix = prefix

    def load(self, name: str) -> torch.Tensor:
        key = self._prefix + name
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"DictResource: no tensor named {key!r}") from None

    def subtree(self, prefix: str) -> "DictResource":
        return DictResource(self._data, f"{self._prefix}{prefix}.")


class SafetensorsResource:
    """safetensors-directory-backed ``RuntimeResource``.

    Mirrors the on-disk convention a repacked HF checkpoint uses (N shards +
    ``model.safetensors.index.json``): each name is looked up in the index
    for its shard file, and only that one tensor is read — via
    ``safetensors.safe_open`` (mmap'd, lazy-per-tensor) — straight onto
    *device*. One shard handle is opened at most once and reused across
    names, including across ``subtree`` views.
    """

    def __init__(self, ckpt_dir: str, prefix: str = "", device: str = "cuda") -> None:
        self._ckpt_dir = ckpt_dir
        self._prefix = prefix
        self._device = device
        self._handles: dict[str, Any] = {}
        self._weight_map: dict[str, str] | None = None

    def _index(self) -> dict[str, str]:
        if self._weight_map is None:
            import json  # noqa: PLC0415 -- stdlib, cheap, only needed here
            from pathlib import Path  # noqa: PLC0415

            index_path = Path(self._ckpt_dir) / "model.safetensors.index.json"
            with open(index_path, encoding="utf-8") as fh:
                self._weight_map = dict(json.load(fh)["weight_map"])
        return self._weight_map

    def _shard_for(self, key: str) -> str:
        try:
            return self._index()[key]
        except KeyError:
            raise KeyError(f"SafetensorsResource: no tensor named {key!r}") from None

    def load(self, name: str) -> torch.Tensor:
        from pathlib import Path  # noqa: PLC0415

        from safetensors import safe_open  # noqa: PLC0415 -- optional runtime dep

        key = self._prefix + name
        shard = self._shard_for(key)
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(Path(self._ckpt_dir) / shard), framework="pt", device=self._device)
            self._handles[shard] = handle
        return handle.get_tensor(key)

    def subtree(self, prefix: str) -> "SafetensorsResource":
        child = SafetensorsResource(self._ckpt_dir, f"{self._prefix}{prefix}.", self._device)
        child._weight_map = self._weight_map
        child._handles = self._handles
        return child


__all__ = ["DictResource", "RuntimeResource", "SafetensorsResource"]
