"""``RuntimeResource`` — checkpoint access surface: load a tensor (or group)
by name, scope to a child namespace. ``DictResource`` is an in-memory/test
double; ``SafetensorsResource`` reads a safetensors checkpoint
directory. See docs/spec/runtime.md §1.5.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Protocol, Union

import torch


@dataclasses.dataclass(frozen=True)
class Absolute:
    """An absolute raw checkpoint key, not relative to the current subtree."""

    name: str


Convert = Callable[[torch.Tensor], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class Preprocessed:
    """One raw checkpoint name and the preprocessing its tensor needs."""

    name: "str | Absolute"
    read: Convert

    def __post_init__(self) -> None:
        if isinstance(self.name, tuple):
            raise TypeError(
                f"Preprocessed: {self.name!r} names a one-to-many group, whose "
                "value is a function of several raw tensors; that is a weight "
                "converter, run by prepare, not preprocessing on one tensor"
            )


# One raw name, or (one-to-many, e.g. per-expert) a tuple in declared order.
AliasValue = Union[str, "tuple[str, ...]", Absolute, Preprocessed]
AliasMap = Mapping[str, AliasValue]


def _resolved_device(device: str) -> str:
    """A device string with a bare ``"cuda"`` pinned to the current device.

    ``safetensors`` reads the string itself and takes bare ``"cuda"`` as index 0,
    while every torch spelling of it -- ``.cuda()``, ``device="cuda"``, a
    ``torch.device`` context -- means whichever device the process has selected.
    On a machine with one card those agree; on a machine with several, the same word
    lands on two of them and a tensor loaded here cannot be compared against one
    computed anywhere else. Resolving it once here makes the word mean what torch
    means by it. An index the caller wrote is left alone.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return device
    return f"cuda:{torch.cuda.current_device()}"


def _alias_lookup(alias: AliasMap, prefix: str, name: str) -> "AliasValue | None":
    qualified = f"{prefix}{name}"
    if qualified in alias:
        return alias[qualified]
    if name in alias:
        return alias[name]
    return None


def _resolve_key(
    alias: AliasMap, prefix: str, name: str,
) -> "tuple[str | tuple[str, ...], Convert | None]":
    hit = _alias_lookup(alias, prefix, name)
    read = None
    if isinstance(hit, Preprocessed):
        hit, read = hit.name, hit.read
    if hit is None:
        return f"{prefix}{name}", read
    if isinstance(hit, Absolute):
        return hit.name, read
    if isinstance(hit, tuple):
        return tuple(f"{prefix}{one}" for one in hit), read
    return f"{prefix}{hit}", read


def _resolve_segment(alias: AliasMap, prefix: str, seg: str) -> str:
    hit = _alias_lookup(alias, prefix, seg)
    if hit is None:
        return seg
    if isinstance(hit, Preprocessed):
        raise TypeError(
            f"RuntimeResource.subtree: segment {seg!r} resolves to {hit!r}; a subtree "
            "segment is a path, not a tensor"
        )
    if isinstance(hit, tuple):
        raise TypeError(
            f"RuntimeResource.subtree: segment {seg!r} resolves to a "
            f"tuple-valued alias {hit!r}; a subtree segment must resolve to "
            f"one name"
        )
    if isinstance(hit, Absolute):
        raise TypeError(
            f"RuntimeResource.subtree: segment {seg!r} resolves to an absolute "
            f"alias {hit!r}; a subtree segment must resolve to one relative "
            f"name"
        )
    return hit


def _reject_group(where: str, name: str, resolved: "str | tuple[str, ...]") -> str:
    if isinstance(resolved, tuple):
        raise TypeError(
            f"{where}: {name!r} resolves to a tuple-valued alias "
            f"{resolved!r} ({len(resolved)} raw name(s)); use load_group "
            f"instead"
        )
    return resolved


class RuntimeResource(Protocol):
    """Checkpoint access surface: load one tensor (or group) by name, scope
    to a subtree."""

    def load(self, name: str) -> torch.Tensor: ...

    def load_group(self, name: str) -> "tuple[torch.Tensor, ...] | None": ...

    def subtree(self, seg: str) -> "RuntimeResource": ...


class DictResource:
    """dict-backed ``RuntimeResource`` — in-memory / test double.

    *data* is a flat, dot-prefixed dict (``{"layer0.attention.w": tensor}``)
    shared by every scoped view; ``subtree`` only extends the prefix.
    """

    def __init__(
        self, data: Mapping[str, torch.Tensor], prefix: str = "",
        alias: AliasMap | None = None,
    ) -> None:
        self._data = data
        self._prefix = prefix
        self._alias = alias or {}

    def load(self, name: str) -> torch.Tensor:
        resolved, read = _resolve_key(self._alias, self._prefix, name)
        raw_key = _reject_group(
            "DictResource.load", name, resolved
        )
        try:
            value = self._data[raw_key]
        except KeyError:
            raise KeyError(
                f"DictResource: no tensor named {name!r} (raw key {raw_key!r})"
            ) from None
        return value if read is None else read(value)

    def load_group(self, name: str) -> "tuple[torch.Tensor, ...] | None":
        resolved, _ = _resolve_key(self._alias, self._prefix, name)
        if not isinstance(resolved, tuple):
            return None
        out = []
        for raw in resolved:
            try:
                out.append(self._data[raw])
            except KeyError:
                raise KeyError(
                    f"DictResource: no tensor named {name!r} (raw key {raw!r})"
                ) from None
        return tuple(out)

    def subtree(self, seg: str) -> "DictResource":
        resolved = _resolve_segment(self._alias, self._prefix, seg)
        return DictResource(self._data, f"{self._prefix}{resolved}.", alias=self._alias)


class SafetensorsResource:
    """safetensors-directory-backed ``RuntimeResource``.

    Reads a safetensors checkpoint directory via mmap'd ``safe_open``: either
    N shards plus ``model.safetensors.index.json``, or a single unsharded
    ``model.safetensors`` with no index. One shard handle is opened at most once
    and reused across ``subtree`` views.
    """

    def __init__(
        self, ckpt_dir: str, prefix: str = "", device: str = "cuda",
        alias: AliasMap | None = None,
    ) -> None:
        self._ckpt_dir = ckpt_dir
        self._prefix = prefix
        self._device = device
        self._alias = alias or {}
        self._handles: dict[str, Any] = {}
        self._weight_map: dict[str, str] | None = None

    def _index(self) -> dict[str, str]:
        if self._weight_map is None:
            import json  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            directory = Path(self._ckpt_dir)
            index_path = directory / "model.safetensors.index.json"
            if index_path.is_file():
                with open(index_path, encoding="utf-8") as fh:
                    self._weight_map = dict(json.load(fh)["weight_map"])
            else:
                self._weight_map = self._unsharded_map(directory)
        return self._weight_map

    @staticmethod
    def _unsharded_map(directory: "Path") -> dict[str, str]:
        """The one shard's own key list, for a directory that ships no index."""
        from safetensors import safe_open  # noqa: PLC0415

        shard = "model.safetensors"
        path = directory / shard
        if not path.is_file():
            raise FileNotFoundError(
                f"SafetensorsResource: {directory} holds neither "
                f"model.safetensors.index.json nor {shard}"
            )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict.fromkeys(handle.keys(), shard)

    def _read_one(self, raw_key: str) -> torch.Tensor:
        from pathlib import Path  # noqa: PLC0415

        from safetensors import safe_open  # noqa: PLC0415

        shard = self._index()[raw_key]  # KeyError propagates the raw key
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(
                str(Path(self._ckpt_dir) / shard),
                framework="pt",
                device=_resolved_device(self._device),
            )
            self._handles[shard] = handle
        return handle.get_tensor(raw_key)

    def load(self, name: str) -> torch.Tensor:
        resolved, read = _resolve_key(self._alias, self._prefix, name)
        raw_key = _reject_group(
            "SafetensorsResource.load", name, resolved
        )
        try:
            value = self._read_one(raw_key)
        except KeyError:
            raise KeyError(
                f"SafetensorsResource: no tensor named {name!r} (raw key {raw_key!r})"
            ) from None
        return value if read is None else read(value)

    def load_group(self, name: str) -> "tuple[torch.Tensor, ...] | None":
        resolved, _ = _resolve_key(self._alias, self._prefix, name)
        if not isinstance(resolved, tuple):
            return None
        out = []
        for raw in resolved:
            try:
                out.append(self._read_one(raw))
            except KeyError:
                raise KeyError(
                    f"SafetensorsResource: no tensor named {name!r} (raw key {raw!r})"
                ) from None
        return tuple(out)

    def subtree(self, seg: str) -> "SafetensorsResource":
        resolved = _resolve_segment(self._alias, self._prefix, seg)
        child = SafetensorsResource(
            self._ckpt_dir, f"{self._prefix}{resolved}.", self._device,
            alias=self._alias,
        )
        child._weight_map = self._weight_map
        child._handles = self._handles
        return child


__all__ = ["Absolute", "DictResource", "Preprocessed", "RuntimeResource", "SafetensorsResource"]
