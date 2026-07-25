"""``RuntimeResource`` — checkpoint access surface a ``RuntimeModule`` needs:
load one tensor (or a one-to-many group) by its own (unprefixed) name, and
scope down to a child namespace. ``DictResource`` is an in-memory / test
double; ``SafetensorsResource`` reads a repacked (N-shard + ``index.json``)
safetensors checkpoint directory. Both take an optional ``alias`` table
(canonical name -> raw name or tuple of raw names), resolved by the shared
``_alias_lookup`` helper — see docs/spec/runtime.md §1.4.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Union

import torch

# A canonical name maps to one raw name, or (one-to-many, e.g. per-expert
# tensors) a tuple of raw names, in declared order.
AliasValue = Union[str, "tuple[str, ...]"]
AliasMap = Mapping[str, AliasValue]


def _alias_lookup(alias: AliasMap, prefix: str, name: str) -> "AliasValue | None":
    """Path-qualified alias entry (``f"{prefix}{name}"``), else a bare-name
    entry, else ``None`` (no alias applies). A value renames one path segment
    or leaf *within* the current scope; the caller joins it onto the
    accumulated prefix, so one bare entry (``"gamma_kv": "kv_norm.weight"``)
    serves every layer."""
    qualified = f"{prefix}{name}"
    if qualified in alias:
        return alias[qualified]
    if name in alias:
        return alias[name]
    return None


def _resolve_key(alias: AliasMap, prefix: str, name: str) -> AliasValue:
    """Resolve a leaf tensor *name* to its raw key(s) for ``load`` /
    ``load_group``, joined onto *prefix* whether or not an alias hit — a leaf
    entry renames the leaf inside the current scope, exactly as a segment
    entry renames one segment, so one entry serves every layer. An unaliased
    name is the plain prefix-join."""
    hit = _alias_lookup(alias, prefix, name)
    if hit is None:
        return f"{prefix}{name}"
    if isinstance(hit, tuple):
        return tuple(f"{prefix}{one}" for one in hit)
    return f"{prefix}{hit}"


def _resolve_segment(alias: AliasMap, prefix: str, seg: str) -> str:
    """Resolve a ``subtree`` segment *seg*. The identity fallback is *seg*
    itself, unqualified — like ``_resolve_key``'s hit, the caller (``subtree``)
    prepends its own accumulated prefix uniformly, so a value here must never
    itself carry ``prefix``."""
    hit = _alias_lookup(alias, prefix, seg)
    if hit is None:
        return seg
    if isinstance(hit, tuple):
        raise TypeError(
            f"RuntimeResource.subtree: segment {seg!r} resolves to a "
            f"tuple-valued alias {hit!r}; a subtree segment must resolve to "
            f"one name"
        )
    return hit


def _reject_group(where: str, name: str, resolved: AliasValue) -> str:
    """Guard a single-tensor call site (``load``) against a tuple-valued
    (one-to-many) alias resolution."""
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
    """dict-backed ``RuntimeResource`` — test / in-memory fallback.

    *data* is a flat, dot-prefixed ``{"layer0.attention.w": tensor, ...}``
    dict shared by every scoped view; ``subtree`` only extends the prefix
    each ``load`` name is joined onto (through *alias*, carried down to
    every child view).
    """

    def __init__(
        self, data: Mapping[str, torch.Tensor], prefix: str = "",
        alias: AliasMap | None = None,
    ) -> None:
        self._data = data
        self._prefix = prefix
        self._alias = alias or {}

    def load(self, name: str) -> torch.Tensor:
        resolved = _reject_group(
            "DictResource.load", name, _resolve_key(self._alias, self._prefix, name)
        )
        try:
            return self._data[resolved]
        except KeyError:
            raise KeyError(
                f"DictResource: no tensor named {name!r} (raw key {resolved!r})"
            ) from None

    def load_group(self, name: str) -> "tuple[torch.Tensor, ...] | None":
        resolved = _resolve_key(self._alias, self._prefix, name)
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

    Mirrors the on-disk convention a repacked HF checkpoint uses (N shards +
    ``model.safetensors.index.json``): each name is looked up in the index
    for its shard file, and only that one tensor is read — via
    ``safetensors.safe_open`` (mmap'd, lazy-per-tensor) — straight onto
    *device*. One shard handle is opened at most once and reused across
    names, including across ``subtree`` views.
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
            import json  # noqa: PLC0415 -- stdlib, cheap, only needed here
            from pathlib import Path  # noqa: PLC0415

            index_path = Path(self._ckpt_dir) / "model.safetensors.index.json"
            with open(index_path, encoding="utf-8") as fh:
                self._weight_map = dict(json.load(fh)["weight_map"])
        return self._weight_map

    def _read_one(self, raw_key: str) -> torch.Tensor:
        from pathlib import Path  # noqa: PLC0415

        from safetensors import safe_open  # noqa: PLC0415 -- optional runtime dep

        shard = self._index()[raw_key]  # KeyError propagates the raw key
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(Path(self._ckpt_dir) / shard), framework="pt", device=self._device)
            self._handles[shard] = handle
        return handle.get_tensor(raw_key)

    def load(self, name: str) -> torch.Tensor:
        resolved = _reject_group(
            "SafetensorsResource.load", name, _resolve_key(self._alias, self._prefix, name)
        )
        try:
            return self._read_one(resolved)
        except KeyError:
            raise KeyError(
                f"SafetensorsResource: no tensor named {name!r} (raw key {resolved!r})"
            ) from None

    def load_group(self, name: str) -> "tuple[torch.Tensor, ...] | None":
        resolved = _resolve_key(self._alias, self._prefix, name)
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
            self._ckpt_dir, f"{self._prefix}{resolved}.", self._device, alias=self._alias
        )
        child._weight_map = self._weight_map
        child._handles = self._handles
        return child


__all__ = ["DictResource", "RuntimeResource", "SafetensorsResource"]
