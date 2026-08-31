"""``RuntimeResource`` — checkpoint access surface.

``RuntimeResource`` — checkpoint access surface: load a tensor (or group)
by name, scope to a child namespace. ``DictResource`` is an in-memory/test
double; ``SafetensorsResource`` reads a safetensors checkpoint
directory. See [runtime §1.5](docs/spec/runtime.md#15-runtimeresource).
"""
from __future__ import annotations

import dataclasses
import weakref
from typing import Any, Callable, Mapping, Protocol, Union

import torch

from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.tensor_type import TensorType


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
    """Checkpoint access surface: load one tensor (or group) by name, scope to a subtree.

    Checkpoint access surface: load one tensor (or group) by name, scope
    to a subtree.
    """

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
    and reused across ``subtree`` views. The tensor cache is also shared across
    those views. Reads are idempotent, so that cache is weak: overlapping users
    hit it, and an entry disappears after the last user releases its tensor.
    """

    def __init__(
        self, ckpt_dir: str, prefix: str = "", device: str = "cuda",
        alias: AliasMap | None = None,
    ) -> None:
        self._ckpt_dir = ckpt_dir
        self._prefix = prefix
        self._device = device
        self._alias = alias or {}
        self._open_shards: dict[str, Any] = {}
        self._shard_of_key: dict[str, str] | None = None
        self._read_tensors: weakref.WeakValueDictionary[str, torch.Tensor] = (
            weakref.WeakValueDictionary()
        )

    def _index(self) -> dict[str, str]:
        if self._shard_of_key is None:
            import json  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            directory = Path(self._ckpt_dir)
            index_path = directory / "model.safetensors.index.json"
            if index_path.is_file():
                with open(index_path, encoding="utf-8") as fh:
                    self._shard_of_key = dict(json.load(fh)["weight_map"])
            else:
                self._shard_of_key = self._unsharded_map(directory)
        return self._shard_of_key

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

        hit = self._read_tensors.get(raw_key)
        if hit is not None:
            return hit
        shard = self._index()[raw_key]
        handle = self._open_shards.get(shard)
        if handle is None:
            handle = safe_open(
                str(Path(self._ckpt_dir) / shard),
                framework="pt",
                device=_resolved_device(self._device),
            )
            self._open_shards[shard] = handle
        value = handle.get_tensor(raw_key)
        self._read_tensors[raw_key] = value
        return value

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
        child._shard_of_key = self._shard_of_key
        child._open_shards = self._open_shards
        child._read_tensors = self._read_tensors
        return child


class DrawnResource:
    """Each weight drawn the first time it is asked for, from its declared type.

    Drawn rather than pre-built so a leaf never materialises the tensors its
    siblings declare, and cached so the two sides of a comparison are handed
    the same draw rather than two draws of the same shape.
    """

    def __init__(
        self,
        module: Module,
        generator: torch.Generator,
        device: str,
        drawn: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
    ) -> None:
        self._module, self._generator = module, generator
        self._device, self._prefix = device, prefix
        self._drawn = {} if drawn is None else drawn

    def load(self, name: str) -> torch.Tensor:
        """Draw a declared weight on first use and retain it strongly."""
        key = f"{self._prefix}{name}"
        if key not in self._drawn:
            declared: TensorType = self._module.weights[name]
            self._drawn[key] = draw_tensor(declared, self._generator, self._device)
        return self._drawn[key]

    def load_group(self, name: str) -> None:
        """Drawn tensors have no one-to-many alias map."""
        return None

    def subtree(self, seg: str) -> "DrawnResource":
        for child in self._module.modules:
            if child.name == seg:
                return DrawnResource(
                    child,
                    self._generator,
                    self._device,
                    drawn=self._drawn,
                    prefix=f"{self._prefix}{seg}.",
                )
        raise KeyError(seg)


def _extents(type_) -> tuple[int, ...]:
    """Return concrete extents, refusing a shape that still contains a range."""
    shape = []
    for extent in type_.shape:
        if not isinstance(extent, int):
            raise ValueError(
                f"shape {type_.shape} still states {extent} as a range; bind it with --dim"
            )
        shape.append(extent)
    return tuple(shape)


def draw_tensor(declared: TensorType, generator: torch.Generator, device: str) -> torch.Tensor:
    """Draw one tensor of *dtype* from *generator*."""
    shape = _extents(declared)
    dtype = declared.dtype
    torch_dtype = to_torch_dtype(dtype)
    if torch_dtype.is_floating_point:
        return torch.randn(shape, generator=generator, device=device).to(torch_dtype)
    if torch_dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=generator, device=device).to(torch_dtype)
    return torch.randint(0, 8, shape, generator=generator, device=device).to(torch_dtype)


__all__ = [
    "Absolute",
    "DictResource",
    "DrawnResource",
    "Preprocessed",
    "RuntimeResource",
    "SafetensorsResource",
    "draw_tensor",
]
