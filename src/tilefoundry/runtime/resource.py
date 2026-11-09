"""``Resource`` — checkpoint access surface.

``Resource`` — checkpoint access surface: load a tensor (or group)
by name, scope to a child namespace. ``DictResource`` is an in-memory/test
double; ``SafetensorsResource`` reads a safetensors checkpoint
directory. See [runtime §1.8](docs/spec/runtime.md#18-runtimeresourcepy).
"""
from __future__ import annotations

import dataclasses
import json
import weakref
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Union

from tilefoundry import utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core, types

safetensors = utils.LazyModule("safetensors")


@dataclasses.dataclass(frozen=True)
class Absolute:
    """An absolute raw checkpoint key, not relative to the current subtree."""

    name: str


Convert = Callable[["utils.torch.Tensor"], "utils.torch.Tensor"]


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


def resolved_device(device: str) -> str:
    """Pin a bare accelerator name to the current device before opening a shard.

    Safetensors reads a bare device name as index 0, while torch uses the device
    currently selected by the process. On multiple devices those can differ, so
    tensors loaded here cannot be compared with tensors computed elsewhere.
    Resolve the name to torch's current index; leave an explicit index alone.
    """
    if device == "cpu" or ":" in device:
        return device
    module = utils.torch.get_device_module(device)
    if not module.is_available():
        return device
    return f"{device}:{module.current_device()}"


def alias_lookup(alias: AliasMap, prefix: str, name: str) -> "AliasValue | None":
    qualified = f"{prefix}{name}"
    if qualified in alias:
        return alias[qualified]
    if name in alias:
        return alias[name]
    return None


def resolve_key(
    alias: AliasMap, prefix: str, name: str,
) -> "tuple[str | tuple[str, ...], Convert | None]":
    hit = alias_lookup(alias, prefix, name)
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


def resolve_segment(alias: AliasMap, prefix: str, seg: str) -> str:
    hit = alias_lookup(alias, prefix, seg)
    if hit is None:
        return seg
    if isinstance(hit, Preprocessed):
        raise TypeError(
            f"Resource.subtree: segment {seg!r} resolves to {hit!r}; a subtree "
            "segment is a path, not a tensor"
        )
    if isinstance(hit, tuple):
        raise TypeError(
            f"Resource.subtree: segment {seg!r} resolves to a "
            f"tuple-valued alias {hit!r}; a subtree segment must resolve to "
            f"one name"
        )
    if isinstance(hit, Absolute):
        raise TypeError(
            f"Resource.subtree: segment {seg!r} resolves to an absolute "
            f"alias {hit!r}; a subtree segment must resolve to one relative "
            f"name"
        )
    return hit


def reject_group(where: str, name: str, resolved: "str | tuple[str, ...]") -> str:
    if isinstance(resolved, tuple):
        raise TypeError(
            f"{where}: {name!r} resolves to a tuple-valued alias "
            f"{resolved!r} ({len(resolved)} raw name(s)); use load_group "
            f"instead"
        )
    return resolved


class Resource(Protocol):
    """Checkpoint access surface: load one tensor (or group) by name, scope to a subtree.

    Checkpoint access surface: load one tensor (or group) by name, scope
    to a subtree.
    """

    def load(self, name: str) -> utils.torch.Tensor: ...

    def load_group(self, name: str) -> "tuple[utils.torch.Tensor, ...] | None": ...

    def subtree(self, seg: str) -> "Resource": ...


class DictResource:
    """dict-backed ``Resource`` — in-memory / test double.

    *data* is a flat, dot-prefixed dict (``{"layer0.attention.w": tensor}``)
    shared by every scoped view; ``subtree`` only extends the prefix.
    """

    def __init__(
        self, data: Mapping[str, utils.torch.Tensor], prefix: str = "",
        alias: AliasMap | None = None,
    ) -> None:
        self.data = data
        self.prefix = prefix
        self.alias = alias or {}

    def load(self, name: str) -> utils.torch.Tensor:
        resolved, read = resolve_key(self.alias, self.prefix, name)
        raw_key = reject_group(
            "DictResource.load", name, resolved
        )
        try:
            value = self.data[raw_key]
        except KeyError:
            raise KeyError(
                f"DictResource: no tensor named {name!r} (raw key {raw_key!r})"
            ) from None
        return value if read is None else read(value)

    def load_group(self, name: str) -> "tuple[utils.torch.Tensor, ...] | None":
        resolved, _ = resolve_key(self.alias, self.prefix, name)
        if not isinstance(resolved, tuple):
            return None
        out = []
        for raw in resolved:
            try:
                out.append(self.data[raw])
            except KeyError:
                raise KeyError(
                    f"DictResource: no tensor named {name!r} (raw key {raw!r})"
                ) from None
        return tuple(out)

    def subtree(self, seg: str) -> "DictResource":
        resolved = resolve_segment(self.alias, self.prefix, seg)
        return DictResource(self.data, f"{self.prefix}{resolved}.", alias=self.alias)


def unsharded_map(directory: Path) -> dict[str, str]:
    """The one shard's own key list, for a directory that ships no index."""
    shard = "model.safetensors"
    path = directory / shard
    if not path.is_file():
        raise FileNotFoundError(
            f"SafetensorsResource: {directory} holds neither "
            f"model.safetensors.index.json nor {shard}"
        )
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict.fromkeys(handle.keys(), shard)


class SafetensorsResource:
    """safetensors-directory-backed ``Resource``.

    Reads a safetensors checkpoint directory via mmap'd ``safe_open``: either
    N shards plus ``model.safetensors.index.json``, or a single unsharded
    ``model.safetensors`` with no index. One shard handle is opened at most once
    and reused across ``subtree`` views. The tensor cache is also shared across
    those views. Reads are idempotent, so that cache is weak: overlapping users
    hit it, and an entry disappears after the last user releases its tensor.
    """

    def __init__(
        self, ckpt_dir: str, prefix: str = "", *, device: str,
        alias: AliasMap | None = None,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.device = device
        self.alias = alias or {}
        self.open_shards: dict[tuple[str, str], Any] = {}
        self.shard_of_key: dict[str, str] | None = None
        self.read_tensors: weakref.WeakValueDictionary[
            tuple[str, str], utils.torch.Tensor
        ] = weakref.WeakValueDictionary()

    def share_cache_of(self, parent: "SafetensorsResource") -> None:
        """Use *parent*'s index, shard handles and tensor cache."""
        self.shard_of_key = parent.shard_of_key
        self.open_shards = parent.open_shards
        self.read_tensors = parent.read_tensors

    def index(self) -> dict[str, str]:
        if self.shard_of_key is None:
            directory = Path(self.ckpt_dir)
            index_path = directory / "model.safetensors.index.json"
            if index_path.is_file():
                with open(index_path, encoding="utf-8") as fh:
                    self.shard_of_key = dict(json.load(fh)["weight_map"])
            else:
                self.shard_of_key = unsharded_map(directory)
        return self.shard_of_key

    def read_one(self, raw_key: str) -> utils.torch.Tensor:
        """One tensor, read onto the device this process has selected right now.

        A handle carries the device it was opened with, and the same key read
        from two cards is two tensors, so both the handle and the tensor are
        remembered under the resolved device beside the name. Keying either by
        name alone would hand a reader on one card what another card holds.
        """
        device = resolved_device(self.device)
        hit = self.read_tensors.get((raw_key, device))
        if hit is not None:
            return hit
        shard = self.index()[raw_key]
        handle = self.open_shards.get((shard, device))
        if handle is None:
            handle = safetensors.safe_open(
                str(Path(self.ckpt_dir) / shard), framework="pt", device=device,
            )
            self.open_shards[(shard, device)] = handle
        value = handle.get_tensor(raw_key)
        self.read_tensors[(raw_key, device)] = value
        return value

    def load(self, name: str) -> utils.torch.Tensor:
        resolved, read = resolve_key(self.alias, self.prefix, name)
        raw_key = reject_group(
            "SafetensorsResource.load", name, resolved
        )
        try:
            value = self.read_one(raw_key)
        except KeyError:
            raise KeyError(
                f"SafetensorsResource: no tensor named {name!r} (raw key {raw_key!r})"
            ) from None
        return value if read is None else read(value)

    def load_group(self, name: str) -> "tuple[utils.torch.Tensor, ...] | None":
        resolved, _ = resolve_key(self.alias, self.prefix, name)
        if not isinstance(resolved, tuple):
            return None
        out = []
        for raw in resolved:
            try:
                out.append(self.read_one(raw))
            except KeyError:
                raise KeyError(
                    f"SafetensorsResource: no tensor named {name!r} (raw key {raw!r})"
                ) from None
        return tuple(out)

    def subtree(self, seg: str) -> "SafetensorsResource":
        resolved = resolve_segment(self.alias, self.prefix, seg)
        child = SafetensorsResource(
            self.ckpt_dir, f"{self.prefix}{resolved}.", device=self.device,
            alias=self.alias,
        )
        child.share_cache_of(self)
        return child


class DrawnResource:
    """Each weight drawn the first time it is asked for, from its declared type.

    Drawn rather than pre-built so a leaf never materialises the tensors its
    siblings declare, and cached so the two sides of a comparison are handed
    the same draw rather than two draws of the same shape.
    """

    def __init__(
        self,
        module: core.Module | core.ModuleList,
        generator: utils.torch.Generator,
        device: str,
        drawn: dict[str, utils.torch.Tensor] | None = None,
        prefix: str = "",
    ) -> None:
        self.module, self.generator = module, generator
        self.device, self.prefix = device, prefix
        self.drawn = {} if drawn is None else drawn

    def load(self, name: str) -> utils.torch.Tensor:
        """Draw a declared weight on first use and retain it strongly."""
        key = f"{self.prefix}{name}"
        if key not in self.drawn:
            declared: types.TensorType = self.module.weights[name]
            self.drawn[key] = draw_tensor(declared, self.generator, self.device)
        return self.drawn[key]

    def load_group(self, name: str) -> None:
        """Drawn tensors have no one-to-many alias map."""
        return None

    def subtree(self, seg: str) -> "DrawnResource":
        if isinstance(self.module, core.ModuleList):
            try:
                index = int(seg)
            except ValueError:
                raise KeyError(seg) from None
            child = self.module.elements[index]
        else:
            slot = self.module.children[seg]
            child = slot if slot.indexed else slot.elements[0]
        return DrawnResource(
            child, self.generator, self.device, drawn=self.drawn,
            prefix=f"{self.prefix}{seg}.",
        )


def extents(type_) -> tuple[int, ...]:
    """Return concrete extents, refusing a shape that still contains a range."""
    shape = []
    for extent in type_.shape:
        if not isinstance(extent, int):
            raise ValueError(
                f"shape {type_.shape} still states {extent} as a range; bind it with --dim"
            )
        shape.append(extent)
    return tuple(shape)


def draw_tensor(declared: types.TensorType, generator: utils.torch.Generator, device: str) -> utils.torch.Tensor:
    """Draw one tensor of *dtype* from *generator*."""
    shape = extents(declared)
    dtype = declared.dtype
    torch_dtype = evaluator.to_torch_dtype(dtype)
    if torch_dtype.is_floating_point:
        return utils.torch.randn(shape, generator=generator, device=device).to(torch_dtype)
    if torch_dtype == utils.torch.bool:
        return utils.torch.randint(0, 2, shape, generator=generator, device=device).to(torch_dtype)
    return utils.torch.randint(0, 8, shape, generator=generator, device=device).to(torch_dtype)


__all__ = [
    "Absolute",
    "DictResource",
    "DrawnResource",
    "Preprocessed",
    "Resource",
    "SafetensorsResource",
    "draw_tensor",
    "resolved_device",
]
