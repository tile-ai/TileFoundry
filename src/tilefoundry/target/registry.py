"""The registry of Target classes by name, and the loading of installed Target packages."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Mapping
from importlib.metadata import Distribution, entry_points
from types import MappingProxyType
from typing import TypeVar

from .target import Target

TargetT = TypeVar("TargetT", bound=Target)


class LoadError(RuntimeError):
    """An installed target package could not be loaded as declared."""


def provider_of(target_type: type[Target]) -> tuple[str, str]:
    """The module and qualified name that define *target_type*."""
    return (target_type.__module__, target_type.__qualname__)


def top_level(distribution: Distribution | None) -> set[str]:
    """Read declared top-level packages, falling back to installed file records."""
    provided = set()
    if distribution is not None:
        declared = distribution.read_text("top_level.txt")
        if declared is not None:
            provided = {name.strip() for name in declared.splitlines() if name.strip()}
        else:
            for path in distribution.files or ():
                if not path.parts:
                    continue
                root = path.parts[0]
                if root.startswith("..") or root.endswith((".dist-info", ".pth")):
                    continue
                if root != "__pycache__":
                    provided.add(root.removesuffix(".py"))
    if not provided:
        name = distribution.name if distribution is not None else "<unknown distribution>"
        raise LoadError(f"cannot determine the top-level packages of {name}")
    return provided


class Catalog:
    """The Target classes registered by name, and whether installed packages are loaded."""

    def __init__(self) -> None:
        self.classes: dict[str, type[Target]] = {}
        self.providers: dict[str, tuple[str, str]] = {}
        self.loaded = False

    def register(self, target_type: type[TargetT]) -> type[TargetT]:
        """Register a concrete Target class under its explicitly declared name."""
        if not isinstance(target_type, type) or not issubclass(target_type, Target):
            raise TypeError("@register_target expects a Target subclass")
        if target_type is Target or inspect.isabstract(target_type):
            raise TypeError("@register_target expects a concrete Target subclass")
        if "name" not in vars(target_type):
            raise ValueError(
                f"@register_target {target_type.__qualname__}: declare a class-level "
                "non-empty name instead of inheriting one"
            )
        name = target_type.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"@register_target {target_type.__qualname__}: name must be a "
                "non-empty string"
            )
        provider = provider_of(target_type)
        existing_provider = self.providers.get(name)
        if existing_provider is not None and existing_provider != provider:
            existing = self.classes[name]
            raise ValueError(
                f"@register_target: name {name!r} is already owned by "
                f"{existing.__module__}.{existing.__qualname__}; cannot register "
                f"{target_type.__module__}.{target_type.__qualname__}"
            )
        if existing_provider != provider:
            self.classes[name] = target_type
            self.providers[name] = provider
        return target_type

    def unregister(self, target_type: type[Target]) -> None:
        """Remove one exact provider class while undoing a persisted module load."""
        name = target_type.name
        if self.classes.get(name) is target_type:
            self.classes.pop(name)
            self.providers.pop(name)

    def snapshot(self) -> tuple[dict[str, type[Target]], dict[str, tuple[str, str]]]:
        """Copy the registered classes and their providers, for a later ``restore``."""
        return dict(self.classes), dict(self.providers)

    def restore(self, saved: tuple[dict[str, type[Target]], dict[str, tuple[str, str]]]) -> None:
        """Return the registered classes and providers to a ``snapshot``."""
        classes, providers = saved
        self.classes.clear()
        self.classes.update(classes)
        self.providers.clear()
        self.providers.update(providers)

    def registered(self) -> Mapping[str, type[Target]]:
        """Return a read-only view of registered Target classes by name."""
        return MappingProxyType(self.classes)

    def load(self) -> Mapping[str, type[Target]]:
        """Import each installed target package once; importing registers its contents."""
        if not self.loaded:
            top_levels = {}
            for ep in entry_points(group="tilefoundry.targets"):
                distribution = ep.dist.name if ep.dist is not None else "<unknown distribution>"
                vendor, _, target = ep.value.partition(".")
                if not vendor.startswith("tilefoundry_") or target != ep.name:
                    raise LoadError(
                        f"{distribution}: entry point {ep.name}={ep.value} "
                        f"must name subpackage tilefoundry_<vendor>.{ep.name}"
                    )
                if ep.dist is None or distribution not in top_levels:
                    top_levels[distribution] = top_level(ep.dist)
                provided = top_levels[distribution]
                if provided != {vendor}:
                    raise LoadError(
                        f"{distribution}: installs {sorted(provided)}; a target distribution "
                        f"installs only {vendor}"
                    )
                if ep.value in sys.modules:
                    continue
                try:
                    importlib.import_module(ep.value)
                except Exception as error:
                    raise LoadError(
                        f"{distribution}: importing {ep.value}: {error}"
                    ) from error
                if ep.name not in self.classes:
                    raise LoadError(f"{ep.value} registered no Target named {ep.name}")
            self.loaded = True
        return self.registered()


catalog = Catalog()
register = catalog.register
unregister = catalog.unregister
registered = catalog.registered
load = catalog.load

__all__ = [
    "Catalog",
    "LoadError",
    "catalog",
    "load",
    "register",
    "registered",
    "unregister",
]
