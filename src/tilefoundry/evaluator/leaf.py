"""Leaf implementation registry + module-tree weight ``post_init`` runner.

M1 of docs/plans/agent-kernel-loop/P0a-tonight-nested-module-e2e.md: a
``Module`` tree's nested functions ("leaves") can each have a swap-in
:class:`ImplementationPackage` registered against them; the evaluator (see
``tilefoundry.evaluator.interpreter``) intercepts a ``Call`` to a registered
leaf and runs the implementation instead of recursing into the callee's HIR
body. Only ``language="torch"`` is wired tonight.

Registration is keyed by ``(module path, leaf name)`` — the dotted path of
the ``Module`` that owns the leaf function, relative to whatever root the
tree is addressed from (the root itself is path ``()``). :func:`leaf_paths`
derives this path for every function in a tree, so a caller only needs the
leaf's own name to build a :class:`LeafRegistry`; the evaluator itself only
ever sees the flat ``{fn_name: ImplementationPackage}`` view
(:meth:`LeafRegistry.by_function_name`), since function names are unique
within one evaluated tree.

:class:`WeightLoader` runs each module-with-``post_init``'s hook exactly once
over a flat ``{"layer0.moe.shared_expert.w1_weight": tensor, ...}`` weights
dict (module-path-prefixed names, per M1) and caches the result so re-running
it (e.g. a second decode step against the same weights) does not re-run the
conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tilefoundry.ir.core.module import Module

LeafFn = Callable[..., Any]


@dataclass(frozen=True)
class ImplementationPackage:
    """A leaf's swap-in implementation.

    Only ``language="torch"`` is wired tonight: ``fn_or_source`` is a plain
    callable — torch tensors in, torch tensor(s) out, running on the leaf's
    target device (cuda). The remaining fields are kept so a cuda-c++ /
    triton / tilelang / cute-dsl backend can plug in later without reshaping
    this contract.
    """

    language: str
    fn_or_source: LeafFn
    entry: str
    launch_cfg: object | None = None
    workspace_bytes: int | None = None


class LeafRegistry:
    """``(module path, leaf name) -> ImplementationPackage``."""

    def __init__(self) -> None:
        self._table: dict[tuple[tuple[str, ...], str], ImplementationPackage] = {}

    def register(self, path: tuple[str, ...], name: str, impl: ImplementationPackage) -> None:
        self._table[(tuple(path), name)] = impl

    def get(self, path: tuple[str, ...], name: str) -> ImplementationPackage | None:
        return self._table.get((tuple(path), name))

    def by_function_name(self) -> dict[str, ImplementationPackage]:
        """Flatten to ``{fn_name: impl}`` — the shape the evaluator consumes
        (function names are unique within one evaluated tree)."""
        return {name: impl for (_path, name), impl in self._table.items()}

    def __len__(self) -> int:
        return len(self._table)


def walk(root: Module, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Module]]:
    """Every module in the tree paired with its dotted path *relative to
    ``root``* (``root`` itself at path ``()``)."""
    out = [(prefix, root)]
    for child in root.modules:
        out.extend(walk(child, (*prefix, child.name)))
    return out


def leaf_paths(root: Module) -> dict[str, tuple[str, ...]]:
    """``fn name -> path of the module that owns it`` — a declarative helper
    for building a :class:`LeafRegistry` from a module tree's own structure
    instead of hand-tracking paths."""
    out: dict[str, tuple[str, ...]] = {}
    for path, module in walk(root):
        for fn in module.functions:
            out[fn.name] = path
    return out


def _own_namespace(raw: dict[str, Any], prefix: str, module: Module) -> dict[str, Any]:
    """The slice of ``raw`` directly owned by ``module`` at ``prefix``: keys
    under ``prefix`` but not also under one of ``module``'s own children's
    (deeper) prefixes."""
    child_prefixes = tuple(f"{prefix}{child.name}." for child in module.modules)
    ns = {}
    for key, value in raw.items():
        if not key.startswith(prefix):
            continue
        if any(key.startswith(cp) for cp in child_prefixes):
            continue
        ns[key[len(prefix):]] = value
    return ns


class WeightLoader:
    """Runs each module-with-``post_init``'s hook exactly once (cached by
    module path) over a flat, module-path-prefixed weights dict."""

    def __init__(self, root: Module) -> None:
        self.root = root
        self._cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.post_init_runs = 0

    def load(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Return a new flat dict: every module-with-``post_init``'s own
        weight keys are replaced by its (cached) transformed namespace; every
        other key passes through unchanged."""
        out = dict(raw)
        for path, module in walk(self.root):
            if module.post_init is None:
                continue
            prefix = "".join(f"{p}." for p in path)
            if path not in self._cache:
                namespace = _own_namespace(raw, prefix, module)
                self._cache[path] = module.post_init(namespace)
                self.post_init_runs += 1
            for key, value in self._cache[path].items():
                out[prefix + key] = value
        return out


__all__ = [
    "ImplementationPackage",
    "LeafRegistry",
    "WeightLoader",
    "leaf_paths",
    "walk",
]
