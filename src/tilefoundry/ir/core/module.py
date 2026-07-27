"""``Module`` — top-level compilation unit: functions, child modules, and plain
orchestration methods. See docs/spec/core-ir.md §1.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import types
from dataclasses import dataclass, field
from typing import Mapping, Union

from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.target import Target

ModuleFunction = Union[HirFunction, PrimFunction]


def _owned_by(child: "Module", parent: "Module") -> "Module":
    """*child* linked back to *parent* as its owner.

    An unowned child is claimed in place, so the authored tree keeps the very
    objects its class body produced. A child that already belongs to another
    owner is claimed as a copy instead: the two owners may resolve different
    effective context, and re-pointing the original would silently change what
    the first owner's subtree answers.
    """
    owner = getattr(child, "_parent", None)
    if owner is None or owner is parent:
        object.__setattr__(child, "_parent", parent)
        return child
    clone = copy.copy(child)
    object.__setattr__(
        clone, "modules", tuple(_owned_by(node, clone) for node in child.modules)
    )
    object.__setattr__(clone, "_parent", parent)
    return clone


@dataclass(frozen=True)
class Module:
    """Frozen container of functions + the name of the public entry function.

    A Module is also the execution domain of the functions it owns: it carries
    the hardware ``target`` and the ordered ``topologies`` budget those
    functions run against. ``topologies=None`` declares nothing and inherits
    from the owning Module; ``topologies=()`` declares a topology-free Module.
    """

    name: str
    functions: tuple[ModuleFunction, ...]
    entry: str
    modules: tuple["Module", ...] = field(default_factory=tuple)
    target: Target | None = None
    topologies: tuple[Topology, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    methods: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seal each function against further authoring mutation, reject a
        name shared by two of functions / modules / methods, validate the
        declared execution context, and link each child to this owner."""
        for fn in self.functions:
            seal = getattr(fn, "seal", None)
            if callable(seal):
                seal()
        clash = sorted({fn.name for fn in self.functions} & {m.name for m in self.modules})
        if clash:
            raise ValueError(
                f"Module {self.name!r}: name(s) {clash} used by both a "
                f"function and a child module; names must be disjoint"
            )
        method_clash = sorted(
            set(self.methods)
            & ({fn.name for fn in self.functions} | {m.name for m in self.modules})
        )
        if method_clash:
            raise ValueError(
                f"Module {self.name!r}: name(s) {method_clash} used by both a "
                f"method and a function/child module; names must be disjoint"
            )
        if self.topologies is not None:
            names = [t.name for t in self.topologies]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(
                    f"Module {self.name!r}: duplicate topology name(s) {dupes}; "
                    f"one name must map to one level of the ordered hierarchy"
                )
        for child in self.modules:
            if child.target is not None:
                raise ValueError(
                    f"Module {self.name!r}: child module {child.name!r} declares "
                    f"its own target {child.target!r}; only the root module "
                    f"declares a target and children inherit it"
                )
        # Own the subtree rather than sharing it. A child links back to its
        # owner to resolve inherited context, so one child value placed under
        # two owners could otherwise resolve against whichever constructed
        # last -- including a copy made by ``dataclasses.replace``.
        object.__setattr__(
            self, "modules", tuple(_owned_by(child, self) for child in self.modules)
        )

    def _owner_path(self) -> str:
        """This Module's dotted path from the outermost declared owner."""
        names = [self.name]
        node = getattr(self, "_parent", None)
        while node is not None:
            names.append(node.name)
            node = getattr(node, "_parent", None)
        return ".".join(reversed(names))

    def resolve_target(self) -> Target:
        """The effective Target: this Module's declaration, else the nearest
        owner's. Raises when no owner in the chain declares one."""
        node: "Module | None" = self
        while node is not None:
            if node.target is not None:
                return node.target
            node = getattr(node, "_parent", None)
        raise ValueError(
            f"Module {self._owner_path()!r}: no target is declared by this "
            f"module or any of its owners; the root module must declare one"
        )

    def effective_topologies(self) -> tuple[Topology, ...]:
        """The effective ordered Topology tuple: this Module's declaration,
        else the nearest owner's, else empty at an undeclared root."""
        node: "Module | None" = self
        while node is not None:
            if node.topologies is not None:
                return node.topologies
            node = getattr(node, "_parent", None)
        return ()

    def resolve_topology(self, name: str) -> Topology:
        """The effective Topology named *name*; raises unless exactly one of
        the effective levels matches."""
        levels = self.effective_topologies()
        matches = tuple(t for t in levels if t.name == name)
        if not matches:
            available = ", ".join(t.name for t in levels) or "none"
            raise ValueError(
                f"Module {self._owner_path()!r}: no topology named {name!r}; "
                f"effective topology levels are {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Module {self._owner_path()!r}: topology {name!r} resolves to "
                f"{len(matches)} levels; one name must map to one level"
            )
        return matches[0]

    @property
    def weights(self) -> Mapping[str, TensorType]:
        """The union of every function's ``ConstTensor`` params, in (function,
        param) order."""
        result: dict[str, TensorType] = {}
        owner: dict[str, str] = {}
        for fn in self.functions:
            for p in fn.params:
                if not p.is_const:
                    continue
                prior = result.get(p.name)
                if prior is not None and prior != p.type:
                    raise ValueError(
                        f"Module {self.name!r}: weight {p.name!r} has "
                        f"conflicting TensorType between {owner[p.name]!r} "
                        f"({prior!r}) and {fn.name!r} ({p.type!r})"
                    )
                result[p.name] = p.type
                owner[p.name] = fn.name
        return result

    def __getattr__(self, name: str):
        """Resolve *name* to a function, a child module, or a bound method. A
        function resolves to a **callable that runs it**, not to the IR node —
        use ``lookup`` for the node."""
        if name.startswith("_"):
            raise AttributeError(name)
        matches = tuple(fn for fn in self.functions if fn.name == name)
        if len(matches) == 1:
            return functools.partial(self._run, matches[0])
        if len(matches) > 1:
            raise AttributeError(
                f"Module {self.name!r}: {name!r} resolves to {len(matches)} "
                f"entries; one name must map to one function"
            )
        mod_matches = tuple(m for m in self.modules if m.name == name)
        if len(mod_matches) == 1:
            return mod_matches[0]
        if len(mod_matches) > 1:
            raise AttributeError(
                f"Module {self.name!r}: {name!r} resolves to {len(mod_matches)} "
                f"child modules; one name must map to one module"
            )
        method = self.methods.get(name)
        if method is not None:
            return types.MethodType(method, self)
        raise AttributeError(
            f"Module {self.name!r} has no function, child module, or method {name!r}"
        )

    def function_named(self, name: str) -> tuple[ModuleFunction, ...]:
        """The functions whose name matches, in source order (0 or 1 of them in
        a verified module)."""
        return tuple(fn for fn in self.functions if fn.name == name)

    def lookup(self, name: str) -> ModuleFunction:
        """The function named *name*; raises unless exactly one matches."""
        matches = self.function_named(name)
        if len(matches) != 1:
            raise ValueError(
                f"Module {self.name!r}: {name!r} must resolve to exactly one "
                f"function, found {len(matches)}"
            )
        return matches[0]

    def entry_function(self) -> ModuleFunction:
        matches = self.function_named(self.entry)
        if not matches:
            raise ValueError(
                f"Module {self.name!r}: entry {self.entry!r} not in functions"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Module {self.name!r}: entry {self.entry!r} resolves to "
                f"{len(matches)} functions; entry must be a unique callable"
            )
        return matches[0]

    def load(self, resource) -> None:
        """Bind this node's weights by name from *resource*, then recurse into
        each child."""
        bound: dict[str, object] = {}
        for name in self.weights:
            try:
                bound[name] = resource.load(name)
            except KeyError as e:
                raise KeyError(f"Module {self.name!r}: missing weight {name!r}") from e
        object.__setattr__(self, "_bound", bound)
        for child in self.modules:
            child.load(resource.subtree(child.name))

    def _run(self, fn: ModuleFunction, *acts):
        """Evaluate *fn*, weights filled by name from ``load``, the rest from
        *acts* positionally."""
        from tilefoundry.evaluator import evaluate  # noqa: PLC0415 -- avoid IR→evaluator cycle

        bound = getattr(self, "_bound", {})
        args = []
        activations = iter(acts)
        for param in fn.params:
            if param.is_const:
                try:
                    args.append(bound[param.name])
                except KeyError:
                    raise KeyError(
                        f"Module {self.name!r}: weight {param.name!r} of "
                        f"{fn.name!r} is not bound; call load(resource) first"
                    ) from None
            else:
                args.append(next(activations))
        return evaluate(fn, *args)

    def forward(self, *acts):
        """Run this node's step: its ``forward`` orchestration method if it has
        one, else the entry function."""
        method = self.methods.get("forward")
        if method is not None:
            return method(self, *acts)
        return self._run(self.lookup(self.entry), *acts)

    __call__ = forward

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Run every node's per-weight converters over *raw* and write the
        canonical weights to *out_dir* as one safetensors shard plus an index.
        See docs/spec/runtime.md §1.1.2."""
        flat: dict[str, object] = {}
        self._prepare_into(raw, "", flat, device)

        import json  # noqa: PLC0415 -- stdlib, only needed here
        from pathlib import Path  # noqa: PLC0415

        from safetensors.torch import save_file  # noqa: PLC0415 -- optional runtime dep

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        shard = "model-00001-of-00001.safetensors"
        save_file(flat, str(out / shard))
        (out / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard for name in flat}})
        )

    def _prepare_into(self, raw, prefix: str, flat: dict, device: str) -> None:
        import torch  # noqa: PLC0415 -- optional runtime dep

        from tilefoundry.evaluator import evaluate  # noqa: PLC0415 -- avoid IR→evaluator cycle
        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415

        converter_map: dict[str, ModuleFunction] = {}
        for fn in self.functions:
            for weight_name, conv in getattr(fn, "converters", ()):
                prior = converter_map.get(weight_name)
                if prior is not None and prior is not conv:
                    raise ValueError(
                        f"Module {self.name!r}: weight {weight_name!r} has "
                        f"more than one registered converter"
                    )
                converter_map[weight_name] = conv

        def _fetch(name):
            # A one-to-many alias is stacked here — prepare's only reshaping.
            parts = raw.load_group(name)
            return torch.stack(parts) if parts is not None else raw.load(name)

        for w, decl_type in self.weights.items():
            conv = converter_map.get(w)
            key = prefix + w
            if conv is None:
                value = _fetch(w)
            else:
                value = evaluate(conv, *[_fetch(p.name) for p in conv.params], device=device)
            source = f"converter for weight {key!r}" if conv is not None else f"raw weight {key!r}"
            if tuple(value.shape) != tuple(decl_type.shape):
                raise ValueError(
                    f"Module {self.name!r}: {source} has shape "
                    f"{tuple(value.shape)}, declared {tuple(decl_type.shape)}"
                )
            expected_dtype = to_torch_dtype(decl_type.dtype)
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Module {self.name!r}: {source} has dtype "
                    f"{value.dtype}, declared {decl_type.dtype}"
                )
            flat[key] = value.detach().contiguous().cpu()

        for child in self.modules:
            child._prepare_into(raw.subtree(child.name), f"{prefix}{child.name}.", flat, device)

    def renamed(self, name: str) -> "Module":
        """A copy of this node under a different ``name``. Shallow: children are
        shared, so an independent instance needs a fresh build."""
        return dataclasses.replace(self, name=name)


__all__ = ["Module", "ModuleFunction"]
