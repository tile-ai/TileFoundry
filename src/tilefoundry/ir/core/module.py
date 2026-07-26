"""``Module`` — top-level compilation unit: functions, child modules, and plain
orchestration methods. See docs/spec/core-ir.md §1.
"""
from __future__ import annotations

import dataclasses
import functools
import types
from dataclasses import dataclass, field
from typing import Mapping, Union

from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.ir.types.tensor_type import TensorType

ModuleFunction = Union[HirFunction, PrimFunction]


@dataclass(frozen=True)
class Module:
    """Frozen container of functions + the name of the public entry function."""

    name: str
    functions: tuple[ModuleFunction, ...]
    entry: str
    modules: tuple["Module", ...] = field(default_factory=tuple)
    topologies: tuple[Topology, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)
    methods: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seal each function against further authoring mutation, and reject a
        name shared by two of functions / modules / methods."""
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
