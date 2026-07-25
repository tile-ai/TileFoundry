"""Module — top-level compilation unit.

``entry`` names the public entry function; ``modules`` nests child
``Module``s, addressed by attribute path (e.g. ``root.layer0.attention``). A
class body collects three member kinds (see ``tilefoundry.module``): DSL
functions, child ``Module``s, and plain Python orchestration methods
(``methods``, bound like instance methods — ``m.forward(...)``).
``weights`` is derived from every function's ``ConstTensor`` params; there
is no ``states`` — a persistent tensor (e.g. a KV cache) is an ordinary
``Tensor`` param the caller owns.

``forward`` runs the step; ``load`` binds weights from a ``RuntimeResource``;
``prepare`` runs every node's per-weight converters offline. See
docs/spec/core-ir.md, docs/spec/runtime.md.
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
        """Seal each function so authoring mutation (``add_variant`` /
        ``.specialize``) is forbidden once it belongs to a Module. Sealing is
        idempotent and only applies to functions that support it (hir
        Functions); other entries are left untouched. Child modules are
        already fully constructed (and so already sealed their own functions)
        by the time they are passed in here, so sealing does not recurse.

        A function name, a child module name, and a method name must be
        disjoint at this module's own level — all three are resolved through
        the same attribute / addressing surface (``__getattr__``), so a name
        used by more than one would be ambiguous."""
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
        """Derived weight schema: the union, in (function order, param
        order), of every function's ``ConstTensor`` params. A name shared by
        two functions must carry an identical ``TensorType``."""
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
        """Attribute access forwards to the function, child module, or bound
        method of that name, so a module reads like the model it mirrors:
        ``decoder.self_attention(...)`` / ``decoder.layer0.attention`` /
        ``decoder.init_caches(...)``. A function name resolves to a **callable**
        that runs it (weights filled by name, activations positional) — the same
        spelling its ``RuntimeModule`` twin answers with a kernel, which is what
        lets one orchestration method serve both sides. The IR node itself is
        reached with ``lookup`` / ``function_named``. Each name maps to at most
        one entry (specialization variants and converters live on their base's
        ``variants`` / ``converters``, not as separate entries). Only fires for
        names absent as real attributes; dunder/private names are never
        functions, modules, or methods and fall through to ``AttributeError``."""
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
        """Return the functions whose name matches, in source order.

        Each name maps to at most one entry, so in a verified module this is
        length 0 or 1 (specialization variants live on the function's
        ``variants``, not as separate same-name entries).
        """
        return tuple(fn for fn in self.functions if fn.name == name)

    def lookup(self, name: str) -> ModuleFunction:
        """Return the function named ``name``; raise unless exactly one matches.

        It is the module-level resolution contract for a ``SymbolRef`` callee.
        """
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
        """Bind this node's ``weights`` by name from *resource*, then recurse
        into each child module under ``resource.subtree(child.name)``."""
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
        """Evaluate *fn* with its ``ConstTensor`` params filled by name from
        ``load``'s bound weights and every other param taken positionally from
        *acts* — the semantic counterpart of the twin calling a kernel."""
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
        """Run this node's step: a registered ``methods["forward"]``
        orchestration callable if present (called bound, like an instance
        method), else the entry @func through the evaluator. A multi-node
        composition is chained by the caller, one ``forward`` per node."""
        method = self.methods.get("forward")
        if method is not None:
            return method(self, *acts)
        return self._run(self.lookup(self.entry), *acts)

    __call__ = forward

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Run every node's per-weight converters over *raw* and write the
        canonical weights to *out_dir* (docs/spec/runtime.md §1.1.2).

        A weight with a registered ``Function.converter`` is built from
        *raw* by the converter's own (raw) param names — ``load_group``
        assembles a one-to-many alias via ``torch.stack``, prepare's only
        reshaping. A weight with no converter passes through *raw*
        unchanged. Output: one safetensors shard +
        ``model.safetensors.index.json``.
        """
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
            """One raw tensor for *name*: a one-to-many alias is assembled here
            (prepare's only reshaping), a one-to-one alias is loaded as is."""
            parts = raw.load_group(name)
            return torch.stack(parts) if parts is not None else raw.load(name)

        for w, decl_type in self.weights.items():
            conv = converter_map.get(w)
            key = prefix + w
            if conv is None:
                # No converter: the canonical form is the raw form (a stack of
                # per-shard tensors still counts — assembly is not a transform).
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
        """Return a copy of this node under a different ``name`` — one
        definition, N addressable instances (e.g. 43 identical decoder
        layers from a factory)."""
        return dataclasses.replace(self, name=name)


__all__ = ["Module", "ModuleFunction"]
