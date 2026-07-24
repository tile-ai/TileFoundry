"""Module — top-level compilation unit.

``entry`` names the public entry function; verify_module checks it resolves.
``metadata`` holds lowering / target configuration (e.g. target).
``topologies`` carries the module-level topology declarations; these form the
namespace against which ``with Mesh(topology="cta", ...)`` strings resolve.
``modules`` nests child ``Module``s (e.g. a decoder layer's attention / MoE
sub-blocks) purely as a namespace / addressing device — a tree of modules is
addressed by attribute path (``root.layer0.attention``); entry resolution and
``Call`` semantics are unaffected, and a child's functions are never folded
into the parent's ``functions``. ``weights`` and ``states`` declare this
module's own named tensor slots (shape/dtype only, no values); they are the
schema ``prepare`` / ``forward`` resolve against, and neither field
participates in typeinfer or verify.

A ``Module`` is directly runnable: ``forward`` evaluates the entry @func — the
same-shaped forward as its runtime twin, run through the evaluator instead of
kernels — and ``prepare`` runs each node's weight ``convert`` @func offline. The
runtime twin (``tilefoundry.runtime.RuntimeModule``) is written separately but
mirrors this structure, loads the directory ``prepare`` writes, and
``runtime.check`` compares the two forwards.
"""
from __future__ import annotations

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
    weights: Mapping[str, TensorType] = field(default_factory=dict)
    states: Mapping[str, TensorType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seal each function so authoring mutation (``add_variant`` /
        ``.specialize``) is forbidden once it belongs to a Module. Sealing is
        idempotent and only applies to functions that support it (hir
        Functions); other entries are left untouched. Child modules are
        already fully constructed (and so already sealed their own functions)
        by the time they are passed in here, so sealing does not recurse.

        A function name and a child module name must be disjoint at this
        module's own level — both are resolved through the same attribute /
        addressing surface (``__getattr__``), so a name used by both would be
        ambiguous."""
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

    def __getattr__(self, name: str) -> "ModuleFunction | Module":
        """Attribute access forwards to the function or child module of that
        name, so a module reads like the model it mirrors:
        ``decoder.self_attention`` / ``decoder.layer0.attention``. Each name
        maps to at most one entry (specialization variants live on the
        function's ``variants``, not as separate entries). Only fires for
        names absent as real attributes; dunder/private names are never
        functions or modules and fall through to ``AttributeError``."""
        if name.startswith("_"):
            raise AttributeError(name)
        matches = tuple(fn for fn in self.functions if fn.name == name)
        if len(matches) == 1:
            return matches[0]
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
        raise AttributeError(f"Module {self.name!r} has no function or child module {name!r}")

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

    def forward(self, resource, *acts, device: str = "cuda"):
        """Evaluate the entry @func with weight/state params filled by name from
        *resource* and activation params supplied positionally — the
        same-shaped forward as a ``runtime.RuntimeModule`` twin, run through the
        evaluator instead of kernels (``runtime.check`` compares the two). A
        multi-node composition is chained by the caller, one ``forward`` per
        node, mirroring the runtime orchestration."""
        from tilefoundry.evaluator import evaluate  # noqa: PLC0415 -- avoid IR→evaluator cycle

        fn = self.lookup(self.entry)
        args = []
        activations = iter(acts)
        for param in fn.params:
            if param.name in self.weights or param.name in self.states:
                args.append(resource.load(param.name))
            else:
                args.append(next(activations))
        return evaluate(fn, *args, device=device)

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Run every node's weight ``convert`` @func over *raw* and write the
        converted (canonical) weights to *out_dir* (dot-prefixed by module
        path).

        A module carries its weight converter as a sibling @func named
        ``convert`` (peer to the compute @func); its params are the raw
        checkpoint names and its results are this module's declared ``weights``
        (declaration order), in the canonical form the compute @func consumes.
        A node with no ``convert`` passes its declared ``weights`` through
        unchanged; declared ``states`` present in *raw* pass through as initial
        values. Output is a plain safetensors directory (one shard +
        ``model.safetensors.index.json``) — no content-hash cache / manifest.
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
        from tilefoundry.evaluator import evaluate  # noqa: PLC0415 -- avoid IR→evaluator cycle

        convert = next((f for f in self.functions if f.name == "convert"), None)
        weight_names = list(self.weights)
        if convert is not None:
            raw_args = [raw.load(p.name) for p in convert.params]
            result = evaluate(convert, *raw_args, device=device)
            values = list(result) if isinstance(result, tuple) else [result]
            if len(values) != len(weight_names):
                raise ValueError(
                    f"Module {self.name!r}: 'convert' returned {len(values)} tensor(s) "
                    f"but the module declares {len(weight_names)} weight(s) {weight_names}"
                )
            for name, value in zip(weight_names, values):
                flat[prefix + name] = value.detach().contiguous().cpu()
        else:
            for name in weight_names:
                flat[prefix + name] = raw.load(name).detach().contiguous().cpu()

        for state_name in self.states:
            try:
                flat[prefix + state_name] = raw.load(state_name).detach().contiguous().cpu()
            except KeyError:
                pass  # no initial value in raw — the runtime allocates it

        for child in self.modules:
            child._prepare_into(raw.subtree(child.name), f"{prefix}{child.name}.", flat, device)


__all__ = ["Module", "ModuleFunction"]
