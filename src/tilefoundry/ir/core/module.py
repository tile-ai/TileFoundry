"""``Module`` — top-level compilation unit: functions, child modules.

``Module`` — top-level compilation unit: functions, child modules, and plain
orchestration methods. See [core-ir §1](docs/spec/core-ir.md#1-module).
"""

from __future__ import annotations

import types
from dataclasses import MISSING, dataclass, field
from dataclasses import replace as _replace
from typing import Mapping, Union

from tilefoundry import evaluator
from tilefoundry.evaluator.interpreter import _run_bound
from tilefoundry.evaluator.value import tensor_type_of
from tilefoundry.ir.core.expr import Expr
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.callable_type import callable_type_for
from tilefoundry.ir.types.mesh import Topology
from tilefoundry.ir.types.substitute import canonicalize_topology_dims
from tilefoundry.ir.types.tensor_type import TensorType, TupleType, UnitType
from tilefoundry.ir.types.utils import types_compatible
from tilefoundry.target.base import Target, target_instance

ModuleFunction = Union[HirFunction, PrimFunction]

_MISSING_PREPARED_WEIGHT = (
    "[runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)"
)


@dataclass(eq=False, init=False)
class ModuleList(Expr):
    """An instance slot holding shared definitions, with a lazily projected type.

    ``ModuleList(definition)`` is a single slot; a count or a tuple makes an
    indexed slot. Its key is assigned when the slot enters ``Module.children``.
    """

    key: str | None
    elements: tuple[Module | type[RuntimeModule], ...]
    indexed: bool

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, definition, count=MISSING):
        if count is not MISSING:
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("ModuleList count must be a positive integer")
            elements = (definition,) * count
            indexed = True
        elif isinstance(definition, tuple):
            elements = definition
            indexed = True
        else:
            elements = (definition,)
            indexed = False
        if not elements:
            raise TypeError("ModuleList requires a Module or a non-empty tuple of Modules")
        if not all(isinstance(element, Module) for element in elements):
            from tilefoundry.runtime.module import RuntimeModule  # noqa: PLC0415

            if not all(
                isinstance(element, type) and issubclass(element, RuntimeModule)
                for element in elements
            ):
                raise TypeError("ModuleList requires Modules or runtime module classes")
        self.key = None
        self.elements = elements
        self.indexed = indexed
        self.metadata = ()

    def __repr__(self) -> str:
        """Describe the slot without projecting types or expanding definitions."""
        names = []
        start = 0
        while start < len(self.elements):
            element = self.elements[start]
            stop = start + 1
            while stop < len(self.elements) and self.elements[stop] is element:
                stop += 1
            count = stop - start
            name = element.__name__ if isinstance(element, type) else element.name
            names.append(f"{name}*{count}" if count > 1 else name)
            start = stop
        elements = ", ".join(names) + ("," if len(names) == 1 else "")
        return f"ModuleList(key={self.key!r}, indexed={self.indexed}, elements=({elements}))"

    @property
    def type(self):
        signatures = tuple(
            callable_type_for(
                (param for param in entry.params if not param.is_const),
                UnitType() if isinstance(entry, PrimFunction) else entry.return_type
            )
            for entry in (element.entry_function() for element in self.elements)
        )
        return TupleType(signatures) if self.indexed else signatures[0]

    @type.setter
    def type(self, value):
        """Keep the definition-derived signature independent of inferred stamps."""
        pass

    def bind(self, key: str) -> None:
        if self.key is not None and self.key != key:
            raise ValueError(
                f"ModuleList is already in slot {self.key!r}; cannot also use slot {key!r}"
            )
        self.key = key

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]


@dataclass(unsafe_hash=True)
class Module:
    """Container of functions + the name of the public entry function.

    A Module is also the execution domain of the functions it owns: it carries
    the hardware ``target`` and the ordered ``topologies`` budget those
    functions run against. ``topologies=None`` inherits from the caller's
    ValuationContext; ``topologies=()`` declares a topology-free Module.
    """

    name: str = field(hash=False)
    functions: tuple[ModuleFunction, ...]

    entry: str | None = None
    children: dict[str, ModuleList] = field(default_factory=dict, hash=False)
    target: Target | None = None
    topologies: tuple[Topology, ...] | None = field(default=None, hash=False)
    metadata: dict[str, object] = field(default_factory=dict, hash=False)
    methods: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Post init.

        Seal each function against further authoring mutation, reject a
        name shared by two of functions / modules / methods, validate the
        declared execution context, and assign each instance slot its key.
        """
        if self.target is not None:
            target_instance(self.target)
        for fn in self.functions:
            seal = getattr(fn, "seal", None)
            if callable(seal):
                seal()
        clash = sorted({fn.name for fn in self.functions} & set(self.children))
        if clash:
            raise ValueError(
                f"Module {self.name!r}: name(s) {clash} used by both a "
                f"function and a child module; names must be disjoint"
            )
        method_clash = sorted(
            set(self.methods)
            & ({fn.name for fn in self.functions} | set(self.children))
        )
        if method_clash:
            raise ValueError(
                f"Module {self.name!r}: name(s) {method_clash} used by both a "
                f"method and a function/child module; names must be disjoint"
            )
        if self.topologies is not None:
            topologies = tuple(canonicalize_topology_dims(t) for t in self.topologies)
            if topologies != self.topologies:
                self.topologies = topologies
            names = [t.name for t in self.topologies]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(
                    f"Module {self.name!r}: duplicate topology name(s) {dupes}; "
                    f"one name must map to one level of the ordered hierarchy"
                )
        for key, slot in self.children.items():
            slot.bind(key)
            for child in slot.elements:
                if not isinstance(child, Module):
                    raise TypeError(f"Module {self.name!r}: slot {key!r} requires Module definitions")

    def resolve_target(self) -> Target:
        """Resolve this definition's own target."""
        root = self
        if root.target is not None:
            return root.target
        raise ValueError(
            f"Module {root.name!r}: no target is declared; declare a Target "
            "instance on the root module. The rule: "
            "tilefoundry spec core-ir target-inheritance"
        )

    def effective_topologies(self) -> tuple[Topology, ...]:
        """This definition's declared topology hierarchy, or an empty one."""
        return self.topologies if self.topologies is not None else ()

    def resolve_topology(self, name: str) -> Topology:
        """The effective Topology named *name*.

        The effective Topology named *name*; raises unless exactly one of
        the effective levels matches.
        """
        levels = self.effective_topologies()
        matches = tuple(t for t in levels if t.name == name)
        if not matches:
            available = ", ".join(t.name for t in levels) or "none"
            raise ValueError(
                f"Module {self.name!r}: no topology named {name!r}; "
                f"effective topology levels are {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Module {self.name!r}: topology {name!r} resolves to "
                f"{len(matches)} levels; one name must map to one level"
            )
        return matches[0]

    @property
    def weights(self) -> Mapping[str, TensorType]:
        """The union of every function's ``ConstTensor`` params, in (function, param) order.

        The union of every function's ``ConstTensor`` params, in (function,
        param) order.
        """
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
        """Resolve *name* to a function, a child module, or a bound method.

        A function resolves to its IR node. Execution is provided by
        ``tilefoundry.evaluator.evaluate``.
        """
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
        slot = self.children.get(name)
        if slot is not None:
            return slot if slot.indexed else slot.elements[0]
        method = self.methods.get(name)
        if method is not None:
            return types.MethodType(method, self)
        raise AttributeError(
            f"Module {self.name!r} has no function, child module, or method {name!r}"
        )

    def lookup(self, name: str) -> ModuleFunction:
        """The function named *name*; raises unless exactly one matches."""
        matches = tuple(fn for fn in self.functions if fn.name == name)
        if len(matches) != 1:
            raise ValueError(
                f"Module {self.name!r}: {name!r} must resolve to exactly one "
                f"function, found {len(matches)}"
            )
        return matches[0]

    def owns(self, function: object, *, derived: bool = False) -> bool:
        """Return whether this module owns *function* by object identity.

        Declared functions and their variants count. Structural equality and
        matching names do not. With ``derived=True``, a recorded
        ``_specialized_from`` chain may lead back to an owned function; the
        whole chain is walked, because a rebuild may itself be rebuilt.

        See [core-ir §1](docs/spec/core-ir.md#1-module).
        """
        for owned in self.functions:
            if function is owned:
                return True
            for variant in getattr(owned, "variants", ()):
                if function is variant:
                    return True
        if not derived:
            return False
        origin = getattr(function, "_specialized_from", None)
        while origin is not None:
            if self.owns(origin):
                return True
            origin = getattr(origin, "_specialized_from", None)
        return False

    def entry_function(self) -> ModuleFunction:
        if self.entry is None:
            names = ", ".join(fn.name for fn in self.functions) or "no functions"
            raise ValueError(
                f"Module {self.name!r} declares no entry, so it has no default "
                f"step. It declares {names}; name the function to use -- "
                "lookup('<name>') for the node, or <module>.<name>(...) to run "
                "it. The rule: tilefoundry spec core-ir default-step"
            )
        matches = tuple(fn for fn in self.functions if fn.name == self.entry)
        if not matches:
            raise ValueError(f"Module {self.name!r}: entry {self.entry!r} not in functions")
        if len(matches) > 1:
            raise ValueError(
                f"Module {self.name!r}: entry {self.entry!r} resolves to "
                f"{len(matches)} functions; entry must be a unique callable"
            )
        return matches[0]

    def __call__(self, *args):
        """Evaluate the declared entry with every parameter supplied."""
        return evaluator.evaluate(self.entry_function(), *args)

    def load(self, resource) -> "LoadedModule":
        """Remember *resource* for this Module and its children.

        Loading is a pure binding operation. Weight values are read by the
        evaluator at the point a function uses a declared ``ConstTensor``.

        [runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)
        """
        return self._load_under(resource, self.target, (), "")

    def _load_under(self, resource, target, topologies, path: str) -> "LoadedModule":
        target = self.target if self.target is not None else target
        topologies = self.topologies if self.topologies is not None else topologies
        children = {}
        for key, slot in self.children.items():
            base = resource.subtree(key)
            child_path = f"{path}.{key}" if path else key
            if slot.indexed:
                children[key] = tuple(
                    child._load_under(base.subtree(str(i)), target, topologies, f"{child_path}.{i}")
                    for i, child in enumerate(slot.elements)
                )
            else:
                children[key] = slot.elements[0]._load_under(base, target, topologies, child_path)
        return LoadedModule(
            module=self, resource=resource, children=children,
            target=target, topologies=topologies, path=path,
        )

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Prepare canonical weights and write the safetensors checkpoint."""
        staged: dict[str, object] = {}
        self._prepare_into(raw, "", staged, device, target=self.target)
        flat = {name: value.detach().contiguous().cpu() for name, value in staged.items()}

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

    def _prepare_into(
        self, raw, prefix: str, flat: dict, device: str, *,
        target=None, topologies=(), path: str = "",
    ) -> "LoadedModule":
        """Stage this node's canonical weights, children before their owner.

        Converters are HIR bodies on this offline path, making the local
        evaluator import the one intentional Module-to-evaluator execution dependency.
        """
        import torch  # noqa: PLC0415 -- optional runtime dep

        from tilefoundry.runtime.resource import (  # noqa: PLC0415 -- runtime depends on Module
            DictResource,
        )

        target = self.target if self.target is not None else target
        topologies = self.topologies if self.topologies is not None else topologies
        children = {}
        for key, slot in self.children.items():
            base = raw.subtree(key)
            child_path = f"{path}.{key}" if path else key
            if slot.indexed:
                children[key] = tuple(
                    child._prepare_into(
                        base.subtree(str(i)), f"{prefix}{key}.{i}.", flat, device,
                        target=target, topologies=topologies, path=f"{child_path}.{i}",
                    )
                    for i, child in enumerate(slot.elements)
                )
            else:
                children[key] = slot.elements[0]._prepare_into(
                    base, f"{prefix}{key}.", flat, device,
                    target=target, topologies=topologies, path=child_path,
                )
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
            parts = raw.load_group(name)
            value = torch.stack(parts) if parts is not None else raw.load(name)
            return value.to(device)

        reading = LoadedModule(
            module=self, resource=DictResource(flat, prefix=prefix), children=children,
            target=target, topologies=topologies, path=path,
        )
        for weight_name, decl_type in self.weights.items():
            conv = converter_map.get(weight_name)
            key = prefix + weight_name
            if conv is None:
                value = _fetch(weight_name)
            else:
                value = _run_bound(
                    conv, [_fetch(param.name) for param in conv.params],
                    device=device, reading=reading,
                )
            actual = tensor_type_of(value, like=decl_type)
            if actual.shape != decl_type.shape:
                source = (
                    f"converter for weight {key!r}"
                    if conv is not None
                    else f"raw weight {key!r}"
                )
                raise ValueError(
                    f"Module {self.name!r}: {source} has shape "
                    f"{actual.shape}, declared {decl_type.shape}"
                )
            if not types_compatible(decl_type, actual):
                source = (
                    f"converter for weight {key!r}"
                    if conv is not None
                    else f"raw weight {key!r}"
                )
                raise ValueError(
                    f"Module {self.name!r}: {source} has dtype {value.dtype}, declared "
                    f"{decl_type.dtype}; the way out is a weight converter on the model, "
                    "not a flag on the read side"
                )
            flat[key] = value
        return reading

@dataclass(frozen=True)
class LoadedModule:
    """An immutable Module binding and the resource from which it reads.

    [runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)
    """

    module: Module
    resource: object
    children: dict[str, LoadedModule | tuple[LoadedModule, ...]] = field(default_factory=dict)
    target: Target | None = None
    topologies: tuple[Topology, ...] = ()
    path: str = ""

    @property
    def name(self) -> str:
        return self.module.name

    def __getattr__(self, name: str):
        """Resolve a function, child module, or method against this loading."""
        if name.startswith("_"):
            raise AttributeError(name)
        module = self.module
        matches = tuple(fn for fn in module.functions if fn.name == name)
        if len(matches) == 1:
            return _replace(
                self, module=_replace(module, entry=name, target=self.target, topologies=self.topologies)
            )
        if len(matches) > 1:
            raise AttributeError(
                f"LoadedModule {self.name!r}: {name!r} resolves to "
                f"{len(matches)} entries; one name must map to one function"
            )
        child = self.children.get(name)
        if child is not None:
            return child
        method = module.methods.get(name)
        if method is not None:
            return types.MethodType(method, self)
        raise AttributeError(
            f"LoadedModule {self.name!r} has no function, child module, or method {name!r}"
        )

    def __call__(self, *args):
        """Evaluate this loading's declared entry with activation arguments."""
        return evaluator.evaluate(self, *args)

    def evaluation_target(self):
        """Run my declared entry, its constants read from this loading."""
        return self.module.entry_function(), self


def resolve_symbol(module, name: str, *, nested: tuple[str, ...] = ()) -> ModuleFunction:
    """Resolve one function in the current module or a child-slot path.

    An iterable is the legacy flat function scope; it cannot contain child slots.
    Every failure names the complete requested path, with no subtree name search.
    """
    path = ".".join((*nested, name))
    if isinstance(module, Module):
        selected = module
        for slot in nested:
            child = selected.children.get(slot)
            if child is None:
                raise ValueError(f"symbol {path!r}: child slot {slot!r} does not exist")
            if child.indexed:
                raise ValueError(f"symbol {path!r}: child slot {slot!r} requires an instance index")
            selected = child.elements[0]
        functions = selected.functions
    else:
        if nested:
            raise ValueError(f"symbol {path!r}: child slots require a Module tree")
        functions = module
    functions = tuple(functions)
    matches = tuple(function for function in functions if function.name == name)
    if not matches:
        matches = tuple(
            variant for function in functions for variant in function.variants
            if variant.name == name
        )
    if len(matches) != 1:
        raise ValueError(f"symbol {path!r} must resolve to exactly one function, found {len(matches)}")
    return matches[0]


__all__ = ["Module", "ModuleList", "LoadedModule", "resolve_symbol"]
