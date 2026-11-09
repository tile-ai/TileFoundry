"""``Module`` — top-level compilation unit: functions, child modules.

``Module`` — top-level compilation unit: functions, child modules, and plain
orchestration methods. See [core-ir §1](docs/spec/core-ir.md#1-module).
"""

from __future__ import annotations

import types
from dataclasses import MISSING, dataclass, field, replace
from typing import Mapping, Union

from tilefoundry import services, target
from tilefoundry.ir import dim, expr
from tilefoundry.ir import types as ir_types

from .function import Function
from .prim_function import PrimFunction

ModuleFunction = Union[Function, PrimFunction]


@dataclass(eq=False, init=False)
class ModuleList(expr.Expr):
    """An instance slot holding shared definitions, with a lazily projected type.

    ``ModuleList(definition)`` is a single slot; a count or a tuple makes an
    indexed slot. Its key is assigned when the slot enters ``Module.children``.
    """

    key: str | None
    elements: tuple[Module | type, ...]
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
        if not all(isinstance(element, (Module, type)) for element in elements):
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
            ir_types.callable_type_for(
                (param for param in entry.params if not param.is_const),
                ir_types.UnitType() if isinstance(entry, PrimFunction) else entry.return_type
            )
            for entry in (element.entry_function() for element in self.elements)
        )
        return ir_types.TupleType(signatures) if self.indexed else signatures[0]

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
    valuation.Context; ``topologies=()`` declares a topology-free Module.
    """

    name: str = field(hash=False)
    functions: tuple[ModuleFunction, ...]

    entry: str | None = None
    children: dict[str, ModuleList] = field(default_factory=dict, hash=False)
    target: target.Target | None = None
    topologies: tuple[ir_types.Topology, ...] | None = field(default=None, hash=False)
    metadata: dict[str, object] = field(default_factory=dict, hash=False)
    methods: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Post init.

        Seal each function against further authoring mutation, reject a
        name shared by two of functions / modules / methods, validate the
        declared execution context, and assign each instance slot its key.
        """
        if self.target is not None:
            target.instance(self.target)
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
            topologies = tuple(dim.canonicalize_topology_dims(t) for t in self.topologies)
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

    def resolve_target(self) -> target.Target:
        """Resolve this definition's own target."""
        root = self
        if root.target is not None:
            return root.target
        raise ValueError(
            f"Module {root.name!r}: no target is declared; declare a Target "
            "instance on the root module. The rule: "
            "tilefoundry spec core-ir target-inheritance"
        )

    def effective_topologies(self) -> tuple[ir_types.Topology, ...]:
        """This definition's declared topology hierarchy, or an empty one."""
        return self.topologies if self.topologies is not None else ()

    def resolve_topology(self, name: str) -> ir_types.Topology:
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
    def weights(self) -> Mapping[str, ir_types.TensorType]:
        """The union of every function's ``ConstTensor`` params, in (function, param) order.

        The union of every function's ``ConstTensor`` params, in (function,
        param) order.
        """
        result: dict[str, ir_types.TensorType] = {}
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
        ``specialized_from`` chain may lead back to an owned function; the
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
        origin = getattr(function, "specialized_from", None)
        while origin is not None:
            if self.owns(origin):
                return True
            origin = getattr(origin, "specialized_from", None)
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

    def replace_functions(self, functions, *, entry: str | None = None) -> None:
        """Replace this Module's functions in place, and its entry when *entry* is given.

        Passes rewrite a Module tree in place: child Modules are shared by identity.
        """
        self.functions = tuple(functions)
        if entry is not None:
            self.entry = entry

    def __call__(self, *args):
        """Evaluate the declared entry with every parameter supplied."""
        return services.evaluate(self.entry_function(), *args)

    def load(self, resource) -> "LoadedModule":
        """Remember *resource* for this Module and its children.

        Loading is a pure binding operation. Weight values are read by the
        evaluator at the point a function uses a declared ``ConstTensor``.

        [runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)
        """
        return self.load_under(resource, self.target, (), "")

    def load_under(self, resource, tgt, topologies, path: str) -> "LoadedModule":
        tgt = self.target if self.target is not None else tgt
        topologies = self.topologies if self.topologies is not None else topologies
        children = {}
        for key, slot in self.children.items():
            base = resource.subtree(key)
            child_path = f"{path}.{key}" if path else key
            if slot.indexed:
                children[key] = tuple(
                    child.load_under(base.subtree(str(i)), tgt, topologies, f"{child_path}.{i}")
                    for i, child in enumerate(slot.elements)
                )
            else:
                children[key] = slot.elements[0].load_under(base, tgt, topologies, child_path)
        return LoadedModule(
            module=self, resource=resource, children=children,
            target=tgt, topologies=topologies, path=path,
        )

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Prepare canonical weights and write the safetensors checkpoint."""
        services.prepare(self, raw, out_dir, device=device)

@dataclass(frozen=True)
class LoadedModule:
    """An immutable Module binding and the resource from which it reads.

    [runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)
    """

    module: Module
    resource: object
    children: dict[str, LoadedModule | tuple[LoadedModule, ...]] = field(default_factory=dict)
    target: target.Target | None = None
    topologies: tuple[ir_types.Topology, ...] = ()
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
            return replace(
                self, module=replace(module, entry=name, target=self.target, topologies=self.topologies)
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
        return services.evaluate(self, *args)

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


__all__ = ["LoadedModule", "Module", "ModuleFunction", "ModuleList", "resolve_symbol"]
