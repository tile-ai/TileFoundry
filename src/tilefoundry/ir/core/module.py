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
from tilefoundry.ir.core.expr import Call, Constant, Expr
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.callable_type import callable_type_for
from tilefoundry.ir.types.mesh import Topology
from tilefoundry.ir.types.substitute import canonicalize_topology_dims
from tilefoundry.ir.types.tensor_type import TensorType, TupleType
from tilefoundry.ir.types.utils import types_compatible
from tilefoundry.target.base import Target, target_instance

ModuleFunction = Union[HirFunction, PrimFunction]

_MISSING_PREPARED_WEIGHT = (
    "[runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)"
)


def subtree(root: "Module"):
    """Definitions in DFS order, yielding each shared definition once."""
    seen: set[int] = set()

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        yield node
        for slot in node.children.values():
            for child in slot.elements:
                yield from visit(child)

    yield from visit(root)


def module_functions(root: "Module") -> tuple[ModuleFunction, ...]:
    """Every function definition once, in DFS definition order."""
    seen: set[int] = set()
    found = []
    for node in subtree(root):
        for function in node.functions:
            if id(function) not in seen:
                seen.add(id(function))
                found.append(function)
    return tuple(found)


def owning_module(root: "Module", function: object) -> "Module | None":
    """The one node of *root*'s subtree that owns *function*, else ``None``.

    A Function carries no execution context and one object is reachable from
    more than one program, so the question is asked within a supplied tree. It
    is answered by identity and by recorded origin, never by a name a copy
    keeps. None covers both no owner and more than one: an unanswerable
    question states nothing rather than defaulting to the root.
    """
    found: "Module | None" = None
    for node in subtree(root):
        if node.owns(function, derived=True):
            if found is not None:
                return None
            found = node
    return found


def child_module_of(root: "Module", caller: object, callee: object) -> "Module | None":
    """The direct child of *caller*'s owner that owns *callee*, else ``None``.

    What a collected call no longer carries: the parser's binding record is
    consumed, so which calls supply activations alone is re-derived from
    ownership within *root*. It fails closed -- a same-owner call, and one
    whose ends no single node owns, keeps its exact declared arity.
    """
    if not isinstance(callee, HirFunction) or caller is None:
        return None
    owner = owning_module(root, caller)
    called = owning_module(root, callee)
    if owner is None or called is None or called is owner:
        return None
    return called if any(child is called for slot in owner.children.values() for child in slot.elements) else None


def called_functions(function: HirFunction) -> tuple[HirFunction, ...]:
    """Every HIR Function called directly by *function*, in definition order."""
    from tilefoundry.ir.core.expr import Call  # noqa: PLC0415 -- cycle
    from tilefoundry.ir.visitor import collect_exprs  # noqa: PLC0415 -- cycle

    return tuple(
        expr.target
        for expr in collect_exprs(function.body)
        if isinstance(expr, Call) and isinstance(expr.target, HirFunction)
    )


def reachable_functions(root: HirFunction) -> tuple[HirFunction, ...]:
    """*root* and every HIR Function it calls, callers before callees."""
    reached: list[HirFunction] = []
    seen: set[int] = set()

    def visit(function: HirFunction) -> None:
        if id(function) in seen:
            return
        seen.add(id(function))
        reached.append(function)
        for callee in called_functions(function):
            visit(callee)

    visit(root)
    return tuple(reached)


def _refuse_bare_call(module: "Module", kind: str) -> None:
    """Refuse bare call.

    Refuse a bare call on a *kind* whose *module* has no default entry, naming
    what to call instead.
    """
    if module.entry is not None:
        return
    named = sorted({fn.name for fn in module.functions} | set(module.methods))
    raise TypeError(
        f"{kind} {module.name!r} has no entry, so it has no default step; "
        f"call one by name" + (f" -- {', '.join(named)}" if named else "")
    )


@dataclass(eq=False, init=False)
class ModuleList(Expr):
    """An instance slot holding shared definitions, with a lazily projected type.

    ``ModuleList(definition)`` is a single slot; a count or a tuple makes an
    indexed slot. Its key is assigned when the slot enters ``Module.children``.
    """

    key: str | None
    elements: tuple[Module, ...]
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
        if not elements or not all(isinstance(element, Module) for element in elements):
            raise TypeError("ModuleList requires a Module or a non-empty tuple of Modules")
        self.key = None
        self.elements = elements
        self.indexed = indexed
        self.metadata = ()

    @property
    def type(self):
        signatures = tuple(
            callable_type_for(
                (param for param in entry.params if not param.is_const), entry.return_type
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


def instance_slot(target):
    """The slot and optional index carried by an instance-call target."""
    from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem  # noqa: PLC0415

    if isinstance(target, ModuleList):
        if target.indexed:
            raise ValueError(f"ModuleList slot {target.key!r} must be indexed before calling")
        return target, None
    if (
        isinstance(target, Call)
        and isinstance(target.target, TupleGetItem)
        and isinstance(target.args[0], ModuleList)
    ):
        slot, index = target.args
        if not slot.indexed:
            raise ValueError(f"single slot {slot.key!r} cannot be indexed")
        if isinstance(index, Constant):
            taken = index.value
            if isinstance(taken, bool) or not isinstance(taken, int) or not 0 <= taken < len(slot):
                raise ValueError(f"ModuleList slot {slot.key!r}: index {taken!r} out of range for N={len(slot)}")
            return slot, taken
        if any(element is not slot.elements[0] for element in slot.elements[1:]):
            raise ValueError(f"ModuleList slot {slot.key!r}: variable index requires the same definition")
        return slot, index
    return None


@dataclass(unsafe_hash=True)
class Module:
    """Container of functions + the name of the public entry function.

    A Module is also the execution domain of the functions it owns: it carries
    the hardware ``target`` and the ordered ``topologies`` budget those
    functions run against. ``topologies=None`` inherits from the instance
    stack; ``topologies=()`` declares a topology-free Module.
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
                if child.target is not None:
                    raise ValueError(
                        f"Module {self.name!r}: child module {key!r} declares "
                        f"its own target {child.target!r}; only the root module "
                        "declares a target and children inherit it. The rule: "
                        "tilefoundry spec core-ir target-inheritance"
                    )

    def resolve_target(self, instances: tuple[Module, ...] = ()) -> Target:
        """Resolve the root target in an explicit instance stack."""
        root = instances[0] if instances else self
        if root.target is not None:
            return root.target
        raise ValueError(
            f"Module {root.name!r}: no target is declared; declare a Target "
            "instance on the root module. The rule: "
            "tilefoundry spec core-ir target-inheritance"
        )

    def effective_topologies(self, instances: tuple[Module, ...] = ()) -> tuple[Topology, ...]:
        """The nearest declaration in the explicit instance stack."""
        for node in reversed(instances or (self,)):
            if node.topologies is not None:
                return node.topologies
        return ()

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
        return LoadedModule(
            module=self,
            resource=resource,
            modules=tuple(child.load(resource.subtree(child.name)) for child in self.modules),
        )

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Prepare canonical weights and write the safetensors checkpoint."""
        staged: dict[str, object] = {}
        self._prepare_into(raw, "", staged, device)
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

    def _prepare_into(self, raw, prefix: str, flat: dict, device: str) -> "LoadedModule":
        """Stage this node's canonical weights, children before their owner.

        Converters are HIR bodies on this offline path, making the local
        evaluator import the one intentional Module-to-evaluator execution dependency.
        """
        import torch  # noqa: PLC0415 -- optional runtime dep

        from tilefoundry.runtime.resource import (  # noqa: PLC0415 -- runtime depends on Module
            DictResource,
        )

        children = tuple(
            child._prepare_into(raw.subtree(child.name), f"{prefix}{child.name}.", flat, device)
            for child in self.modules
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
            module=self, resource=DictResource(flat, prefix=prefix), modules=children
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
    modules: tuple["LoadedModule", ...] = field(default_factory=tuple)

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
            return _replace(self, module=_reentered(module, name))
        if len(matches) > 1:
            raise AttributeError(
                f"LoadedModule {self.name!r}: {name!r} resolves to "
                f"{len(matches)} entries; one name must map to one function"
            )
        children = tuple(child for child in self.modules if child.name == name)
        if len(children) == 1:
            return children[0]
        if len(children) > 1:
            raise AttributeError(
                f"LoadedModule {self.name!r}: {name!r} resolves to "
                f"{len(children)} child modules; one name must map to one module"
            )
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


def _reentered(module: Module, entry: str) -> Module:
    """*module* re-entried at *entry*.

    ``replace`` rebuilds the value without its owner backlink, so a child
    selected out of a tree would lose the Target and hierarchy it inherits. The
    copy therefore carries the context it resolved through.
    """
    try:
        target = module.resolve_target()
    except ValueError:
        target = module.target
    return _replace(
        module,
        entry=entry,
        target=target,
        topologies=module.effective_topologies(),
    )


def select(module: Module, path: str) -> Module:
    """The node dotted *path* names below *module*, as a Module.

    Each segment names a child Module, except that the last may instead name one
    of the reached Module's own functions -- which selects that Module re-entried
    at it, so what comes back is always something with a Target and a topology
    hierarchy to be measured against. An empty *path* is *module* itself.

    See [core-ir §1.2](docs/spec/core-ir.md#12-selecting-a-node-by-path).
    """
    selected = module
    segments = path.split(".") if path else []
    if any(not segment for segment in segments):
        raise ValueError(
            f"selector {path!r}: an empty segment names nothing. Dropping it would "
            f"make two different paths mean the same node"
        )
    for index, name in enumerate(segments):
        children = {child.name: child for child in selected.modules}
        if name in children:
            selected = children[name]
            continue
        if index != len(segments) - 1:
            raise ValueError(
                f"selector {path!r}: Module {selected.name!r} has no child module {name!r}"
            )

        selected.lookup(name)
        return _reentered(selected, name)
    return selected


def function_selectors(module: Module, prefix: str = "") -> tuple[tuple[str, HirFunction], ...]:
    """One selector per definition, using its first DFS instance path."""
    seen: set[int] = set()
    functions: set[int] = set()
    found = []

    def visit(node, path):
        if id(node) in seen:
            return
        seen.add(id(node))
        for function in node.functions:
            if isinstance(function, HirFunction) and id(function) not in functions:
                functions.add(id(function))
                found.append((f"{path}{function.name}", function))
        for key, slot in node.children.items():
            for index, child in enumerate(slot.elements):
                suffix = f"{key}.{index}." if slot.indexed else f"{key}."
                visit(child, f"{path}{suffix}")

    visit(module, prefix)
    return tuple(found)


__all__ = [
    "LoadedModule",
    "Module",
    "ModuleList",
    "instance_slot",
    "ModuleFunction",
    "called_functions",
    "function_selectors",
    "module_functions",
    "owning_module",
    "reachable_functions",
    "select",
]
