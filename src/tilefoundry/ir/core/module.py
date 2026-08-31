"""``Module`` — top-level compilation unit: functions, child modules.

``Module`` — top-level compilation unit: functions, child modules, and plain
orchestration methods. See [core-ir §1](docs/spec/core-ir.md#1-module).
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Mapping, Union

from tilefoundry import evaluator
from tilefoundry.evaluator.interpreter import _run_bound
from tilefoundry.evaluator.value import tensor_type_of
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.ir.types.substitute import canonicalize_topology_dims
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.ir.types.utils import types_compatible
from tilefoundry.target.base import Target, target_instance

ModuleFunction = Union[HirFunction, PrimFunction]

_MISSING_PREPARED_WEIGHT = (
    "[runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)"
)


def subtree(root: "Module"):
    """*root* and every Module below it, owners before the children they hold."""
    yield root
    for child in root.modules:
        yield from subtree(child)


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
    return called if any(child is called for child in owner.modules) else None


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


def _owned_by(child: "Module", parent: "Module") -> "Module":
    """*child* linked back to *parent* as its owner.

    An unowned child is claimed in place, so the authored tree keeps the very
    objects its class body produced. A child that already belongs to another
    owner is claimed as a copy instead: the two owners may resolve different
    effective context, and re-pointing the original would silently change what
    the first owner's subtree answers.
    """
    owner = child._parent
    if owner is None or owner is parent:
        child._parent = parent
        return child
    clone = copy.copy(child)
    clone.modules = tuple(_owned_by(node, clone) for node in child.modules)
    clone._parent = parent
    return clone


@dataclass(unsafe_hash=True)
class Module:
    """Container of functions + the name of the public entry function.

    A Module is also the execution domain of the functions it owns: it carries
    the hardware ``target`` and the ordered ``topologies`` budget those
    functions run against. ``topologies=None`` declares nothing and inherits
    from the owning Module; ``topologies=()`` declares a topology-free Module.
    """

    name: str = field(hash=False)
    functions: tuple[ModuleFunction, ...]

    entry: str | None = None
    modules: tuple["Module", ...] = field(default_factory=tuple, hash=False)
    target: Target | None = None
    topologies: tuple[Topology, ...] | None = field(default=None, hash=False)
    metadata: dict[str, object] = field(default_factory=dict, hash=False)
    methods: Mapping[str, object] = field(default_factory=dict, hash=False)
    _parent: "Module | None" = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        """Post init.

        Seal each function against further authoring mutation, reject a
        name shared by two of functions / modules / methods, validate the
        declared execution context, and link each child to this owner.
        """
        if self.target is not None:
            target_instance(self.target)
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
        for child in self.modules:
            if child.target is not None:
                raise ValueError(
                    f"Module {self.name!r}: child module {child.name!r} declares "
                    f"its own target {child.target!r}; only the root module "
                    "declares a target and children inherit it. The rule: "
                    "tilefoundry spec core-ir target-inheritance"
                )

        self.modules = tuple(_owned_by(child, self) for child in self.modules)

    def _owner_path(self) -> str:
        """This Module's dotted path from the outermost declared owner."""
        names = [self.name]
        node = self._parent
        while node is not None:
            names.append(node.name)
            node = node._parent
        return ".".join(reversed(names))

    def resolve_target(self) -> Target:
        """Resolve this Module's owner-chain target lookup.

        See docs/spec/core-ir.md § Target inheritance.
        """
        node: "Module | None" = self
        while node is not None:
            if node.target is not None:
                return node.target
            node = getattr(node, "_parent", None)
        raise ValueError(
            f"Module {self._owner_path()!r}: no target is declared by this "
            "module or any of its owners; declare a Target instance on the root "
            "module. The rule: tilefoundry spec core-ir target-inheritance"
        )

    def effective_topologies(self) -> tuple[Topology, ...]:
        """The effective ordered Topology tuple.

        The effective ordered Topology tuple: this Module's declaration,
        else the nearest owner's, else empty at an undeclared root.
        """
        node: "Module | None" = self
        while node is not None:
            if node.topologies is not None:
                return node.topologies
            node = getattr(node, "_parent", None)
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
        """The functions whose name matches, in source order (0 or 1 of them in a verified module).

        The functions whose name matches, in source order (0 or 1 of them in
        a verified module).
        """
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
        matches = self.function_named(self.entry)
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

        [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)
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

    def cloned(self) -> "Module":
        """An independent copy of the IR graph: functions, bodies, children.

        An independent copy of the IR graph: functions, bodies, children, and
        every internal ``Call.target`` redirected to the copy. Immutable outside
        context -- owner, ``target``, ``topologies`` -- stays shared.
        """
        memo: dict[int, object] = {}
        for kept in (getattr(self, "_parent", None), self.target, *(self.topologies or ())):
            if kept is not None:
                memo[id(kept)] = kept
        return copy.deepcopy(self, memo)

    def renamed(self, name: str) -> "Module":
        """An independent copy of this node under a different ``name``."""
        clone = self.cloned()
        clone.name = name
        return clone


@dataclass(frozen=True)
class LoadedModule:
    """An immutable Module binding and the resource from which it reads.

    [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)
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
    """Every HIR function in *module*'s tree, each with the selector naming it.

    Root-relative and dotted, the same paths :func:`select` resolves, so a leaf's
    name is qualified by the children it was reached through: two child Modules
    may each define a ``moe``, and an unqualified name would make them one entry
    in an inventory meant to be countable.

    A ``PrimFunction`` is not one of these: it is an implementation of a function
    rather than a function of the model.

    See [core-ir §1.2](docs/spec/core-ir.md#12-selecting-a-node-by-path).
    """
    found: list[tuple[str, HirFunction]] = [
        (f"{prefix}{function.name}", function)
        for function in module.functions
        if isinstance(function, HirFunction)
    ]
    for child in module.modules:
        found.extend(function_selectors(child, f"{prefix}{child.name}."))
    return tuple(found)


__all__ = [
    "LoadedModule",
    "Module",
    "ModuleFunction",
    "called_functions",
    "function_selectors",
    "owning_module",
    "reachable_functions",
    "select",
]
