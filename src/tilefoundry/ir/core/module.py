"""``Module`` — top-level compilation unit: functions, child modules.

``Module`` — top-level compilation unit: functions, child modules, and plain
orchestration methods. See [core-ir §1](docs/spec/core-ir.md#1-module).
"""

from __future__ import annotations

import copy
import functools
import types
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Mapping, Union

from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.target.base import Target, target_instance
from tilefoundry.utils.spec_ref import spec_ref_render

ModuleFunction = Union[HirFunction, PrimFunction]

_MISSING_PREPARED_WEIGHT = (
    "[runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)"
)


def _validate_declared(module_name, source, value, decl_type) -> None:
    """Refuse a checkpoint value that disagrees with its declared tensor type."""
    from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415 -- avoid evaluator cycle

    if tuple(value.shape) != tuple(decl_type.shape):
        raise ValueError(
            f"Module {module_name!r}: {source} has shape "
            f"{tuple(value.shape)}, declared {tuple(decl_type.shape)}"
        )
    expected_dtype = to_torch_dtype(decl_type.dtype)
    if value.dtype != expected_dtype:
        raise ValueError(
            f"Module {module_name!r}: {source} has dtype {value.dtype}, declared "
            f"{decl_type.dtype}; the way out is a weight converter on the model, "
            "not a flag on the read side"
        )


def _refuse_bare_call(module: "Module", kind: str) -> None:
    """Refuse bare call.

    Refuse a bare call on a *kind* whose *module* has neither a ``forward``
    method nor an entry, naming what to call instead.
    """
    if module.entry is not None:
        return
    named = sorted({fn.name for fn in module.functions} | set(module.methods))
    raise TypeError(
        f"{kind} {module.name!r} has no forward method and no entry, so a bare "
        f"call has nothing to run; call one by name" + (f" -- {', '.join(named)}" if named else "")
    )


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
    object.__setattr__(clone, "modules", tuple(_owned_by(node, clone) for node in child.modules))
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

    entry: str | None = None
    modules: tuple["Module", ...] = field(default_factory=tuple)
    target: Target | None = None
    topologies: tuple[Topology, ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    methods: Mapping[str, object] = field(default_factory=dict)

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

        object.__setattr__(self, "modules", tuple(_owned_by(child, self) for child in self.modules))

    def _owner_path(self) -> str:
        """This Module's dotted path from the outermost declared owner."""
        names = [self.name]
        node = getattr(self, "_parent", None)
        while node is not None:
            names.append(node.name)
            node = getattr(node, "_parent", None)
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

        Resolve *name* to a function, a child module, or a bound method. A
        function resolves to a **callable that runs it over every declared
        parameter**, not to the IR node — use ``lookup`` for the node, and
        ``load(resource)`` for a runner that fills constants from bindings.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        matches = tuple(fn for fn in self.functions if fn.name == name)
        if len(matches) == 1:
            return functools.partial(self._run_authored, matches[0])
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
        ``_specialized_from`` chain may lead back to an owned function.

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
        return origin is not None and self.owns(origin)

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

    def load(self, resource) -> "LoadedModule":
        """This Module's constants read from *resource*, as a ``LoadedModule``.

        Returns rather than mutates: the Module is IR and may be read repeatedly.

        [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)
        """
        constants: dict[str, object] = {}
        for name, decl_type in self.weights.items():
            try:
                value = resource.load(name)
            except KeyError as e:
                raise KeyError(
                    f"Module {self.name!r}: missing declared weight {name!r}; prepare produces it "
                    f"({spec_ref_render(_MISSING_PREPARED_WEIGHT)})"
                ) from e
            _validate_declared(self.name, f"weight {name!r}", value, decl_type)
            constants[name] = value
        return LoadedModule(
            module=self,
            constants=constants,
            modules=tuple(child.load(resource.subtree(child.name)) for child in self.modules),
        )

    def _run_authored(self, fn: ModuleFunction, *args):
        """Evaluate *fn* over the arguments given, one per declared parameter.

        Evaluate *fn* over the arguments given, one per declared parameter --
        a ``ConstTensor`` one included, since a Module holds no constants.
        """
        from tilefoundry.evaluator import evaluate  # noqa: PLC0415 -- avoid IR→evaluator cycle

        if len(args) != len(fn.params):
            consts = sum(1 for p in fn.params if p.is_const)
            hint = (
                f"; {consts} of them are ConstTensor, which an authored Module "
                f"does not hold -- pass them here, or call load(resource) and "
                f"run the LoadedModule with activations alone"
                if consts
                else ""
            )
            raise TypeError(
                f"Module {self.name!r}: {fn.name!r} declares {len(fn.params)} "
                f"parameters but got {len(args)}{hint}"
            )
        return evaluate(fn, *args)

    def forward(self, *args):
        """Run this node's step.

        Run this node's step: its ``forward`` orchestration method if it has
        one, else the entry function over the arguments given.
        """
        method = self.methods.get("forward")
        if method is not None:
            return method(self, *args)
        _refuse_bare_call(self, "Module")
        return self._run_authored(self.lookup(self.entry), *args)

    __call__ = forward

    def prepare(self, raw, out_dir: str, *, device: str = "cpu") -> None:
        """Prepare.

        Run every node's per-weight converters over *raw* and write the
        canonical weights to *out_dir* as one safetensors shard plus an index.
        See [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward).
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
            return torch.stack(parts) if parts is not None else raw.load(name)

        for w, decl_type in self.weights.items():
            conv = converter_map.get(w)
            key = prefix + w
            if conv is None:
                value = _fetch(w)
            else:
                value = evaluate(conv, *[_fetch(p.name) for p in conv.params], device=device)
            source = f"converter for weight {key!r}" if conv is not None else f"raw weight {key!r}"
            _validate_declared(self.name, source, value, decl_type)
            flat[key] = value.detach().contiguous().cpu()

        for child in self.modules:
            child._prepare_into(raw.subtree(child.name), f"{prefix}{child.name}.", flat, device)

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
        object.__setattr__(clone, "name", name)
        return clone


@dataclass(frozen=True)
class LoadedModule:
    """An IR ``Module`` together with the constants read for *this* loading.

    Two may stand over one Module without sharing bindings. Attribute access
    mirrors ``Module`` except that functions resolve to runners which fill
    ``ConstTensor`` parameters.

    [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)
    """

    module: Module
    constants: Mapping[str, object]
    modules: tuple["LoadedModule", ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return self.module.name

    def __getattr__(self, name: str):
        """Resolve *name* against the Module.

        Resolve *name* against the Module, with functions and children
        answering from this loading rather than from the IR.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        module = self.module
        matches = tuple(fn for fn in module.functions if fn.name == name)
        if len(matches) == 1:
            return functools.partial(self._run_bound, matches[0])
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

    def _run_bound(self, fn: ModuleFunction, *acts):
        """Run bound.

        Evaluate *fn* over *acts*, its ``ConstTensor`` parameters filled by
        name from these bindings.

        Weights and activations must already agree on one device; nothing moves
        implicitly.

        [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)
        """
        from tilefoundry.evaluator import evaluate  # noqa: PLC0415 -- avoid IR→evaluator cycle

        expected = sum(1 for p in fn.params if not p.is_const)
        if len(acts) != expected:
            raise TypeError(
                f"LoadedModule {self.name!r}: {fn.name!r} takes {expected} "
                f"activation(s) -- its {len(fn.params) - expected} ConstTensor "
                f"parameter(s) come from the bindings -- but got {len(acts)}"
            )
        args = []
        activations = iter(acts)
        for param in fn.params:
            if param.is_const:
                try:
                    args.append(self.constants[param.name])
                except KeyError:
                    raise KeyError(
                        f"LoadedModule {self.name!r}: weight {param.name!r} of "
                        f"{fn.name!r} was not read by load(resource)"
                    ) from None
            else:
                args.append(next(activations))
        return evaluate(fn, *args, device=self._placement(fn, acts))

    def _placement(self, fn: ModuleFunction, acts: tuple) -> str | None:
        """Placement.

        The one device all bound constants and tensor activations agree on, or
        ``None`` when none names one.

        [runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)
        """
        where: dict[str, list[str]] = {}
        for name, value in self.constants.items():
            device = getattr(value, "device", None)
            if device is not None:
                where.setdefault(str(device), []).append(f"weight {name!r}")
        for index, value in enumerate(acts):
            device = getattr(value, "device", None)
            if device is not None:
                where.setdefault(str(device), []).append(f"activation {index}")
        if len(where) > 1:
            spread = "; ".join(
                f"{device}: {', '.join(names)}" for device, names in sorted(where.items())
            )
            raise ValueError(
                f"LoadedModule {self.name!r}: {fn.name!r} was given tensors on "
                f"more than one device -- {spread}. Load the weights and build the "
                f"activations on one device; this runner moves nothing."
            )
        return next(iter(where), None)

    def forward(self, *acts):
        """Run this node's step.

        Run this node's step: its ``forward`` orchestration method if it has
        one, else the entry function over the activations given.
        """
        method = self.module.methods.get("forward")
        if method is not None:
            return method(self, *acts)
        _refuse_bare_call(self.module, "LoadedModule")
        return self._run_bound(self.module.entry_function(), *acts)

    __call__ = forward


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
    "function_selectors",
    "select",
]
