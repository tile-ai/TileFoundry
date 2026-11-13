"""``@runtime.func`` / ``@runtime.module``.

``@runtime.func`` / ``@runtime.module`` — authoring surface for the runtime
twin of a semantic ``Module``. See [runtime §1.1](docs/spec/runtime.md#11-runtimemodulepy).
"""

from __future__ import annotations

import inspect
import types
import weakref
from typing import Callable

from tilefoundry import core, utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import types as ir_types
from tilefoundry.ir import utils as irutils

from .function import Function
from .module import Module
from .resource import Resource

RESERVED = (
    "module", "bound", "resource", "placement", "setup", "to_local", "from_local", "weight_of",
)


class Kernels:
    """The plain functions tagged as kernel bodies by ``func``."""

    def __init__(self) -> None:
        self.tagged: weakref.WeakSet = weakref.WeakSet()

    def tag(self, fn: Callable) -> None:
        """Remember *fn* as a kernel body."""
        self.tagged.add(fn)

    def holds(self, value: object) -> bool:
        """Whether *value* is a function tagged by ``tag``."""
        return inspect.isfunction(value) and value in self.tagged


kernels = Kernels()


def func(fn: Callable) -> Callable:
    """Tag *fn* as a kernel body, signature-identical to the semantic ``@func`` of the same name.

    Tag *fn* as a kernel body, signature-identical to the semantic ``@func``
    of the same name. Returns *fn* unwrapped.
    """
    if not inspect.isfunction(fn):
        raise TypeError(f"runtime.func: expected a plain function, got {type(fn).__name__}")
    kernels.tag(fn)
    return fn


def is_kernel_impl(value: object) -> bool:
    """A ``@runtime.func`` method, or a ``Function`` instance standing in for one.

    A ``@runtime.func`` method, or a ``Function`` instance standing in
    for one.
    """
    return kernels.holds(value) or isinstance(value, Function)


def is_child_impl(value: object) -> bool:
    if isinstance(value, ir_core.ModuleList):
        return all(is_child_impl(element) for element in value.elements)
    return isinstance(value, type) and issubclass(value, Module)


def make_kernel_caller(instance: Module, ir_fn, body: Callable) -> Callable:
    """Bind *body* into a callable taking activations only, converting both ways.

    The way in is a ``forward_pre_hook`` and the way out a ``forward_hook``:
    each ``ShardTensor`` argument is projected to this program's slice and each
    ``is_const`` param comes from *instance*'s loaded weights by name, already
    projected, so a body -- a Python reference or a launched kernel alike --
    only sees ``torch.Tensor`` and never the mesh it came out of. The way out
    wraps again only where the declared return type carries a ``ShardLayout``.
    """
    is_kernel_obj = isinstance(body, Function)

    def call(*acts):
        args = []
        remaining = iter(acts)
        for param in ir_fn.params:
            if param.is_const:
                args.append(instance.weight_of(param.name))
            else:
                args.append(instance.to_local(next(remaining)))
        out = body(*args) if is_kernel_obj else body(instance, *args)
        return instance.from_local(out, ir_fn.return_type)

    return call


class Twin(Module):
    """Represent Twin.

    Base class of every ``@runtime.module`` result, holding the behaviour that
    is identical for all of them.
    """

    module: ir_core.Module
    """The authored ``Module`` this twin was generated from."""

    def setup(self, sem: ir_core.Module, raw: dict, class_name: str, ir: ir_core.Module | None) -> None:
        """Build the children and kernel callers of *ir*, a ``Module`` shaped like *sem*.

        *raw* holds the decorated class's kernel bodies and child classes by name.
        """
        ir = sem if ir is None else ir
        self.module = ir
        self.bound: dict[tuple, object] = {}
        self.resource: Resource | None = None
        self.placement: ir_types.Placement | None = None

        children = {}
        for key, slot in ir.children.items():
            child_impl = raw.get(key)
            if child_impl is None:
                raise TypeError(
                    f"{class_name}: ir {ir.name!r} has child "
                    f"{key!r}, no matching runtime child class"
                )
            definitions = child_impl.elements if isinstance(child_impl, ir_core.ModuleList) else (child_impl,)
            instances = tuple(
                child_cls(ir=child_ir)
                for child_cls, child_ir in zip(definitions, ir_core.instances(slot), strict=True)
            )
            child = instances if isinstance(slot, tuple) else instances[0]
            setattr(self, key, child)
            children[key] = child
        Module.__init__(self, name=ir.name, entry=ir.entry, children=children)

        for fn in ir.functions:
            body = raw.get(fn.name)
            if body is None:
                raise TypeError(
                    f"{class_name}: ir {ir.name!r} has function "
                    f"{fn.name!r}, no matching @runtime.func / kernel attribute"
                )
            setattr(self, fn.name, make_kernel_caller(self, fn, body))

    def load(
        self, resource: Resource, *, placement: "ir_types.Placement | None" = None
    ) -> None:
        self.resource = resource
        self.placement = placement
        self.bound.clear()
        super().load(resource, placement=placement)

    def to_local(self, value):
        """Into a body: a ``ShardTensor`` projected to this program's slice.

        Anything the mesh never spread is already what this program holds, so
        it passes through -- the choice is made once here rather than at each
        body. Named as ``DTensor.to_local``, which answers the same question.
        """
        if isinstance(value, core.ShardTensor):
            return value.to_local(self.placement)
        return value

    def from_local(self, value, declared):
        """Out of a body: a returned tensor as a ``ShardTensor`` when it is one.

        The declared return type decides, not the value: what a body hands
        back is a bare ``torch.Tensor`` with nothing on it saying whether it is
        one program's slice or the whole. Only a return type the mesh divides
        is a distributed value; anything else is one tensor and stays one. A
        distributed value is returned scattered, holding only this program's
        piece under the declared global type, so it is never narrowed twice.
        """
        if not isinstance(value, utils.torch.Tensor) or not isinstance(declared, ir_types.TensorType):
            return value
        if ir_types.shard_layout_of(declared.layout) is None:
            return value
        return core.ShardTensor(declared, ((self.placement, value),))

    def weight_of(self, name: str):
        """Read one declared weight, narrow it to this program, and memoize it.

        A narrowed weight is copied out of the read it came from, so the whole
        canonical tensor is released once the shard it contributed is held;
        only a weight the mesh does not divide is kept as the read itself.
        """
        key = name
        try:
            return self.bound[key]
        except KeyError:
            pass
        if self.resource is None:
            raise KeyError(
                f"runtime.Module {self.name!r}: weight {name!r} "
                f"is not bound; call load(resource) first"
            )
        try:
            value = self.resource.load(name)
        except KeyError as error:
            raise KeyError(
                f"runtime.Module {self.name!r}: missing weight {name!r}"
            ) from error
        declared = self.module.weights[name]
        actual = evaluator.tensor_type_of(value, like=declared)
        if actual.shape != declared.shape:
            raise ValueError(
                f"runtime.Module {self.name!r}: weight {name!r} has shape "
                f"{actual.shape}, declared {declared.shape}"
            )
        if not irutils.types_compatible(declared, actual):
            raise ValueError(
                f"runtime.Module {self.name!r}: weight {name!r} has dtype "
                f"{value.dtype}, declared {declared.dtype}; the way out is a weight "
                "converter on the model, not a flag on the read side"
            )
        local = core.ShardTensor(declared, value).to_local(self.placement)
        self.bound[key] = local if local is value else local.clone()
        return self.bound[key]

    def forward(self, *acts):
        """Forward.

        Run the semantic module's own orchestration method against this twin,
        or its entry function if it declares none.
        """
        method = self.module.methods.get("forward")
        if method is not None:
            return method(self, *acts)


        if self.module.entry is None:
            named = sorted({fn.name for fn in self.module.functions} | set(self.module.methods))
            raise TypeError(
                f"runtime.Module {self.module.name!r} has no entry, so it has no default step; "
                f"call one by name" + (f" -- {', '.join(named)}" if named else "")
            )
        return getattr(self, self.module.entry)(*acts)

    def __getattr__(self, name: str):


        if name.startswith("_") or self.module is None:
            raise AttributeError(name)
        method = self.module.methods.get(name)
        if method is not None:
            return types.MethodType(method, self)
        raise AttributeError(
            f"runtime.Module {self.name!r} has no runtime function, child "
            f"module, or method {name!r}"
        )


def check_one_to_one(
    cls_name: str, found_label: str, found: set[str], expected_label: str, expected: set[str],
    *, detail: str = "",
) -> None:
    if found == expected:
        return
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    raise TypeError(
        f"@runtime.module {cls_name!r}: {found_label} {sorted(found)} must "
        f"equal {expected_label} {sorted(expected)}; missing {missing}, extra {extra}{detail}"
    )


def module(sem: ir_core.Module) -> Callable[[type], type]:
    """Class decorator returning *sem*'s runtime twin.

    Class decorator returning *sem*'s runtime twin: same function names, same
    child names, same entry, validated one-to-one here at decoration time.
    """

    def decorate(cls_inner: type) -> type:
        declared = (
            {fn.name for fn in sem.functions}
            | set(sem.children)
            | set(sem.methods)
        )
        reserved = sorted(declared & set(RESERVED))
        if reserved:
            raise TypeError(
                f"@runtime.module {cls_inner.__name__!r}: Module {sem.name!r} declares "
                f"{reserved}, which a runtime twin reserves; rename it in the authored "
                f"Module"
            )

        raw = {
            name: value
            for name, value in vars(cls_inner).items()
            if not (name.startswith("__") and name.endswith("__"))
        }
        written = sorted(set(raw) & set(RESERVED))
        if written:
            raise TypeError(
                f"@runtime.module {cls_inner.__name__!r} writes {written}, which a "
                f"runtime twin reserves; the generated class would answer it instead "
                f"of naming the authored Module"
            )

        kernel_names = {name for name, value in raw.items() if is_kernel_impl(value)}
        child_names = {name for name, value in raw.items() if is_child_impl(value)}

        check_one_to_one(
            cls_inner.__name__, "@runtime.func names", kernel_names,
            "semantic function names", {fn.name for fn in sem.functions},
        )
        semantic_counts = {key: len(ir_core.instances(slot)) for key, slot in sem.children.items()}
        runtime_counts = {
            key: len(raw[key]) if isinstance(raw[key], ir_core.ModuleList) else 1
            for key in child_names
        }
        check_one_to_one(
            cls_inner.__name__, "child module names", child_names,
            "semantic child module names", set(sem.children),
            detail=f"; semantic N={semantic_counts}, runtime N={runtime_counts}",
        )

        for key, slot in sem.children.items():
            child_impl = raw[key]
            runtime_slot = child_impl if isinstance(child_impl, ir_core.ModuleList) else ir_core.ModuleList(child_impl)
            if not all(
                isinstance(element, type) and issubclass(element, Module)
                for element in runtime_slot.elements
            ):
                raise TypeError("ModuleList requires Modules or runtime module classes")
            indexed, count = isinstance(slot, tuple), len(ir_core.instances(slot))
            if runtime_slot.indexed != indexed or len(runtime_slot) != count:
                raise TypeError(
                    f"@runtime.module {cls_inner.__name__!r}: slot {key!r}: "
                    f"semantic N={count}, indexed={indexed}; "
                    f"runtime N={len(runtime_slot)}, indexed={runtime_slot.indexed}"
                )

        def initialize(self, ir: ir_core.Module | None = None) -> None:
            self.setup(sem, raw, cls_inner.__name__, ir)

        namespace = dict(raw)
        namespace["__init__"] = initialize
        namespace["__module__"] = cls_inner.__module__
        if cls_inner.__dict__.get("__doc__") is not None:
            namespace["__doc__"] = cls_inner.__dict__["__doc__"]

        return type(cls_inner.__name__, (Twin,), namespace)

    return decorate


__all__ = ["func", "module"]
