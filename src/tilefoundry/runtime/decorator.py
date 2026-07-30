"""``@runtime_func`` / ``@runtime_module`` — authoring surface for the runtime
twin of a semantic ``Module``. See docs/spec/runtime.md §1.1.
"""
from __future__ import annotations

import inspect
import types
from typing import Callable

from tilefoundry.ir.core.module import Module, _refuse_bare_call
from tilefoundry.runtime.function import RuntimeFunction
from tilefoundry.runtime.module import RuntimeModule
from tilefoundry.runtime.resource import RuntimeResource

_RUNTIME_FUNC_MARK = "_tilefoundry_runtime_func"

#: Names a twin answers itself, so an authored Module may not also use them:
#: functions and children become instance attributes and would shadow these.
_RESERVED = ("module",)


def runtime_func(fn: Callable) -> Callable:
    """Tag *fn* as a kernel body, signature-identical to the semantic ``@func``
    of the same name. Returns *fn* unwrapped."""
    if not inspect.isfunction(fn):
        raise TypeError(f"runtime_func: expected a plain function, got {type(fn).__name__}")
    setattr(fn, _RUNTIME_FUNC_MARK, True)
    return fn


def _is_runtime_func(value: object) -> bool:
    return inspect.isfunction(value) and getattr(value, _RUNTIME_FUNC_MARK, False)


def _is_kernel_impl(value: object) -> bool:
    """A ``@runtime_func`` method, or a ``RuntimeFunction`` instance standing in
    for one."""
    return _is_runtime_func(value) or isinstance(value, RuntimeFunction)


def _is_child_impl(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, RuntimeModule)


def _make_kernel_caller(instance: RuntimeModule, ir_fn, body: Callable) -> Callable:
    """Bind *body* into a callable taking activations only: ``is_const`` params
    come from *instance*'s loaded weights by name, the rest positionally."""
    is_kernel_obj = isinstance(body, RuntimeFunction)

    def _call(*acts):
        args = []
        remaining = iter(acts)
        for param in ir_fn.params:
            if param.is_const:
                try:
                    args.append(instance._bound[param.name])
                except KeyError:
                    raise KeyError(
                        f"RuntimeModule {instance.name!r}: weight {param.name!r} "
                        f"of {ir_fn.name!r} is not bound; call load(resource) first"
                    ) from None
            else:
                args.append(next(remaining))
        return body(*args) if is_kernel_obj else body(instance, *args)

    return _call


class _Twin(RuntimeModule):
    """Base class of every ``@runtime_module`` result, holding the behaviour that
    is identical for all of them."""

    _ir: Module

    @property
    def module(self) -> Module:
        """The authored ``Module`` this twin was generated from."""
        return self._ir

    def load(self, resource: RuntimeResource) -> None:
        for name in self._ir.weights:
            try:
                self._bound[name] = resource.load(name)
            except KeyError as e:
                raise KeyError(
                    f"RuntimeModule {self.name!r}: missing weight {name!r}"
                ) from e
        for child in self.modules:
            child.load(resource.subtree(child.name))

    def forward(self, *acts):
        """Run the semantic module's own orchestration method against this twin,
        or its entry function if it declares none."""
        method = self._ir.methods.get("forward")
        if method is not None:
            return method(self, *acts)
        # The generated class is named after the authored one, but its base
        # `_Twin` is an implementation detail no message should show.
        _refuse_bare_call(self._ir, "RuntimeModule")
        return getattr(self, self._ir.entry)(*acts)

    def __getattr__(self, name: str):
        # Kernels and children are instance attributes, so anything reaching
        # here is an orchestration method or a typo.
        if name.startswith("_"):
            raise AttributeError(name)
        method = self._ir.methods.get(name)
        if method is not None:
            return types.MethodType(method, self)
        raise AttributeError(
            f"RuntimeModule {self.name!r} has no runtime function, child "
            f"module, or method {name!r}"
        )


def _check_one_to_one(
    cls_name: str, found_label: str, found: set[str], expected_label: str, expected: set[str],
) -> None:
    if found == expected:
        return
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    raise TypeError(
        f"@runtime_module {cls_name!r}: {found_label} {sorted(found)} must "
        f"equal {expected_label} {sorted(expected)}; missing {missing}, extra {extra}"
    )


def runtime_module(sem: Module) -> Callable[[type], type]:
    """Class decorator returning *sem*'s runtime twin: same function names, same
    child names, same entry, validated one-to-one here at decoration time."""

    def _decorate(cls_inner: type) -> type:
        declared = (
            {fn.name for fn in sem.functions}
            | {child.name for child in sem.modules}
            | set(sem.methods)
        )
        reserved = sorted(declared & set(_RESERVED))
        if reserved:
            raise TypeError(
                f"@runtime_module {cls_inner.__name__!r}: Module {sem.name!r} declares "
                f"{reserved}, which a runtime twin reserves; rename it in the authored "
                f"Module"
            )

        raw = {
            name: value
            for name, value in vars(cls_inner).items()
            if not (name.startswith("__") and name.endswith("__"))
        }
        written = sorted(set(raw) & set(_RESERVED))
        if written:
            raise TypeError(
                f"@runtime_module {cls_inner.__name__!r} writes {written}, which a "
                f"runtime twin reserves; the generated class would answer it instead "
                f"of naming the authored Module"
            )

        kernel_names = {name for name, value in raw.items() if _is_kernel_impl(value)}
        child_names = {name for name, value in raw.items() if _is_child_impl(value)}

        _check_one_to_one(
            cls_inner.__name__, "@runtime_func names", kernel_names,
            "semantic function names", {fn.name for fn in sem.functions},
        )
        _check_one_to_one(
            cls_inner.__name__, "child module names", child_names,
            "semantic child module names", {m.name for m in sem.modules},
        )

        def __init__(self, ir: Module | None = None) -> None:
            ir = sem if ir is None else ir
            self._ir = ir
            self._bound: dict[str, object] = {}

            children = []
            for child_ir in ir.modules:
                child_cls = raw.get(child_ir.name)
                if child_cls is None:
                    raise TypeError(
                        f"{cls_inner.__name__}: ir {ir.name!r} has child "
                        f"{child_ir.name!r}, no matching runtime child class"
                    )
                child = child_cls(ir=child_ir)
                setattr(self, child_ir.name, child)
                children.append(child)
            RuntimeModule.__init__(self, name=ir.name, entry=ir.entry, modules=tuple(children))

            for fn in ir.functions:
                body = raw.get(fn.name)
                if body is None:
                    raise TypeError(
                        f"{cls_inner.__name__}: ir {ir.name!r} has function "
                        f"{fn.name!r}, no matching @runtime_func / kernel attribute"
                    )
                setattr(self, fn.name, _make_kernel_caller(self, fn, body))

        namespace = dict(raw)
        namespace["__init__"] = __init__
        namespace["__module__"] = cls_inner.__module__
        if cls_inner.__dict__.get("__doc__") is not None:
            namespace["__doc__"] = cls_inner.__dict__["__doc__"]

        return type(cls_inner.__name__, (_Twin,), namespace)

    return _decorate


__all__ = ["runtime_func", "runtime_module"]
