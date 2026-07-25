"""``@runtime_func`` / ``@runtime_module`` -- the runtime-side authoring
surface for a semantic ``Module``'s twin (the nested-module design's Phase
2; the semantic side is ``tilefoundry.ir.core.module.Module`` /
``tilefoundry.module.module``).

``@runtime_func`` tags a method as a kernel body. ``@runtime_module(sem)``
takes the semantic ``Module`` instance and returns a class: its instances
are *sem*'s runtime twin -- same function names, same child tree, same
entry. Orchestration methods (``forward`` / ``init_caches`` / ...) are
reused verbatim from ``sem.methods``, never rewritten here.
"""
from __future__ import annotations

import inspect
import types
from typing import Callable

from tilefoundry.ir.core.module import Module
from tilefoundry.runtime.function import RuntimeFunction
from tilefoundry.runtime.module import RuntimeModule
from tilefoundry.runtime.resource import RuntimeResource

# Marker attribute ``@runtime_func`` stamps on the tagged function; mirrors
# the ``_sealed`` / ``_bound`` private-attribute convention used elsewhere
# for authoring-time-only bookkeeping.
_RUNTIME_FUNC_MARK = "_tilefoundry_runtime_func"


def runtime_func(fn: Callable) -> Callable:
    """Tag *fn* as a kernel body: same call signature as the semantic
    ``@func`` of the same name, weight params included. ``@runtime_module``
    discovers it by this tag and fills weights/activations at call time;
    *fn* is returned unwrapped -- this decorator never wraps the body.
    """
    if not inspect.isfunction(fn):
        raise TypeError(f"runtime_func: expected a plain function, got {type(fn).__name__}")
    setattr(fn, _RUNTIME_FUNC_MARK, True)
    return fn


def _is_runtime_func(value: object) -> bool:
    return inspect.isfunction(value) and getattr(value, _RUNTIME_FUNC_MARK, False)


def _is_kernel_impl(value: object) -> bool:
    """A class attribute implementing a semantic function: a
    ``@runtime_func`` method, or a ``RuntimeFunction`` instance (a heavy
    kernel that owns its own compilation state, built once and shared)."""
    return _is_runtime_func(value) or isinstance(value, RuntimeFunction)


def _is_child_impl(value: object) -> bool:
    """A class attribute implementing a child module: a ``RuntimeModule``
    subclass (typically another ``@runtime_module`` result), constructed
    per-parent against the semantic child of the same name."""
    return isinstance(value, type) and issubclass(value, RuntimeModule)


def _make_kernel_caller(instance: RuntimeModule, ir_fn, body: Callable) -> Callable:
    """Bind *body* into a name-filling callable: walk *ir_fn*'s params in
    order, take ``is_const`` params from *instance*'s loaded weights by
    name and every other param positionally from the call's activations --
    the runtime mirror of ``Module.forward``'s own const-filling loop.
    """
    is_kernel_obj = isinstance(body, RuntimeFunction)

    def _call(*acts):
        args = []
        remaining = iter(acts)
        for param in ir_fn.params:
            if param.is_const:
                try:
                    args.append(instance._bound[param.name])
                except KeyError:
                    # Word-for-word the semantic side's message (``Module._run``):
                    # the twin fails the same way for the same reason.
                    raise KeyError(
                        f"RuntimeModule {instance.name!r}: weight {param.name!r} "
                        f"of {ir_fn.name!r} is not bound; call load(resource) first"
                    ) from None
            else:
                args.append(next(remaining))
        return body(*args) if is_kernel_obj else body(instance, *args)

    return _call


class _Twin(RuntimeModule):
    """Base of every ``@runtime_module`` result: the behaviours that read only
    ``self._ir`` / ``self.modules`` / ``self._bound``, so they are defined once
    here rather than rebuilt as closures per decorated class (docs/spec/
    runtime.md §1.1). An authored ``forward`` shadows this one by ordinary MRO.
    """

    _ir: Module

    def load(self, resource: RuntimeResource) -> None:
        """Bind this node's weights by name from *resource*, then recurse into
        each child under ``resource.subtree(child.name)`` — the runtime mirror
        of ``Module.load``."""
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
        """Run the semantic module's own orchestration method against this
        twin (``self.<fn>`` / ``self.<child>`` resolve to the runtime versions),
        falling back to the entry function for a node that declares none."""
        method = self._ir.methods.get("forward")
        if method is not None:
            return method(self, *acts)
        return getattr(self, self._ir.entry)(*acts)

    def __getattr__(self, name: str):
        """Fallback — kernels and children are instance attributes set by
        ``__init__``, so reaching here means *name* is (or should be) one of
        ``ir.methods``' orchestration methods, bound to this twin."""
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
    """Raise ``TypeError`` naming both sets plus the missing/extra names
    unless *found* == *expected* exactly."""
    if found == expected:
        return
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    raise TypeError(
        f"@runtime_module {cls_name!r}: {found_label} {sorted(found)} must "
        f"equal {expected_label} {sorted(expected)}; missing {missing}, extra {extra}"
    )


def runtime_module(sem: Module) -> Callable[[type], type]:
    """Class decorator: validate *cls_inner* one-to-one against *sem* (every
    ``sem.functions`` name has exactly one ``@runtime_func`` / kernel
    attribute; every ``sem.modules`` name has exactly one child-module class
    attribute) and return a ``RuntimeModule`` subclass whose instances are
    *sem*'s runtime twin.
    """

    def _decorate(cls_inner: type) -> type:
        raw = {
            name: value
            for name, value in vars(cls_inner).items()
            if not (name.startswith("__") and name.endswith("__"))
        }

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

        # ``raw`` carries the authored body through unchanged: kernels (shadowed
        # per instance by their name-filling callers), plain helper methods, and
        # -- if the author wrote one -- a ``forward`` that shadows
        # ``_Twin.forward`` by ordinary MRO.
        namespace = dict(raw)
        namespace["__init__"] = __init__
        namespace["__module__"] = cls_inner.__module__
        if cls_inner.__dict__.get("__doc__") is not None:
            namespace["__doc__"] = cls_inner.__dict__["__doc__"]

        return type(cls_inner.__name__, (_Twin,), namespace)

    return _decorate


__all__ = ["runtime_func", "runtime_module"]
