"""How one call binds its arguments, resolved without asking a public question.

A call into a child Module supplies activations alone and leaves the callee's
``ConstTensor`` parameters to that child's own reading. Which calls those are is
stated, never counted: while a class body is being authored the parser's own
record says so, and afterwards ownership within the walk's scope does. Both the
record and the ownership rule stay here, behind one result.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.visitor_registry.contexts import FunctionScope


def bound_params(callee, *, from_reading: bool) -> tuple:
    """The parameters a call site supplies, in the order it supplies them."""
    if not from_reading:
        return callee.params
    return tuple(param for param in callee.params if not param.is_const)


@dataclass(frozen=True)
class CallBinding:
    """What one call site supplies, and where its callee is read.

    ``params`` are the parameters its arguments bind to, in order. ``scope`` is
    where the callee's body is read, which is what lets a call the callee makes
    in turn be resolved the same way. ``from_reading`` says the parameters left
    out come from a Module reading rather than from this call.
    """

    params: tuple
    scope: FunctionScope | None
    from_reading: bool


def _owned_child(ctx, callee):
    """The child of the scope's function's owner that owns *callee*."""
    scope = getattr(ctx, "scope", None)
    if scope is None or scope.module is None:
        return None
    module_scope = getattr(scope.module, "module_scope", None)
    if module_scope is not None:
        for _name, child in module_scope.items():
            if getattr(child, "owns", lambda *_args, **_kwargs: False)(callee, derived=True):
                return child
        return None
    from tilefoundry.ir.core.module import (  # noqa: PLC0415 — avoid import cycle
        child_module_of,
    )

    return child_module_of(scope.module, scope.function, callee)


def binding_for(callee, call, ctx) -> CallBinding:
    """How a call on *callee* binds its arguments in *ctx*.

    Fails closed: a call whose callee no single child of the caller's owner owns
    binds every declared parameter, so a short argument list is refused rather
    than reinterpreted.
    """
    owner = _owned_child(ctx, callee)
    if owner is not None:
        return CallBinding(
            bound_params(callee, from_reading=True),
            FunctionScope(module=owner, function=callee),
            True,
        )
    scope = getattr(ctx, "scope", None)
    tree = None if scope is None else scope.module
    return CallBinding(
        callee.params,
        None if tree is None else FunctionScope(module=tree, function=callee),
        False,
    )
