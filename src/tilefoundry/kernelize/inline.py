"""``inline_calls(hir) -> Function`` -- flatten nested ``Call(target=Function)``
sites into one flat SSA-DAG of primitive-op Calls, the hard prerequisite for
``kernelize.extract`` (which rejects a nested Function call -- see its
``_postorder`` walk) to process a whole composed layer (e.g. a decoder layer
built from ``self_attention`` + ``mlp`` ``@func``s that themselves nest
further ``@func`` calls, like ``input_rms_norm``).

Pure structural substitution, mirroring the evaluator's own inlining
(``evaluator.interpreter.Evaluator._call_function``): a callee's ``params``
bind to the (already-rewritten) call args, then its ``body`` is copied under
that substitution, recursively flattening any nested calls found inside. No
evaluation, no optimization -- every copied node keeps its original ``.type``
(identical by construction: the parser's ``elaborate()`` already guarantees
``call.type == call.target.body.type`` for the concrete instance a call site
carries, and substitution never changes a node's type).
"""
from __future__ import annotations

import dataclasses
import itertools

from tilefoundry.ir.core import BindingMetadata, Call, Expr, Var, binding_name, replace_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.visitor import ExprMutator


class InlineError(NotImplementedError):
    """A construct ``inline_calls`` does not (yet) support."""


class _Inliner(ExprMutator):
    """Flatten one callee body under ``subst`` (``id(Var) -> bound Expr``),
    prefixing its own bindings with ``prefix``. One fresh instance (fresh
    memo) per call site, so DAG-sharing is preserved within one inlined body
    but never falsely shared across two call sites of the same callee."""

    def __init__(self, subst: dict[int, Expr], prefix: str, counter: itertools.count) -> None:
        self._subst = subst
        self._prefix = prefix
        self._counter = counter
        self._memo: dict[int, Expr] = {}

    def visit(self, expr: Expr) -> Expr:
        cached = self._memo.get(id(expr))
        if cached is not None:
            return cached
        result = self._visit_uncached(expr)
        self._memo[id(expr)] = result
        return result

    def _visit_uncached(self, expr: Expr) -> Expr:
        if isinstance(expr, Var):
            return self._subst.get(id(expr), expr)
        if isinstance(expr, Call):
            return self._visit_call(expr)
        return self.generic_visit(expr)  # Tuple / GridRegionExpr / Constant

    def _visit_call(self, call: Call) -> Expr:
        new_args = tuple(self.visit(a) for a in call.args)
        if isinstance(call.target, Function):
            return _inline_call(call.target, new_args, self._counter)
        same = all(na is oa for na, oa in zip(new_args, call.args))
        rebuilt = call if same else dataclasses.replace(call, args=new_args)
        return _rename(rebuilt, self._prefix)


def _rename(expr: Expr, prefix: str) -> Expr:
    """Prefix ``expr``'s authored bind name; identity-preserving no-op when
    ``prefix`` is empty (top-level scope) or ``expr`` carries no bind name."""
    if not prefix:
        return expr
    name = binding_name(expr)
    if name is None:
        return expr
    return replace_metadata(expr, BindingMetadata(prefix + name))


def _inline_call(callee: Function, args: tuple[Expr, ...], counter: itertools.count) -> Expr:
    """Flatten one ``Call(target=callee)`` into a fresh copy of ``callee.body`` under ``args``."""
    if callee.variants or callee.body is None:
        raise InlineError(
            f"kernelize.inline_calls: {callee.name!r} is a dispatch prototype "
            "(has variants / no body) -- variant selection needs a concrete "
            "runtime shape, which this structural pass does not have"
        )
    if len(args) != len(callee.params):
        raise InlineError(
            f"kernelize.inline_calls: call to {callee.name!r} expects "
            f"{len(callee.params)} args, got {len(args)}"
        )
    subst = {id(p): a for p, a in zip(callee.params, args)}
    prefix = f"{callee.name}{next(counter)}_"
    return _Inliner(subst, prefix, counter).visit(callee.body)


def inline_calls(hir: Function) -> Function:
    """Return ``hir`` with every nested ``Call(target=Function)`` in its body flattened away."""
    if hir.body is None:
        return hir
    new_body = _Inliner({}, "", itertools.count(1)).visit(hir.body)
    if new_body is hir.body:
        return hir
    return dataclasses.replace(hir, body=new_body)


__all__ = ["inline_calls", "InlineError"]
