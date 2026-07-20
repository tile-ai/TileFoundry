"""``@module(entry=...)`` decorator: collect class-body DSL functions into a Module.

The decorator scans the class for ``@func`` / ``@prim_func`` members (which are
``hir.Function`` / ``tir.PrimFunction`` values), builds a ``Module`` in
definition order, and binds the decorated name to it. The class body is a pure
function container: a non-dunder, non-DSL member is rejected; ``entry`` is an
explicit, required name that must resolve to a collected function. A composed
member may call siblings defined above it (the call lowers to a ``Call``
targeting the sibling); forward references stay unresolved and fail loudly.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tilefoundry import func, module, prim_func
from tilefoundry.dsl import T, Tensor, tf  # noqa: F401 — tf/T used by bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.types.shard import Layout, Mesh, Topology


@module(entry="composed")
class _Demo:
    @func
    def leaf(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    @func
    def composed(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        y = leaf(x, g)
        return tf.add(y, y)


def _iter_calls(root):
    """Yield every ``Call`` node reachable from a HIR expression tree."""
    seen: set[int] = set()
    out: list[Call] = []

    def rec(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Call):
            out.append(node)
        if dataclasses.is_dataclass(node):
            for f in dataclasses.fields(node):
                rec(getattr(node, f.name))
        elif isinstance(node, (list, tuple)):
            for item in node:
                rec(item)

    rec(root)
    return out


def test_module_decorator_returns_ir_module():
    """The decorated name binds directly to the IR Module — no attribute hop."""
    assert isinstance(_Demo, Module)


def test_module_collects_functions_in_order():
    assert _Demo.name == "_Demo"
    assert [fn.name for fn in _Demo.functions] == ["leaf", "composed"]


def test_module_attribute_access_resolves_functions():
    """Attribute access mirrors the model: ``mod.<name>`` returns the function;
    a missing name raises ``AttributeError``."""
    assert _Demo.leaf.name == "leaf"
    assert _Demo.composed.name == "composed"
    with pytest.raises(AttributeError):
        _Demo.not_a_function


def test_module_lookup_still_resolves_each_function():
    assert _Demo.lookup("leaf").name == "leaf"
    assert _Demo.lookup("composed").name == "composed"


def test_attribute_access_ambiguous_name_and_real_fields():
    """A duplicated function name is ambiguous under attribute access (raises),
    real Module fields are never intercepted, and ``function_named`` returns all
    matches — the core-ir §2.1 ambiguity rule."""
    base = _Demo.leaf
    dup_a = dataclasses.replace(base, name="dup")
    dup_b = dataclasses.replace(base, name="dup")
    mod = Module(name="Dup", functions=(dup_a, dup_b), entry="dup")
    assert mod.name == "Dup"
    assert len(mod.functions) == 2
    with pytest.raises(AttributeError):
        mod.dup
    assert len(mod.function_named("dup")) == 2


def test_sibling_call_lowers_to_function_target():
    """The composed kernel's body calls the sibling as a ``Call`` whose target
    is the sibling ``Function`` (not an op)."""
    composed = _Demo.composed
    fn_calls = [c for c in _iter_calls(composed.body) if isinstance(c.target, HirFunction)]
    assert any(c.target.name == "leaf" for c in fn_calls)


def test_composed_module_evaluates():
    torch.manual_seed(0)
    x, g = torch.randn(2, 4), torch.randn(4)
    out = evaluate(_Demo.composed, x, g, device="cpu")
    ref = torch.nn.functional.rms_norm(x, (4,), g)
    torch.testing.assert_close(out, ref + ref, atol=1e-5, rtol=1e-5)


# --- strict-surface validation ------------------------------------------


def test_rejects_undecorated_method():
    with pytest.raises(TypeError, match="only DSL functions"):

        @module(entry="only")
        class _HasMethod:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)

            def helper(self):  # noqa: ANN001 — undecorated stray method
                return None


def test_entry_must_name_a_collected_function():
    with pytest.raises(ValueError, match="names no collected function"):

        @module(entry="missing")
        class _BadEntry:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)


def test_bare_module_without_entry_is_an_error():
    """``@module`` with no explicit entry is rejected — entry is required."""
    with pytest.raises(TypeError):

        @module
        class _NoEntry:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)


def test_rejects_duplicate_aliased_function():
    """A class-body alias of a DSL function collects the same function twice;
    duplicate collected names are rejected (one name maps to one function)."""
    with pytest.raises(ValueError, match="duplicate function name"):

        @module(entry="only")
        class _Alias:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)

            also = only  # alias — same HirFunction object, same name


def test_empty_module_rejected():
    with pytest.raises(TypeError, match="no @func / @prim_func members"):

        @module(entry="x")
        class _Empty:
            pass


def test_forward_reference_sibling_fails_loudly():
    """A method that calls a sibling defined *below* it cannot resolve the
    sibling (only callee-before-caller is supported) and raises rather than
    silently mis-parsing the call."""
    with pytest.raises(VerifyError):

        @module(entry="caller")
        class _Forward:
            @func
            def caller(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                y = callee(x, g)  # noqa: F821 — defined below, unresolved on purpose
                return tf.add(y, y)

            @func
            def callee(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)


def test_prim_func_host_resolves_sibling_device_in_class_body():
    """A ``@prim_func`` cpu host can ``launch`` a sibling cuda device kernel
    defined above it in the same ``@module`` class body — class-local sibling
    resolution works for prim_func, not only @func."""

    @module(entry="host")
    class _Launch:
        @prim_func(target="cuda")
        def dev(a: Tensor[(8,), "f32"]):  # noqa: ARG001
            with Mesh(Topology("thread", 8), Layout(shape=(8,), strides=(1,))) as m:
                T.sync(m)

        @prim_func(target="cpu")
        def host(a: Tensor[(8,), "f32"]):
            launch(dev, a, grid=(1, 1, 1), block=(8, 1, 1))  # noqa: F821

    assert isinstance(_Launch, Module)
    assert [fn.name for fn in _Launch.functions] == ["dev", "host"]
    assert _Launch.entry == "host"
