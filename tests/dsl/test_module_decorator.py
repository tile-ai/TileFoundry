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

from tilefoundry import func, module, prim_func
from tilefoundry.dsl import T, Tensor, tf  # noqa: F401 — tf/T used by bodies
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.module import Module
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


def test_module_collects_functions_in_order_and_resolves_them_by_name():
    """Members land on the Module in definition order and each collected name
    resolves through ``lookup`` (the IR-node path — bare attribute access
    instead returns a runnable callable, see ``ir.core.module.Module``); a name
    the class never declared raises ``AttributeError``."""
    assert _Demo.name == "_Demo"
    assert [fn.name for fn in _Demo.functions] == ["leaf", "composed"]
    assert _Demo.lookup("leaf").name == "leaf"
    assert _Demo.lookup("composed").name == "composed"
    with pytest.raises(AttributeError):
        _Demo.not_a_function


def test_attribute_access_ambiguous_name_and_real_fields():
    """A duplicated function name is ambiguous under attribute access (raises),
    real Module fields are never intercepted, and ``function_named`` returns all
    matches — the core-ir §2.1 ambiguity rule."""
    base = _Demo.lookup("leaf")
    dup_a = dataclasses.replace(base, name="dup")
    dup_b = dataclasses.replace(base, name="dup")
    mod = Module(name="Dup", functions=(dup_a, dup_b), entry="dup")
    assert mod.name == "Dup"
    assert len(mod.functions) == 2
    with pytest.raises(AttributeError):
        mod.dup
    assert len(mod.function_named("dup")) == 2


def test_collects_orchestration_method_and_rejects_other_members():
    """A plain Python method is the third member kind — an orchestration method,
    bound on the resulting Module. A member that is none of the three kinds
    (DSL function / child Module / plain function) is still rejected."""

    @module(entry="only")
    class _WithMethod:
        @func
        def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.rms_norm(x, g)

        def describe(self):  # noqa: ANN001 — orchestration method, bound on the Module
            return f"{self.name}/{self.entry}"

    assert _WithMethod.describe() == "_WithMethod/only"

    with pytest.raises(TypeError, match="only these three member kinds"):

        @module(entry="only")
        class _BadMember:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)

            budget = 3  # not a DSL function, a child Module, or a plain function


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
