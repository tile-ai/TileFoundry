"""Single-level nested ``Module``: child-module tree construction, attribute
addressing, the ``@module`` decorator's three member kinds (DSL functions,
child ``Module``s — including a tuple/list of them, and plain-Python
orchestration methods), derived ``weights``, and ``renamed``. Single level
only — a parent with child modules, never grandchildren.
"""
from __future__ import annotations

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.utils import make_tensor_type


@func
def q_proj(x: Tensor[(2, 4), "f32"], w: ConstTensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.matmul(x, w)


@func
def k_proj(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.matmul(x, w)


@func
def decoder_layer(x: Tensor[(2, 4), "f32"], w: ConstTensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.add(q_proj(x, w), q_proj(x, w))


def test_single_level_tree_and_addressing():
    attention = Module(name="attention", functions=(q_proj,), entry="q_proj")
    moe = Module(name="moe", functions=(k_proj,), entry="k_proj")
    root = Module(
        name="layer0",
        functions=(decoder_layer,),
        modules=(attention, moe),
        entry="decoder_layer",
    )
    assert root.attention is attention
    assert root.moe is moe
    assert root.lookup("decoder_layer") is decoder_layer
    # entry resolution only ever considers this module's own functions
    assert root.entry_function() is decoder_layer
    # weights are derived from this node's own functions' ConstTensor params
    assert root.weights == {"w": make_tensor_type((4, 4))}
    assert attention.weights == {"w": make_tensor_type((4, 4))}
    assert moe.weights == {}  # k_proj's `w` is a plain Tensor, not a ConstTensor
    with pytest.raises(AttributeError):
        root.not_a_thing


def test_renamed_returns_a_copy_under_a_new_name():
    """One definition, N addressable instances (e.g. 43 identical layers from
    a factory) — ``renamed`` is a thin ``dataclasses.replace(name=...)``."""
    base = Module(name="attention", functions=(q_proj,), entry="q_proj")
    layer7 = base.renamed("layer7")
    assert layer7.name == "layer7"
    assert layer7.functions is base.functions
    assert layer7.weights == base.weights
    assert base.name == "attention"  # renamed returns a copy; the original is untouched


def test_module_decorator_collects_children():
    @module(entry="leaf")
    class Attention:
        @func
        def leaf(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.matmul(x, w)

    attention_mod = Attention  # prebuilt Module, captured under an unambiguous name

    @module(entry="gate")
    class Moe:
        @func
        def gate(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.matmul(x, w)

    # a tuple of prebuilt Modules is how a factory attaches N identical layers
    moe_layers = tuple(Moe.renamed(f"moe{i}") for i in range(3))

    @module(entry="composed")
    class _Layer:
        Attention = attention_mod  # prebuilt Module assigned as a class attribute
        MoeLayers = moe_layers     # tuple of prebuilt Modules -- child-module member kind

        @module(entry="leaf2")
        class Extra:  # class statement nested directly in the @module body
            @func
            def leaf2(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.matmul(x, w)

        @func
        def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.add(x, x)

        def describe(self):  # plain Python function -- orchestration method
            return f"{self.name}: {len(self.functions)} fn(s), {len(self.modules)} child(ren)"

    assert isinstance(_Layer, Module)
    assert len(_Layer.modules) == 5  # Attention + Extra + the 3 tuple-attached Moe layers
    assert attention_mod in _Layer.modules
    assert all(layer in _Layer.modules for layer in moe_layers)
    assert _Layer.Attention is attention_mod
    assert _Layer.Attention.lookup("leaf").name == "leaf"
    assert _Layer.Extra.lookup("leaf2").name == "leaf2"
    assert _Layer.moe1.name == "moe1"
    assert _Layer.moe1.lookup("gate").name == "gate"
    # plain Python function collected as an orchestration method, bound like
    # an instance method through Module.__getattr__
    assert _Layer.describe() == "_Layer: 1 fn(s), 5 child(ren)"

    with pytest.raises(TypeError, match="only these three member kinds"):

        @module(entry="composed")
        class _BadMember:
            not_a_func = 123

            @func
            def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.add(x, x)
