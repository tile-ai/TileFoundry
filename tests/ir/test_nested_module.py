"""Single-level nested ``Module``: child-module tree construction, attribute
addressing, weights/states declaration slots, and the ``@module`` decorator's
child-module collection (a nested ``@module`` class + a prebuilt ``Module``
assigned as a class attribute). Single level only — a parent with child
modules, never grandchildren.
"""
from __future__ import annotations

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types import DType, TensorType


@func
def q_proj(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.matmul(x, w)


@func
def k_proj(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.matmul(x, w)


@func
def decoder_layer(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.add(q_proj(x, w), q_proj(x, w))


def _tt(shape: tuple[int, ...]) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def test_single_level_tree_and_addressing():
    attention = Module(name="attention", functions=(q_proj,), entry="q_proj")
    moe = Module(name="moe", functions=(k_proj,), entry="k_proj")
    root = Module(
        name="layer0",
        functions=(decoder_layer,),
        modules=(attention, moe),
        entry="decoder_layer",
        weights={"w_head": _tt((4, 4))},
        states={"kv": _tt((2, 4))},
    )
    assert root.attention is attention
    assert root.moe is moe
    assert root.decoder_layer is decoder_layer
    # entry resolution only ever considers this module's own functions
    assert root.entry_function() is decoder_layer
    assert root.weights == {"w_head": _tt((4, 4))}
    assert root.states == {"kv": _tt((2, 4))}
    with pytest.raises(AttributeError):
        root.not_a_thing


def test_name_collisions_rejected():
    with pytest.raises(ValueError, match="used by both a function and a child module"):
        Module(
            name="layer0",
            functions=(q_proj,),
            modules=(Module(name="q_proj", functions=(k_proj,), entry="k_proj"),),
            entry="q_proj",
        )


def test_module_decorator_collects_children():
    @module(entry="leaf")
    class Attention:
        @func
        def leaf(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.matmul(x, w)

    attention_mod = Attention  # prebuilt Module, captured under an unambiguous name

    @module(entry="composed")
    class _Layer:
        Attention = attention_mod  # prebuilt Module assigned as a class attribute

        @module(entry="gate")
        class Moe:  # class statement nested directly in the @module body
            @func
            def gate(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.matmul(x, w)

        @func
        def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.add(x, x)

    assert isinstance(_Layer, Module)
    assert len(_Layer.modules) == 2
    assert attention_mod in _Layer.modules
    assert _Layer.Moe in _Layer.modules
    assert _Layer.Attention is attention_mod
    assert _Layer.Attention.leaf.name == "leaf"
    assert _Layer.Moe.gate.name == "gate"

    with pytest.raises(ValueError, match="duplicate child module name"):

        @module(entry="composed")
        class _Dup:
            Attention = attention_mod
            AttentionAgain = attention_mod  # alias — same Module object, same name

            @func
            def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.add(x, x)

    with pytest.raises(TypeError, match="only DSL functions"):

        @module(entry="composed")
        class _BadMember:
            not_a_func = 123

            @func
            def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.add(x, x)
