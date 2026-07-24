"""Nested ``Module`` (M0 of docs/plans/agent-kernel-loop/P0a-tonight-nested-module-e2e.md):
``modules`` collects child ``Module``s so a tree addresses like the model it
mirrors (``root.layer0.attention.q_proj``), and ``post_init`` is a module's own
weight-conversion hook. This file only exercises tree construction / attribute
addressing / printing and the function-vs-module name-collision guard; M1
exercises ``post_init`` execution against the evaluator.
"""
from __future__ import annotations

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.core.module import Module


@func
def q_proj(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.matmul(x, w)


@func
def attention_entry(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return q_proj(x, w)


@func
def gate(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.matmul(x, w)


@func
def moe_entry(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return gate(x, w)


@func
def decoder_layer(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
    return tf.add(attention_entry(x, w), moe_entry(x, w))


def _build_tree() -> Module:
    """``layer0`` module: (attention, moe) children + its own composed entry."""
    attention = Module(name="attention", functions=(q_proj, attention_entry), entry="attention_entry")
    moe = Module(name="moe", functions=(gate, moe_entry), entry="moe_entry")
    return Module(
        name="layer0", functions=(decoder_layer,), modules=(attention, moe), entry="decoder_layer",
    )


class TestNestedModuleConstruction:
    def test_three_level_tree_constructs(self) -> None:
        attention = Module(name="attention", functions=(q_proj,), entry="q_proj")
        moe = Module(name="moe", functions=(gate,), entry="gate")
        layer0 = Module(
            name="layer0", functions=(decoder_layer,), modules=(attention, moe), entry="decoder_layer",
        )
        root = Module(name="root", functions=(), modules=(layer0,), entry="decoder_layer")
        assert root.modules == (layer0,)
        assert layer0.modules == (attention, moe)

    def test_addresses_by_attribute_path(self) -> None:
        layer0 = _build_tree()
        root = Module(name="root", functions=(), modules=(layer0,), entry="decoder_layer")
        assert root.layer0 is layer0
        assert root.layer0.attention.q_proj is q_proj
        assert root.layer0.moe.gate is gate
        # existing flat function resolution is untouched
        assert root.layer0.decoder_layer is decoder_layer

    def test_missing_name_raises_attribute_error(self) -> None:
        layer0 = _build_tree()
        with pytest.raises(AttributeError):
            layer0.attention.not_a_thing
        with pytest.raises(AttributeError):
            layer0.not_a_child_module

    def test_printing_does_not_crash(self) -> None:
        layer0 = _build_tree()
        root = Module(name="root", functions=(), modules=(layer0,), entry="decoder_layer")
        text = repr(root)
        assert "root" in text
        assert "layer0" in text

    def test_function_named_module_name_collision_rejected(self) -> None:
        """A function whose own name equals a child module's name is
        ambiguous under attribute addressing — rejected at construction."""
        with pytest.raises(ValueError, match="used by both a function and a child module"):
            Module(
                name="layer0",
                functions=(q_proj,),  # q_proj.name == "q_proj"; no clash here...
                modules=(Module(name="q_proj", functions=(gate,), entry="gate"),),  # ...until this
                entry="q_proj",
            )

    def test_entry_resolution_unaffected_by_modules(self) -> None:
        """entry_function() / lookup() only ever resolve this module's own
        functions — nesting does not change entry semantics."""
        layer0 = _build_tree()
        assert layer0.entry_function() is decoder_layer
        assert layer0.lookup("decoder_layer") is decoder_layer
        with pytest.raises(ValueError):
            layer0.lookup("q_proj")  # belongs to the child module, not layer0


class TestModuleDecoratorNesting:
    def test_prebuilt_module_assigned_as_class_attribute(self) -> None:
        """The "equivalent Python API" half of M0's either/or: a Module built
        ahead of time and wired in by plain class-body assignment. Attribute
        addressing matches by the child's own identity name (like functions
        are matched by ``fn.name``, not by whichever class-body key holds
        them), so the binding name must equal ``Attention.name``."""

        @module(entry="leaf")
        class Attention:
            @func
            def leaf(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.matmul(x, w)

        attention_mod = Attention  # capture under an unambiguous outer name

        @module(entry="composed")
        class _Layer:
            Attention = attention_mod  # re-bound under the same key as its own .name

            @func
            def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.add(x, x)

        assert isinstance(_Layer, Module)
        assert _Layer.modules == (Attention,)
        assert _Layer.Attention is attention_mod
        assert _Layer.Attention.leaf.name == "leaf"

    def test_directly_nested_class_statement(self) -> None:
        """A ``class`` defined directly inside the ``@module`` class body
        (rather than assigned from an existing name) is collected the same
        way — the class-body member is a ``Module`` either way."""

        @module(entry="composed")
        class _Layer:
            @module(entry="leaf")
            class Attention:
                @func
                def leaf(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                    return tf.matmul(x, w)

            @func
            def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.add(x, x)

        assert isinstance(_Layer, Module)
        assert _Layer.Attention.name == "Attention"
        assert _Layer.Attention.leaf.name == "leaf"

    def test_duplicate_child_module_alias_rejected(self) -> None:
        @module(entry="leaf")
        class _Attention:
            @func
            def leaf(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.matmul(x, w)

        with pytest.raises(ValueError, match="duplicate child module name"):

            @module(entry="composed")
            class _Layer:
                Attention = _Attention
                AttentionAgain = _Attention  # alias — same Module object, same name

                @func
                def composed(x: Tensor[(2, 4), "f32"], w: Tensor[(4, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                    return tf.add(x, x)
