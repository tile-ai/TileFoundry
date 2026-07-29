"""Canonical round trip of a Module tree through the printer and parser.

``as_script`` renders the selected Module as ``@module`` source: its declared
execution context, its entry, every function it owns, and each nested Module as
a class in its body. ``parse_script`` reads that source back into an equal
tree, so a declared context stays declared and an inherited one stays absent.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by bodies
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Topology
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.target import CudaTarget

_CTA = Topology("cta", 132)
_THREAD = Topology("thread", 32)


@module(entry="forward", target=CudaTarget())
class _Tree:
    topologies = (_CTA,)

    @func
    def forward(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def spare(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.square(x)

    @module(entry="step")
    class inherits:
        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)

    @module(entry="step")
    class replaces:
        topologies = (_THREAD,)

        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.square(x)

    @module
    class nominates_nothing:
        @func
        def helper(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)


def _child(mod: Module, name: str) -> Module:
    return next(child for child in mod.modules if child.name == name)


def test_the_root_declaration_and_its_functions_survive_the_round_trip() -> None:
    reparsed = parse_script(as_script(_Tree))

    assert isinstance(reparsed, Module)
    assert reparsed.name == "_Tree"
    assert reparsed.entry == "forward"
    assert reparsed.target == CudaTarget()
    assert reparsed.topologies == (_CTA,)
    # Every owned function, in the order the printer emits (callees first).
    assert [fn.name for fn in reparsed.functions] == ["spare", "forward"]
    assert reparsed.entry_function().name == "forward"


def test_each_nested_module_survives_with_its_own_context() -> None:
    """A child's context is either declared or inherited, and the round trip must
    not turn the second into the first: a copied-down target would freeze a child
    that should follow whatever parent it is attached to."""
    reparsed = parse_script(as_script(_Tree))

    assert sorted(child.name for child in reparsed.modules) == [
        "inherits", "nominates_nothing", "replaces",
    ]
    assert _child(reparsed, "inherits").entry == "step"
    assert _child(reparsed, "replaces").entry_function().name == "step"

    inherits = _child(reparsed, "inherits")
    assert inherits.target is None
    assert inherits.topologies is None
    assert inherits.resolve_target() == CudaTarget()
    assert inherits.effective_topologies() == (_CTA,)

    replaces = _child(reparsed, "replaces")
    assert replaces.topologies == (_THREAD,)
    assert replaces.effective_topologies() == (_THREAD,)


def test_a_child_nominating_no_step_prints_as_a_bare_decorator() -> None:
    """``entry="None"`` would re-parse into a Module whose entry names no function,
    so the absence has to print as an absence."""
    source = as_script(_Tree)

    assert "@module\n    class nominates_nothing:" in source
    assert 'entry="None"' not in source
    assert _child(parse_script(source), "nominates_nothing").entry is None


def test_printing_the_reparsed_tree_reaches_a_fixed_point() -> None:
    """The first print names each binding, so the source it produces is what
    every later print reproduces unchanged."""
    once = as_script(parse_script(as_script(_Tree)))

    assert as_script(parse_script(once)) == once


@module(entry="caller")
class _Siblings:
    """A callee declared above its caller, which is what the printer emits for
    any multi-function Module."""

    @func
    def callee(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def caller(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return callee(x)


def test_a_sibling_call_survives_the_round_trip() -> None:
    """A body calling an earlier sibling is canonical output, so parsing it back
    must resolve the callee to that sibling rather than read it as an op name."""
    reparsed = parse_script(as_script(_Siblings))

    assert [fn.name for fn in reparsed.functions] == ["callee", "caller"]
    callee, caller = reparsed.functions
    assert caller.body.target is callee

    once = as_script(reparsed)
    assert as_script(parse_script(once)) == once
