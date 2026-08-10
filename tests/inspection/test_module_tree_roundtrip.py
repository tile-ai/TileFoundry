"""Canonical round trip of a Module tree through the printer and product import.

``as_script`` renders the selected Module as ``@module`` source: its declared
execution context, its entry, every function it owns, and each nested Module as
a class in its body. Importing that file runs the authoring decorators to build
an equal tree, so a declared context stays declared and an inherited one stays
absent.
"""

from __future__ import annotations

from tests._source import import_dsl
from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by bodies
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_CTA = Topology("cta", 132)
_THREAD = Topology("thread", 32)


@module(entry="forward", target=CudaTarget("nvidia.h200_sxm"), topologies=(_CTA,))
class _Tree:
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

    @module(entry="step", topologies=(_THREAD,))
    class replaces:
        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.square(x)

    @module(topologies=())
    class topology_free:
        @func
        def helper(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)

    @module
    class nominates_nothing:
        @func
        def helper(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)


def _child(mod: Module, name: str) -> Module:
    return next(child for child in mod.modules if child.name == name)


def test_the_root_declaration_and_its_functions_survive_the_round_trip() -> None:
    imported = import_dsl(as_script(_Tree))

    assert isinstance(imported, Module)
    assert imported.name == "_Tree"
    assert imported.entry == "forward"
    assert imported.target == CudaTarget("nvidia.h200_sxm")
    assert imported.topologies == (_CTA,)

    assert [fn.name for fn in imported.functions] == ["spare", "forward"]
    assert imported.entry_function().name == "forward"


def test_each_nested_module_survives_with_its_own_context() -> None:
    """A child's context is either declared or inherited.

    A child's context is either declared or inherited, and the round trip must
    not turn the second into the first: a copied-down target would freeze a child
    that should follow whatever parent it is attached to.
    """
    imported = import_dsl(as_script(_Tree))

    assert sorted(child.name for child in imported.modules) == [
        "inherits",
        "nominates_nothing",
        "replaces",
        "topology_free",
    ]
    assert _child(imported, "inherits").entry == "step"
    assert _child(imported, "replaces").entry_function().name == "step"

    inherits = _child(imported, "inherits")
    assert inherits.target is None
    assert inherits.topologies is None
    assert inherits.resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert inherits.effective_topologies() == (_CTA,)

    replaces = _child(imported, "replaces")
    assert replaces.topologies == (_THREAD,)
    assert replaces.effective_topologies() == (_THREAD,)

    topology_free = _child(imported, "topology_free")
    assert topology_free.topologies == ()
    assert topology_free.effective_topologies() == ()


def test_a_child_nominating_no_step_prints_as_a_bare_decorator() -> None:
    """``entry="None"`` would import as a Module whose entry names no function.

    ``entry="None"`` would import as a Module whose entry names no function,
    so the absence has to print as an absence.
    """
    source = as_script(_Tree)

    assert "@module\n    class nominates_nothing:" in source
    assert 'entry="None"' not in source
    assert _child(import_dsl(source), "nominates_nothing").entry is None


def test_printing_the_imported_tree_reaches_a_fixed_point_twice() -> None:
    """The first print names each binding.

    The first print names each binding, so the source it produces is what
    every later print reproduces unchanged.
    """
    source = as_script(_Tree)
    once = as_script(import_dsl(source, "_Tree"))
    twice = as_script(import_dsl(once, "_Tree"))

    assert once == source
    assert twice == source


@module(entry="caller")
class _Siblings:
    """A callee declared above its caller.

    A callee declared above its caller, which is what the printer emits for
    any multi-function Module.
    """

    @func
    def callee(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def caller(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return callee(x)


def test_a_sibling_call_survives_the_round_trip() -> None:
    """A body calling an earlier sibling is canonical output.

    A body calling an earlier sibling is canonical output, so importing it
    must resolve the callee to that sibling rather than read it as an op name.
    """
    imported = import_dsl(as_script(_Siblings))

    assert [fn.name for fn in imported.functions] == ["callee", "caller"]
    callee, caller = imported.functions
    assert caller.body.target is callee

    once = as_script(imported)
    assert as_script(import_dsl(once)) == once
