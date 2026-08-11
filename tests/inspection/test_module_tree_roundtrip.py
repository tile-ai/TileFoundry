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
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, Mesh, Tensor, tf  # noqa: F401
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.specialize import origin_of
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_CTA = Topology("cta", 132)
_THREAD = Topology("thread", 32)
_N = DimVar("n_print", 1, 9)


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


def test_a_child_nominating_no_step_prints_an_empty_argument_list() -> None:
    """``entry="None"`` would import as a Module whose entry names no function.

    ``entry="None"`` would import as a Module whose entry names no function, so
    the absence has to print as an absence. The decorator is still called: a bare
    one has not run while a class body naming a child call is evaluated.
    """
    source = as_script(_Tree)

    assert "@module()\n    class nominates_nothing:" in source
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


@module(entry="run")
class _Weighted:
    @func
    def run(
        x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
    ) -> Tensor[(4, 8), "f32"]:
        return tf.matmul(x, w)


@module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
class _Composed:
    first = _Weighted
    second = _Weighted

    @func
    def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.add(first(x), second(x))  # noqa: F821


def test_a_child_call_prints_by_its_binding_and_carries_activations_only() -> None:
    """The callee's weight is the child's, so the call site never names it.

    Importing restores the complete signature from the child's own class body
    rather than inventing an argument at the call site.
    """
    source = as_script(_Composed)

    assert "v0 = first(x)" in source and "v1 = second(x)" in source
    assert "_Weighted" not in source and ".run(" not in source

    imported = import_dsl(source)
    left, right = imported.entry_function().body.args
    assert len(left.args) == 1
    assert [param.is_const for param in left.target.params] == [False, True]


def test_two_attached_copies_of_one_child_stay_distinct_through_the_trip() -> None:
    imported = import_dsl(as_script(_Composed))

    first, second = imported.modules
    assert (first.name, second.name) == ("first", "second")
    left, right = imported.entry_function().body.args
    assert left.target is first.entry_function()
    assert right.target is second.entry_function()


def test_a_composed_tree_prints_to_a_fixed_point() -> None:
    source = as_script(_Composed)
    once = as_script(import_dsl(source, "_Composed"))

    assert once == source
    assert as_script(import_dsl(once, "_Composed")) == source


def test_a_child_before_the_functions_naming_it() -> None:
    """A class body binds in source order, so the attribute has to exist first."""
    source = as_script(_Composed)

    assert source.index("class first:") < source.index("def fused(")
    assert source.index("class second:") < source.index("def fused(")


@module(entry="root", target=CudaTarget("nvidia.h200_sxm"), topologies=(_CTA,))
class _Rebuilt:
    leaf = _Weighted

    @func
    def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            local = tf.reshard(x, (4 @ cta.tile, 8), "gmem")
            return leaf(local)  # noqa: F821


def test_a_call_site_rebuilt_target_still_prints_by_its_binding() -> None:
    """The printed target is a rebuild, recognised through what it records.

    A layout reaching the callee rebuilds it, so the call no longer targets the
    attached entry itself; matching on the name would accept any function called
    the same, which is what the recorded origin replaces.
    """
    (child,) = _Rebuilt.modules
    call = _Rebuilt.entry_function().body
    assert call.target is not child.entry_function()
    assert origin_of(call.target) is child.entry_function()

    source = as_script(_Rebuilt)
    assert "leaf(" in source
    imported = import_dsl(source)
    (imported_child,) = imported.modules
    rebuilt = imported.entry_function().body
    assert origin_of(rebuilt.target) is imported_child.entry_function()


@module(entry="run")
class _WeightedAtAnySize:
    @func
    def run(
        x: Tensor[(_N, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
    ) -> Tensor[(_N, 8), "f32"]:
        return tf.matmul(x, w)


@module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
class _Dispatching:
    leaf = _WeightedAtAnySize

    @func
    def dispatch(x: Tensor[(_N, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        pass

    @dispatch.specialize(DimVarRangePat("n_print", 1, 9))
    def _(x: Tensor[(_N, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        return leaf(x)  # noqa: F821


def test_a_child_call_in_a_specialization_body_survives_the_round_trip() -> None:
    """A variant body reaches a child the same way a base body does.

    A prototype has no body, so the variant is the only place this call exists;
    printing it by the callee's own name would name nothing the class body binds.
    """
    source = as_script(_Dispatching)
    block = source[source.index("@dispatch.specialize(") :]
    assert "leaf(x)" in block

    imported = import_dsl(source)
    (imported_child,) = imported.modules
    (variant,) = imported.entry_function().variants
    assert imported.entry_function().body is None
    assert variant.specializations == (DimVarRangePat("n_print", 1, 9),)
    assert variant.body.target is imported_child.entry_function()
    assert len(variant.body.args) == 1
    assert [param.is_const for param in variant.body.target.params] == [False, True]
