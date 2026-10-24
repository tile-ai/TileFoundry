"""Canonical round trip of a Module tree through the printer and product import.

``as_script`` renders the selected Module as ``@module`` source: its declared
execution context, its entry, every function it owns, and each shared Module
definition with its instance slot bindings. Importing that file runs the
authoring decorators to build
an equal tree, so a declared context stays declared and an inherited one stays
absent.
"""

from __future__ import annotations

import pytest

from tests.core._source import import_dsl
from tests.nvidia.cuda.fixtures.logical import module_context as context_fixture
from tests.nvidia.cuda.fixtures.logical.gqa_static import static_online_attend
from tests.nvidia.cuda.fixtures.logical.hir_composition import Expert
from tests.nvidia.cuda.fixtures.placed.derived_prefill import DerivedPrefill
from tests.nvidia.cuda.fixtures.placed.flash_split_k_decode import FlashSplitKDecode
from tests.nvidia.cuda.fixtures.placed.prefill_decode_attention import PrefillDecodeAttention
from tests.nvidia.cuda.fixtures.shapes import scaled_modules
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.inspection.dot import hir_function_to_dot
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.where import Where
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry_nvidia import CudaTarget

_CTA = context_fixture.CONTEXT_CTA
_WARP = context_fixture.CONTEXT_WARP
_THREAD = context_fixture.CONTEXT_THREAD
ContextTree = context_fixture.ContextTree
_N = DimVar("n_print", 1, 8)


def test_a_derived_topology_and_mesh_geometry_survive_the_round_trip() -> None:
    source = as_script(DerivedPrefill)
    imported = import_dsl(source)
    signature = source[source.index("    def prefill(") : source.index("        local =")]

    assert 'prefill_n = DimVar("prefill_n", 1, 64)' in source
    assert 'topology_only = DimVar("topology_only", 1, 1024)' in source
    assert 'Topology("cta", ((prefill_n - 1) // 8) + 1)' in source
    assert 'Topology("thread", topology_only)' in source
    assert "ShardLayout(" in signature
    assert imported.topologies == DerivedPrefill.topologies
    assert imported.entry_function().params[0].type == (
        DerivedPrefill.entry_function().params[0].type
    )
    assert as_script(imported) == source


def test_prefill_decode_specializations_survive_the_round_trip() -> None:
    source = as_script(PrefillDecodeAttention)
    imported = import_dsl(source)
    restored = import_dsl(as_script(imported))

    assert as_script(restored) == as_script(imported)

    assert "arange(" in source
    assert "where(" in source
    for roundtripped in (imported, restored):
        variants = roundtripped.entry_function().variants
        assert len(variants) == 2
        for variant in variants:
            targets = {
                type(expr.target) for expr in collect_exprs(variant.body) if isinstance(expr, Call)
            }
            assert Arange in targets
            assert Where in targets


def test_flash_split_k_decode_survives_the_round_trip() -> None:
    source = as_script(FlashSplitKDecode)
    imported = import_dsl(source)
    restored = import_dsl(as_script(imported))

    assert as_script(restored) == as_script(imported)

    for roundtripped in (imported, restored):
        assert roundtripped.topologies == FlashSplitKDecode.topologies
        slices = [
            expr
            for expr in collect_exprs(roundtripped.entry_function().body)
            if isinstance(expr, Call) and isinstance(expr.target, Slice)
        ]
        assert len(slices) == 2


def _child(mod: Module, name: str) -> Module:
    return mod.children[name].elements[0]


def test_the_root_declaration_and_its_functions_survive_the_round_trip() -> None:
    imported = import_dsl(as_script(ContextTree))

    assert isinstance(imported, Module)
    assert imported.name == "ContextTree"
    assert imported.entry == "forward"
    assert imported.target == CudaTarget("nvidia.h200_sxm")
    assert imported.topologies == (_CTA, _WARP)

    assert [fn.name for fn in imported.functions] == ["spare", "forward"]
    assert imported.entry_function().name == "forward"


def test_each_nested_module_survives_with_its_own_context() -> None:
    """A child's context is either declared or inherited.

    A child's context is either declared or inherited, and the round trip must
    not turn the second into the first: a copied-down target would freeze a child
    that should follow whatever parent it is attached to.
    """
    imported = import_dsl(as_script(ContextTree))

    assert sorted(imported.children) == [
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
    assert list_call_paths(imported)["inherits.step"].target == CudaTarget("nvidia.h200_sxm")
    assert list_call_paths(imported)["inherits.step"].topologies == (_CTA, _WARP)

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
    source = as_script(ContextTree)

    assert "@module()\nclass nominates_nothing:" in source
    assert 'entry="None"' not in source
    assert _child(import_dsl(source), "nominates_nothing").entry is None


def test_printing_the_imported_tree_reaches_a_fixed_point_twice() -> None:
    """The first print names each binding.

    The first print names each binding, so the source it produces is what
    every later print reproduces unchanged.
    """
    source = as_script(ContextTree)
    once = as_script(import_dsl(source, "ContextTree"))
    twice = as_script(import_dsl(once, "ContextTree"))

    assert once == source
    assert twice == source


@module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
class _Composed:
    first = Expert
    second = Expert

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
    assert "class Expert:" in source and ".run(" not in source
    assert "first = Expert" in source and "second = Expert" in source

    imported = import_dsl(source)
    left, right = imported.entry_function().body.args
    assert len(left.args) == 1
    assert [param.is_const for param in left.target.elements[0].entry_function().params] == [False, True]


def test_two_slots_share_one_child_definition_through_the_trip() -> None:
    imported = import_dsl(as_script(_Composed))

    first, second = imported.children.values()
    assert tuple(imported.children) == ("first", "second")
    assert first.elements[0] is second.elements[0]
    left, right = imported.entry_function().body.args
    assert left.target is first
    assert right.target is second


def test_a_composed_tree_prints_to_a_fixed_point() -> None:
    source = as_script(_Composed)
    once = as_script(import_dsl(source, "_Composed"))

    assert once == source
    assert as_script(import_dsl(once, "_Composed")) == source


def test_a_child_before_the_functions_naming_it() -> None:
    """A class body binds in source order, so the attribute has to exist first."""
    source = as_script(_Composed)

    assert source.index("first = Expert") < source.index("def fused(")
    assert source.index("second = Expert") < source.index("def fused(")


@module(entry="run")
class _WeightedAtAnySize:
    @func
    def run(x: Tensor[(_N, 8), "f32"], w: ConstTensor[(8, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        return tf.matmul(x, w)


@module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
class _Dispatching:
    leaf = _WeightedAtAnySize

    @func
    def dispatch(x: Tensor[(_N, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        pass

    @dispatch.specialize(RangePattern("n_print", 1, 8))
    def child_dispatch(x: Tensor[(_N, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
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
    (imported_child,) = imported.children.values()
    (variant,) = imported.entry_function().variants
    assert imported.entry_function().body is None
    assert variant.specializations == (RangePattern("n_print", 1, 8),)
    assert variant.body.target is imported_child
    assert len(variant.body.args) == 1
    assert [param.is_const for param in variant.body.target.elements[0].entry_function().params] == [False, True]


def test_hir_function_dot_keeps_loop_regions_as_opaque_leaves() -> None:
    """The public DOT form keeps structured regions as white leaf boxes."""
    dot = hir_function_to_dot(static_online_attend.entry_function())

    assert len(dot.splitlines()) == 33
    assert 'label="LoopRegion", fillcolor="#ffffff"' in dot
    assert "TupleType(" not in dot


@pytest.mark.parametrize("name", ["PairedScaledParent", "StackedScaledParent"])
def test_shared_slots_and_instance_calls_round_trip(name) -> None:
    root = getattr(scaled_modules, name)
    source = as_script(root)
    imported = import_dsl(source, root.name)
    assert as_script(imported) == source
    assert tuple(imported.children) == tuple(root.children)
    slots = tuple(imported.children.values())
    for key, slot in imported.children.items():
        assert slot.key == key
        assert slot.indexed == root.children[key].indexed
        assert len(slot.elements) == len(root.children[key].elements)
        assert all(element is slots[0].elements[0] for element in slot.elements)
    original_calls = [expr for expr in collect_exprs(root.entry_function().body) if isinstance(expr, Call)]
    restored_calls = [expr for expr in collect_exprs(imported.entry_function().body) if isinstance(expr, Call)]
    assert len(restored_calls) == len(original_calls)
    targets = [expr.target for expr in restored_calls if isinstance(expr.target, ModuleList)]
    if name == "PairedScaledParent":
        assert targets == list(imported.children.values())
    else:
        (index,) = (expr for expr in restored_calls if isinstance(expr.target, TupleGetItem))
        assert index.args[0] is imported.children["layers"]
        assert any(expr.target is index for expr in restored_calls)
