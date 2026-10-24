"""Structural parser and printer checks attached to NVIDIA input cases."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical import module_context as context_fixture
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, Constant, Var, binding_name
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.where import Where
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry_nvidia import CudaTarget


def _child(mod: Module, name: str) -> Module:
    return mod.children[name].elements[0]


def _structural_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Function):
        return (
            a.name == b.name
            and len(a.params) == len(b.params)
            and all(_structural_equal(pa, pb) for pa, pb in zip(a.params, b.params))
            and _structural_equal(a.body, b.body)
            and _structural_equal(a.return_type, b.return_type)
        )
    if isinstance(a, Var):
        return a.name == b.name and _structural_equal(a.type, b.type)
    if isinstance(a, Constant):
        return a.value == b.value
    if isinstance(a, MeshRegion):
        return (
            _attr_equal(a.mesh, b.mesh)
            and _structural_equal(a.body, b.body)
            and _structural_equal(a.type, b.type)
        )
    if isinstance(a, Call):
        if type(a.target) is not type(b.target) or len(a.args) != len(b.args):
            return False
        if not all(_structural_equal(aa, bb) for aa, bb in zip(a.args, b.args)):
            return False
        for pi in type(a.target).params():
            if pi.kind == "attribute":
                if getattr(a.target, pi.name, None) != getattr(b.target, pi.name, None):
                    if not _attr_equal(
                        getattr(a.target, pi.name, None),
                        getattr(b.target, pi.name, None),
                    ):
                        return False
        return _structural_equal(a.type, b.type)
    if isinstance(a, TensorType):
        return (
            a.shape == b.shape
            and a.dtype == b.dtype
            and a.storage == b.storage
            and _attr_equal(a.layout, b.layout)
        )
    return a == b


def _attr_equal(a, b) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_attr_equal(aa, bb) for aa, bb in zip(a, b))
    if hasattr(a, "__dataclass_fields__"):
        return all(_attr_equal(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)
    return a == b


def _derived_geometry(state, problems):
    def check(ir, printed, rebuilt):
        signature = printed[printed.index("    def prefill(") : printed.index("        local =")]

        assert 'prefill_n = DimVar("prefill_n", 1, 64)' in printed
        assert 'topology_only = DimVar("topology_only", 1, 1024)' in printed
        assert 'Topology("cta", ((prefill_n - 1) // 8) + 1)' in printed
        assert 'Topology("thread", topology_only)' in printed
        assert "ShardLayout(" in signature
        assert rebuilt.topologies == ir.topologies
        assert rebuilt.entry_function().params[0].type == (ir.entry_function().params[0].type)

    problems.run("_derived_geometry", check, state.ir, state.printed, state.rebuilt)


def _root_context(state, problems):
    def check(printed, rebuilt):
        assert isinstance(rebuilt, Module)
        assert rebuilt.name == "ContextTree"
        assert rebuilt.entry == "forward"
        assert rebuilt.target == CudaTarget("nvidia.h200_sxm")
        assert rebuilt.topologies == (_CTA, _WARP)

        assert [fn.name for fn in rebuilt.functions] == ["spare", "forward"]
        assert rebuilt.entry_function().name == "forward"

    problems.run("_root_context", check, state.printed, state.rebuilt)


def _child_contexts(state, problems):
    def check(printed, rebuilt):
        assert sorted(rebuilt.children) == [
            "inherits",
            "nominates_nothing",
            "replaces",
            "topology_free",
        ]
        assert _child(rebuilt, "inherits").entry == "step"
        assert _child(rebuilt, "replaces").entry_function().name == "step"

        inherits = _child(rebuilt, "inherits")
        assert inherits.target is None
        assert inherits.topologies is None
        assert list_call_paths(rebuilt)["inherits.step"].target == CudaTarget("nvidia.h200_sxm")
        assert list_call_paths(rebuilt)["inherits.step"].topologies == (_CTA, _WARP)

        replaces = _child(rebuilt, "replaces")
        assert replaces.topologies == (_THREAD,)
        assert replaces.effective_topologies() == (_THREAD,)

        topology_free = _child(rebuilt, "topology_free")
        assert topology_free.topologies == ()
        assert topology_free.effective_topologies() == ()

    problems.run("_child_contexts", check, state.printed, state.rebuilt)


def _empty_child_entry(state, problems):
    def check(printed, rebuilt):
        assert "@module()\nclass nominates_nothing:" in printed
        assert 'entry="None"' not in printed
        assert _child(rebuilt, "nominates_nothing").entry is None

    problems.run("_empty_child_entry", check, state.printed, state.rebuilt)


_CTA = context_fixture.CONTEXT_CTA
_WARP = context_fixture.CONTEXT_WARP
_THREAD = context_fixture.CONTEXT_THREAD


def _prefill_variants(state, problems):
    def check(printed, rebuilt):
        assert "arange(" in printed
        assert "where(" in printed
        variants = rebuilt.entry_function().variants
        assert len(variants) == 2
        for variant in variants:
            targets = {
                type(expr.target) for expr in collect_exprs(variant.body) if isinstance(expr, Call)
            }
            assert Arange in targets
            assert Where in targets

    problems.run("_prefill_variants", check, state.printed, state.rebuilt)


def _flash_slices(state, problems):
    def check(ir, rebuilt):
        assert rebuilt.topologies == ir.topologies
        slices = [
            expr
            for expr in collect_exprs(rebuilt.entry_function().body)
            if isinstance(expr, Call) and isinstance(expr.target, Slice)
        ]
        assert len(slices) == 2

    problems.run("_flash_slices", check, state.ir, state.rebuilt)


def _composed_tree(state, problems):
    def check(printed, rebuilt):
        assert "v0 = first(x)" in printed and "v1 = second(x)" in printed
        assert "class Expert:" in printed and ".run(" not in printed
        assert "first = Expert" in printed and "second = Expert" in printed
        left, right = rebuilt.entry_function().body.args
        assert len(left.args) == 1
        assert [param.is_const for param in left.target.elements[0].entry_function().params] == [
            False,
            True,
        ]
        first, second = rebuilt.children.values()
        assert tuple(rebuilt.children) == ("first", "second")
        assert first.elements[0] is second.elements[0]
        assert left.target is first
        assert right.target is second
        assert printed.index("first = Expert") < printed.index("def fused(")
        assert printed.index("second = Expert") < printed.index("def fused(")

    problems.run("_composed_tree", check, state.printed, state.rebuilt)


def _dispatch_tree(state, problems):
    def check(printed, rebuilt):
        block = printed[printed.index("@dispatch.specialize(") :]
        assert "leaf(x)" in block

        (rebuilt_child,) = rebuilt.children.values()
        (variant,) = rebuilt.entry_function().variants
        assert rebuilt.entry_function().body is None
        assert variant.specializations == (RangePattern("n_print", 1, 8),)
        assert variant.body.target is rebuilt_child
        assert len(variant.body.args) == 1
        assert [
            param.is_const for param in variant.body.target.elements[0].entry_function().params
        ] == [False, True]

    problems.run("_dispatch_tree", check, state.printed, state.rebuilt)


def _scaled_common(state, problems):
    def check(ir, rebuilt):
        assert tuple(rebuilt.children) == tuple(ir.children)
        slots = tuple(rebuilt.children.values())
        for key, slot in rebuilt.children.items():
            assert slot.key == key
            assert slot.indexed == ir.children[key].indexed
            assert len(slot.elements) == len(ir.children[key].elements)
            assert all(element is slots[0].elements[0] for element in slot.elements)
        original_calls = [
            expr for expr in collect_exprs(ir.entry_function().body) if isinstance(expr, Call)
        ]
        restored_calls = [
            expr for expr in collect_exprs(rebuilt.entry_function().body) if isinstance(expr, Call)
        ]
        assert len(restored_calls) == len(original_calls)

    problems.run("_scaled_common", check, state.ir, state.rebuilt)


def _paired_calls(state, problems):
    def check(rebuilt):
        restored_calls = [
            expr for expr in collect_exprs(rebuilt.entry_function().body) if isinstance(expr, Call)
        ]
        targets = [expr.target for expr in restored_calls if isinstance(expr.target, ModuleList)]
        assert targets == list(rebuilt.children.values())

    problems.run("_paired_calls", check, state.rebuilt)


def _stacked_calls(state, problems):
    def check(rebuilt):
        restored_calls = [
            expr for expr in collect_exprs(rebuilt.entry_function().body) if isinstance(expr, Call)
        ]
        (index,) = (expr for expr in restored_calls if isinstance(expr.target, TupleGetItem))
        assert index.args[0] is rebuilt.children["layers"]
        assert any(expr.target is index for expr in restored_calls)

    problems.run("_stacked_calls", check, state.rebuilt)


def _rms_structure(state, problems):
    def check(ir, printed, rebuilt):
        compile(printed, "<test>", "exec")
        assert "topologies=" in printed
        assert rebuilt.topologies == ir.topologies
        assert _structural_equal(ir.entry_function(), rebuilt.entry_function())

    problems.run("_rms_structure", check, state.ir, state.printed, state.rebuilt)


def _region_escapes(state, problems):
    def check(rebuilt):
        body = rebuilt.entry_function().body
        producer = next(
            expr
            for expr in collect_exprs(body)
            if isinstance(expr, MeshRegion) and binding_name(expr.body) == "v2"
        )
        assert sum(producer in expr_children(expr) for expr in collect_exprs(body)) == 2

    problems.run("_region_escapes", check, state.rebuilt)
