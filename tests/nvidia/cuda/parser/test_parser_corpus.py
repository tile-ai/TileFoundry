"""NVIDIA inspection, canonical TIR, and named tree inputs share one parser runner."""

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import case, cases
from tests.core.parser._parser_case import ParserCase, check_parser_case, unique_parser_cases
from tilefoundry.cli.source import load_namespace
from tilefoundry.ir.core import Call, Constant, Var, binding_name
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.where import Where
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.pattern import RangePattern
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry_nvidia import CudaTarget

FIXTURES = Path(__file__).parents[1] / "fixtures"


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


@case
class NanValueFill(ParserCase):
    path: Path = FIXTURES / "inspection/nan_value_fill.py"
    prints: tuple[str, ...] = ("math.nan",)

    def guard(self, state, problems):
        value = state.rebuilt.entry_function().body[0].body[0].args[1]
        assert isinstance(value, Constant)
        assert math.isnan(value.value)


@case
class Composed(ParserCase):
    path: Path = FIXTURES / "inspection/composed.py"
    name: str | None = "_Composed"

    def guard(self, state, problems):
        printed, rebuilt = state.printed, state.rebuilt
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


@case
class Dispatching(ParserCase):
    path: Path = FIXTURES / "inspection/dispatching.py"
    name: str | None = "_Dispatching"

    def guard(self, state, problems):
        printed, rebuilt = state.printed, state.rebuilt
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


@case
class ContextTree(ParserCase):
    path: Path = FIXTURES / "logical/module_context.py"
    name: str | None = "ContextTree"

    def guard(self, state, problems):
        problems.run("root context", self.root_context, state.rebuilt)
        problems.run("child contexts", self.child_contexts, state.rebuilt)
        problems.run("empty child entry", self.empty_child_entry, state.printed, state.rebuilt)

    def root_context(self, rebuilt):
        context = load_namespace(str(self.path))[0]
        assert isinstance(rebuilt, Module)
        assert rebuilt.name == "ContextTree"
        assert rebuilt.entry == "forward"
        assert rebuilt.target == CudaTarget("nvidia.h200_sxm")
        assert rebuilt.topologies == (context["CONTEXT_CTA"], context["CONTEXT_WARP"])

        assert [fn.name for fn in rebuilt.functions] == ["spare", "forward"]
        assert rebuilt.entry_function().name == "forward"

    def child_contexts(self, rebuilt):
        context = load_namespace(str(self.path))[0]
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
        assert list_call_paths(rebuilt)["inherits.step"].topologies == (
            context["CONTEXT_CTA"],
            context["CONTEXT_WARP"],
        )

        replaces = _child(rebuilt, "replaces")
        assert replaces.topologies == (context["CONTEXT_THREAD"],)
        assert replaces.effective_topologies() == (context["CONTEXT_THREAD"],)

        topology_free = _child(rebuilt, "topology_free")
        assert topology_free.topologies == ()
        assert topology_free.effective_topologies() == ()

    def empty_child_entry(self, printed, rebuilt):
        assert "@module()\nclass nominates_nothing:" in printed
        assert 'entry="None"' not in printed
        assert _child(rebuilt, "nominates_nothing").entry is None


@case
class DerivedPrefill(ParserCase):
    path: Path = FIXTURES / "placed/derived_prefill.py"
    name: str | None = "DerivedPrefill"

    def guard(self, state, problems):
        ir, printed, rebuilt = state.ir, state.printed, state.rebuilt
        signature = printed[printed.index("    def prefill(") : printed.index("        local =")]

        assert 'prefill_n = DimVar("prefill_n", 1, 64)' in printed
        assert 'topology_only = DimVar("topology_only", 1, 1024)' in printed
        assert 'Topology("cta", ((prefill_n - 1) // 8) + 1)' in printed
        assert 'Topology("thread", topology_only)' in printed
        assert "ShardLayout(" in signature
        assert rebuilt.topologies == ir.topologies
        assert rebuilt.entry_function().params[0].type == (ir.entry_function().params[0].type)


@case
class FlashSplitKDecode(ParserCase):
    path: Path = FIXTURES / "placed/flash_split_k_decode.py"
    name: str | None = "FlashSplitKDecode"

    def guard(self, state, problems):
        ir, rebuilt = state.ir, state.rebuilt
        assert rebuilt.topologies == ir.topologies
        slices = [
            expr
            for expr in collect_exprs(rebuilt.entry_function().body)
            if isinstance(expr, Call) and isinstance(expr.target, Slice)
        ]
        assert len(slices) == 2


@case
class PrefillDecodeAttention(ParserCase):
    path: Path = FIXTURES / "placed/prefill_decode_attention.py"
    name: str | None = "PrefillDecodeAttention"

    def guard(self, state, problems):
        printed, rebuilt = state.printed, state.rebuilt
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


@case
class RmsnormModule(ParserCase):
    path: Path = FIXTURES / "placed/rmsnorm.py"
    name: str | None = "RmsnormModule"

    def guard(self, state, problems):
        ir, printed, rebuilt = state.ir, state.printed, state.rebuilt
        compile(printed, "<test>", "exec")
        assert "topologies=" in printed
        assert rebuilt.topologies == ir.topologies
        assert _structural_equal(ir.entry_function(), rebuilt.entry_function())


@case
class RegionBoundaries(ParserCase):
    path: Path = FIXTURES / "placed/region_boundaries.py"
    name: str | None = "RegionBoundaries"

    def guard(self, state, problems):
        body = state.rebuilt.entry_function().body
        producer = next(
            expr
            for expr in collect_exprs(body)
            if isinstance(expr, MeshRegion) and binding_name(expr.body) == "v2"
        )
        assert sum(producer in expr_children(expr) for expr in collect_exprs(body)) == 2


@dataclass(frozen=True)
class ScaledParent(ParserCase):
    def guard(self, state, problems):
        problems.run("scaled common", self.scaled_common, state.ir, state.rebuilt)
        problems.run("scaled calls", self.scaled_calls, state.rebuilt)

    def scaled_common(self, ir, rebuilt):
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


@case
class PairedScaledParent(ScaledParent):
    path: Path = FIXTURES / "shapes/scaled_modules.py"
    name: str | None = "PairedScaledParent"

    def scaled_calls(self, rebuilt):
        restored_calls = [
            expr for expr in collect_exprs(rebuilt.entry_function().body) if isinstance(expr, Call)
        ]
        targets = [expr.target for expr in restored_calls if isinstance(expr.target, ModuleList)]
        assert targets == list(rebuilt.children.values())


@case
class StackedScaledParent(ScaledParent):
    path: Path = FIXTURES / "shapes/scaled_modules.py"
    name: str | None = "StackedScaledParent"

    def scaled_calls(self, rebuilt):
        restored_calls = [
            expr for expr in collect_exprs(rebuilt.entry_function().body) if isinstance(expr, Call)
        ]
        (index,) = (expr for expr in restored_calls if isinstance(expr.target, TupleGetItem))
        assert index.args[0] is rebuilt.children["layers"]
        assert any(expr.target is index for expr in restored_calls)


EXPECTED = {
    "inspection/type_printer_sugar": dict(
        golden=FIXTURES / "inspection/type_printer_sugar.printed.txt"
    ),
    "inspection/device": dict(prints=("T.sync(thread)", "T.sync(thread[:])")),
    "inspection/launched": dict(prints=("cluster=(1, 1, 1), dynamic_smem=512)",)),
    "tir/async_sync": dict(canonical=True),
    "tir/e4m3_gemm": dict(canonical=True),
    "tir/mma": dict(canonical=True),
    "tir/split_stage": dict(canonical=True),
    "tir/square": dict(canonical=True),
    "tir/sync": dict(canonical=True),
}
NAMED = (
    ContextTree(),
    ParserCase(path=FIXTURES / "logical/gqa_static.py", name="static_online_attend"),
    DerivedPrefill(),
    FlashSplitKDecode(),
    PrefillDecodeAttention(),
    RmsnormModule(),
    RegionBoundaries(),
    PairedScaledParent(),
    StackedScaledParent(),
    ParserCase(path=FIXTURES / "scaffold/tir_dispatch.py", name="Root"),
    ParserCase(
        path=FIXTURES / "scaffold/tir_launches.py",
        name="Root",
        prints=("launch(child.run,",),
    ),
)
CASES = (
    cases(FIXTURES / "inspection", FIXTURES / "tir", case_type=ParserCase, expected=EXPECTED)
    + NAMED
)
CASES = unique_parser_cases(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_parser_case(case: ParserCase) -> None:
    check_parser_case(case)
