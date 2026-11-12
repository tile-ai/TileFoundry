"""NVIDIA inspection, canonical TIR, and named tree inputs share one parser runner."""

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import case, cases
from tests.core.parser._parser_case import ParserCase, check_parser_case, unique_parser_cases
from tilefoundry import cli, codegen, parser, verifier
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, pattern, tir, types
from tilefoundry.ir.hir import tensor
from tilefoundry_nvidia import cuda

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _child(mod: core.Module, name: str) -> core.Module:
    return mod.children[name]


def _structural_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, core.Function):
        return (
            a.name == b.name
            and len(a.params) == len(b.params)
            and all(_structural_equal(pa, pb) for pa, pb in zip(a.params, b.params))
            and _structural_equal(a.body, b.body)
            and _structural_equal(a.return_type, b.return_type)
        )
    if isinstance(a, expr.Var):
        return a.name == b.name and _structural_equal(a.type, b.type)
    if isinstance(a, expr.Constant):
        return a.value == b.value
    if isinstance(a, core.MeshRegion):
        return (
            _attr_equal(a.mesh, b.mesh)
            and _structural_equal(a.body, b.body)
            and _structural_equal(a.type, b.type)
        )
    if isinstance(a, expr.Call):
        if type(a.target) is not type(b.target) or len(a.args) != len(b.args):
            return False
        if not all(_structural_equal(aa, bb) for aa, bb in zip(a.args, b.args)):
            return False
        for pi in expr.parameters_of(type(a.target)):
            if pi.kind == "attribute":
                if getattr(a.target, pi.name, None) != getattr(b.target, pi.name, None):
                    if not _attr_equal(
                        getattr(a.target, pi.name, None),
                        getattr(b.target, pi.name, None),
                    ):
                        return False
        return _structural_equal(a.type, b.type)
    if isinstance(a, types.TensorType):
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
        assert isinstance(value, expr.Constant)
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
        assert [param.is_const for param in core.resolve(rebuilt, left.target)[1].params] == [
            False,
            True,
        ]
        first, second = rebuilt.children.values()
        assert tuple(rebuilt.children) == ("first", "second")
        assert first is second
        assert left.target == core.FunctionRef(("first",), None)
        assert right.target == core.FunctionRef(("second",), None)
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

        (key,) = rebuilt.children
        (variant,) = rebuilt.entry_function().variants
        assert rebuilt.entry_function().body is None
        assert variant.specializations == (pattern.Range("n_print", 1, 8),)
        assert variant.body.target == core.FunctionRef((key,), None)
        assert len(variant.body.args) == 1
        assert [
            param.is_const for param in core.resolve(rebuilt, variant.body.target)[1].params
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
        context = cli.load_namespace(str(self.path))[0]
        assert isinstance(rebuilt, core.Module)
        assert rebuilt.name == "ContextTree"
        assert rebuilt.entry == "forward"
        assert rebuilt.target == cuda.Target("nvidia.h200_sxm")
        assert rebuilt.topologies == (context["CONTEXT_CTA"], context["CONTEXT_WARP"])

        assert [fn.name for fn in rebuilt.functions] == ["spare", "forward"]
        assert rebuilt.entry_function().name == "forward"

    def child_contexts(self, rebuilt):
        context = cli.load_namespace(str(self.path))[0]
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
        assert valuation.collect_functions(rebuilt)[core.FunctionRef(("inherits",), "step")].target == cuda.Target("nvidia.h200_sxm")
        assert valuation.collect_functions(rebuilt)[core.FunctionRef(("inherits",), "step")].topologies == (
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
            node
            for node in core.collect_exprs(rebuilt.entry_function().body)
            if isinstance(node, expr.Call) and isinstance(node.target, tensor.Slice)
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
                type(node.target) for node in core.collect_exprs(variant.body) if isinstance(node, expr.Call)
            }
            assert tensor.Arange in targets
            assert tensor.Where in targets


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
            node
            for node in core.collect_exprs(body)
            if isinstance(node, core.MeshRegion) and expr.binding_name(node.body) == "v2"
        )
        assert sum(producer in core.expr_children(node) for node in core.collect_exprs(body)) == 2


@dataclass(frozen=True)
class ScaledParent(ParserCase):
    def guard(self, state, problems):
        problems.run("scaled common", self.scaled_common, state.ir, state.rebuilt)
        problems.run("scaled calls", self.scaled_calls, state.rebuilt)

    def scaled_common(self, ir, rebuilt):
        assert tuple(rebuilt.children) == tuple(ir.children)
        slots = tuple(rebuilt.children.values())
        for key, slot in rebuilt.children.items():
            assert isinstance(slot, tuple) == isinstance(ir.children[key], tuple)
            assert len(core.instances(slot)) == len(core.instances(ir.children[key]))
            assert all(element is core.instances(slots[0])[0] for element in core.instances(slot))
        original_calls = [
            node for node in core.collect_exprs(ir.entry_function().body) if isinstance(node, expr.Call)
        ]
        restored_calls = [
            node for node in core.collect_exprs(rebuilt.entry_function().body) if isinstance(node, expr.Call)
        ]
        assert len(restored_calls) == len(original_calls)


@case
class PairedScaledParent(ScaledParent):
    path: Path = FIXTURES / "shapes/scaled_modules.py"
    name: str | None = "PairedScaledParent"

    def scaled_calls(self, rebuilt):
        restored_calls = [
            node for node in core.collect_exprs(rebuilt.entry_function().body) if isinstance(node, expr.Call)
        ]
        targets = [node.target for node in restored_calls if isinstance(node.target, core.FunctionRef)]
        assert targets == [core.FunctionRef((key,), None) for key in rebuilt.children]


@case
class StackedScaledParent(ScaledParent):
    path: Path = FIXTURES / "shapes/scaled_modules.py"
    name: str | None = "StackedScaledParent"

    def scaled_calls(self, rebuilt):
        restored_calls = [
            node for node in core.collect_exprs(rebuilt.entry_function().body) if isinstance(node, expr.Call)
        ]
        (call,) = (node for node in restored_calls if isinstance(node.target, core.FunctionRef))
        assert call.target.path[0] == "layers" and isinstance(rebuilt.children["layers"], tuple)


INDEXED_CALLS = FIXTURES / "scaffold/tir_indexed_calls.py"
INDEXED_LAUNCH = FIXTURES / "scaffold/tir_indexed_launch.py"


def _evaluates(function):
    """The top-level Evaluate statements of *function*, in source order."""
    return [stmt for stmt in function.body.body if isinstance(stmt, core.Evaluate)]


@case
class IndexedCalls(ParserCase):
    """Named TIR calls pick instances by constant index, or by run-time index on one definition."""

    path: Path = INDEXED_CALLS
    name: str | None = "Root"
    prints: tuple[str, ...] = ("a.layers[2].run(x)", "a.layers[i].run(x)", "a.mixed[1].run(x)")

    def guard(self, state, problems):
        for module in (state.ir, state.rebuilt):
            main = module.entry_function()
            x, i = main.params
            calls = {stmt.callable.path: stmt for stmt in _evaluates(main)}
            assert set(calls) == {("a", "layers", 2), ("a", "layers", None), ("a", "mixed", 1)}
            assert all(stmt.callable.name == "run" for stmt in calls.values())
            dynamic = calls["a", "layers", None]
            assert dynamic.callable.indices(dynamic.args) == (i,)
            assert dynamic.callable.arguments(dynamic.args) == (x,)
            for path in (("a", "layers", 2), ("a", "mixed", 1)):
                assert calls[path].callable.indices(calls[path].args) == ()
                assert calls[path].callable.arguments(calls[path].args) == (x,)
            stack = module.children["a"]
            assert core.resolve(module, calls["a", "layers", 2].callable)[1] is stack.children["layers"][2].lookup("run")
            assert core.resolve(module, dynamic.callable)[1] is stack.children["layers"][0].lookup("run")
            other = core.resolve(module, calls["a", "mixed", 1].callable)
            assert other[0] is stack.children["mixed"][1] and other[0] is not stack.children["mixed"][0]


@case
class IndexedLaunch(ParserCase):
    """Constant-index launches resolve to the definitions of the chosen instances."""

    path: Path = INDEXED_LAUNCH
    name: str | None = "Root"
    prints: tuple[str, ...] = ("launch(layers[0].run, x", "launch(layers[1].run, x")

    def guard(self, state, problems):
        for module in (state.ir, state.rebuilt):
            refs = [stmt.args[0] for stmt in _evaluates(module.entry_function())]
            assert refs == [core.FunctionRef(("layers", 0), "run"), core.FunctionRef(("layers", 1), "run")]
            scale, shift = module.children["layers"]
            chosen = [core.resolve(module, ref)[1] for ref in refs]
            assert chosen == [scale.lookup("run"), shift.lookup("run")]
            assert chosen[0] is not chosen[1]
        edges = codegen.build_context(module).launch_edges
        assert [edge.callee for edge in edges] == chosen
        host = module.entry_function()
        dynamic = core.Evaluate(
            tir.launch.Launch(), (core.FunctionRef(("layers", None), "run"), *_evaluates(host)[0].args[1:])
        )
        with pytest.raises(expr.VerifyError, match="a launch index must be a constant"):
            verifier.verify_prim_function(core.PrimFunction(
                "host", host.params, core.Sequential((dynamic,)), target=host.target
            ))


@pytest.mark.parametrize(
    ("source", "old", "new", "refused"),
    [
        (INDEXED_LAUNCH, "host(x: Tensor[(8,), \"f32\"]):\n        launch(layers[0].run",
         "host(x: Tensor[(8,), \"f32\"], i: Tensor[(), \"i32\"]):\n        launch(layers[i].run",
         "a launch index must be a constant"),
        (INDEXED_CALLS, "a.mixed[1].run(x)", "a.mixed[i].run(x)", "variable index requires the same definition"),
        (INDEXED_CALLS, "a.layers[2].run(x)", "a.layers[True].run(x)", "index must be an integer scalar"),
        (INDEXED_CALLS, "a.layers[2].run(x)", "a.layers[0.5].run(x)", "index must be an integer scalar"),
        (INDEXED_CALLS, "a.layers[2].run(x)", "a.layers[-1].run(x)", "index -1 out of range for N=3"),
        (INDEXED_CALLS, "a.layers[2].run(x)", "a.layers[3].run(x)", r"'a\.layers\.3\.run'.*index 3 out of range for N=3"),
        (INDEXED_LAUNCH, "launch(layers[0].run", "launch(layers[2].run", r"'layers\.2\.run'.*index 2 out of range for N=2"),
        (INDEXED_LAUNCH, "launch(layers[0].run", "launch(layers[0].step", r"'layers\.0\.step' must resolve to exactly one function"),
    ],
    ids=["dynamic-launch", "mixed-dynamic", "bool", "float", "negative", "past-end", "launch-past-end", "launch-missing"],
)
def test_indexed_reference_refusals(tmp_path, source, old, new, refused):
    text = source.read_text()
    assert old in text
    path = tmp_path / source.name
    path.write_text(text.replace(old, new))
    with pytest.raises(parser.ParseError, match=refused):
        cli.load_authored_ir(f"{path}:Root")


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
    IndexedCalls(),
    IndexedLaunch(),
)
CASES = (
    cases(FIXTURES / "inspection", FIXTURES / "tir", case_type=ParserCase, expected=EXPECTED)
    + NAMED
)
CASES = unique_parser_cases(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_parser_case(case: ParserCase) -> None:
    check_parser_case(case)
