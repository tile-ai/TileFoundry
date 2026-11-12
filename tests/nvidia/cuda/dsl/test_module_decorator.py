"""``@module(entry=...)`` decorator: collect class-body DSL functions into a Module.

The decorator scans the class for ``@func`` / ``@prim_func`` members (which are
``hir.Function`` / ``tir.PrimFunction`` values), builds a ``Module`` in
definition order, and binds the decorated name to it. The class body is a pure
function container: a non-dunder, non-DSL member is rejected; ``entry`` is an
explicit, required name that must resolve to a collected function. A composed
member may call siblings defined above it (the call lowers to a ``Call``
targeting the sibling); forward references stay unresolved and fail loudly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import cli, inspection
from tilefoundry.dsl import *
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import nn
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

DSL = Path(__file__).parents[1] / "fixtures" / "dsl"


def test_held_child_takes_owner_target():
    """A root types its own calls of a child under its Target; the child keeps its body.

    The children declare no Target, so their matmuls and loop accumulators,
    variants included, keep their stored types wherever they are held. A CUDA
    root's call of the direct child is an f32 register tile and a CPU root's
    call uses shared memory; neither changes the child. A roundtrip keeps the inherited
    Target and the call's storage, and a call the Target refuses fails where the
    root makes it, leaving the child's types as they were.
    """
    namespace = cli.load_namespace(str(DSL / "child_matmul_cpu_root.py"))[0]
    K, M, N = (namespace[name] for name in ("K", "M", "N"))

    def storages(owner):
        found = []
        for function in owner.functions:
            for body in (variant.body for variant in (function, *function.variants)):
                for node in () if body is None else core.collect_exprs(body):
                    if isinstance(node, expr.Call) and isinstance(node.target, nn.MatMul):
                        found.append(node.type.storage)
                    if isinstance(node, core.LoopRegion):
                        found.extend(p.type.storage for p in node.params if p.name == "acc")
        return found

    def stored_types(owner):
        exprs = [
            node
            for function in owner.functions
            for variant in (function, *function.variants)
            for node in core.collect_exprs(variant.body)
        ]
        params = [
            param
            for node in exprs
            if isinstance(node, (core.LoopRegion, core.MeshRegion))
            for param in node.params
        ]
        return [(node, node.type) for node in (*exprs, *params)]

    def call_storage(owner, name):
        (function,) = (function for function in owner.functions if function.name == name)
        return function.body.type.storage

    cuda_root = namespace["ChildMatmulRoot"]
    held = tuple(cuda_root.children.values())
    for child in held:
        assert child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
    assert valuation.collect_functions(cuda_root)[core.FunctionRef(("direct",), "direct")].target is cuda_root.target
    assert call_storage(cuda_root, "gemm_on_chip") is StorageKind.RMEM
    before = [stored_types(child) for child in held]

    _CpuRoot = namespace["ChildMatmulCpuRoot"]
    children = tuple(_CpuRoot.children.values())
    for child, original, types in zip(children, held, before, strict=True):
        assert child is original and child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
        assert all(node.type is stored for node, stored in types)
    assert call_storage(_CpuRoot, "gemm_on_chip") is StorageKind.SMEM
    assert call_storage(cuda_root, "gemm_on_chip") is StorageKind.RMEM
    callees = [
        node.target
        for function in _CpuRoot.functions
        for node in core.collect_exprs(function.body)
        if isinstance(node, expr.Call)
    ]
    assert all(any(target == core.FunctionRef((key,), None) for target in callees) for key in _CpuRoot.children)

    imported = reload(inspection.as_script(cuda_root), "ChildMatmulRoot")
    for child in imported.children.values():
        assert child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
    assert valuation.collect_functions(imported)[core.FunctionRef(("direct",), "direct")].target == cuda_root.target
    assert call_storage(imported, "gemm_on_chip") is StorageKind.RMEM

    _Refused = cli.load_authored_ir(str(DSL / "refused_child_matmul.py"))
    parsed = stored_types(_Refused)
    refused = _Refused
    with pytest.raises(expr.VerifyError, match="no nvidia.h200_sxm MMA reads lhs f32 smem"):

        @module(
            entry="call",
            target=cuda.Target("nvidia.h200_sxm"),
            topologies=(Topology("cta", 1),),
        )
        class _RefusingRoot:
            child = refused

            @func
            def call(a: Tensor[(M, K), "f32"], b: Tensor[(K, N), "f32"]):
                return child(a, b)

    assert storages(_Refused) == [StorageKind.UMAT] * 3
    assert all(node.type is stored for node, stored in parsed)


class ModuleDecoratorCase(Case):
    """A program whose ``@module`` class body the decorator collected into a Module."""


@case
class PrimFuncSiblingLaunch(ModuleDecoratorCase):
    """Test prim func host resolves sibling device in class body.

    A ``@prim_func`` cpu host can ``launch`` a sibling cuda device kernel
    defined above it in the same ``@module`` class body — class-local sibling
    resolution works for prim_func, not only @func.
    """

    path: Path = DSL / "prim_func_sibling_launch.py"

    def guard(self, state, problems):
        assert isinstance(state, core.Module)
        assert [fn.name for fn in state.functions] == ["dev", "host"]
        assert state.entry == "host"
        assert state.lookup("host").target == cpu.Target()


@pytest.mark.parametrize(
    ("fixture", "refused"),
    [
        ("list_slot", "use ModuleList"),
        ("tuple_slot", "use ModuleList"),
        ("zero_count", "positive integer"),
        ("fractional_count", "positive integer"),
        ("empty_tuple", "non-empty tuple"),
        ("non_module", "requires Modules"),
        ("aliased_slot", "layers.*other"),
        ("large_index", "layers.*N=3"),
        ("negative_index", "layers.*N=3"),
        ("fractional_index", "integer scalar"),
        ("indexed_single_slot", "single slot.*cannot be indexed"),
        ("unindexed_list", "must be indexed"),
        ("mixed_definitions", "same definition"),
        ("non_entry_call", "only calling an instance entry"),
        ("extra_activation", "arity mismatch.*1 activation.*passed 2"),
        ("missing_weight", "arity mismatch.*2 parameter.*passed 1"),
        ("missing_entry", "declares no entry"),
        ("unbound_slot", "instance calls must use a slot name"),
    ],
)
def test_module_rejects_invalid_instances(fixture, refused):
    source = Path(__file__).parents[1] / f"fixtures/diagnostics/parser_module_instance_{fixture}.py"
    with pytest.raises((TypeError, ValueError, expr.VerifyError), match=refused):
        cli.load_authored_ir(str(source))


def test_entryless_module_occupies_slot():
    scaled_modules = Path(__file__).parents[1] / "fixtures/shapes/scaled_modules.py"
    ScaledChild = cli.load_authored_ir(f"{scaled_modules}:ScaledChild")
    slot = dataclasses.replace(ScaledChild, entry=None)
    root = core.Module("Container", (), None, children={"unused": slot})
    assert dataclasses.replace(root).children["unused"] is slot
    assert slot.entry is None


CASES = special(ModuleDecoratorCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_module_decorator_case(case: ModuleDecoratorCase) -> None:
    check_case(case)
