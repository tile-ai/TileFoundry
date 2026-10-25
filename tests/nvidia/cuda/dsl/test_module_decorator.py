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
from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

DSL = Path(__file__).parents[1] / "fixtures" / "dsl"


def test_a_held_child_takes_its_own_owners_target_types():
    """A root types its own calls of a child under its Target; the child keeps its body.

    The children declare no Target, so their matmuls and loop accumulators,
    variants included, keep their stored types wherever they are held. A CUDA
    root's call of the direct child is an f32 register tile and a CPU root's
    call uses shared memory; neither changes the child. A roundtrip keeps the inherited
    Target and the call's storage, and a call the Target refuses fails where the
    root makes it, leaving the child's types as they were.
    """
    namespace = load_namespace(str(DSL / "child_matmul_cpu_root.py"))[0]
    K, M, N = (namespace[name] for name in ("K", "M", "N"))

    def storages(owner):
        found = []
        for function in owner.functions:
            for body in (variant.body for variant in (function, *function.variants)):
                for expr in () if body is None else collect_exprs(body):
                    if isinstance(expr, Call) and isinstance(expr.target, MatMul):
                        found.append(expr.type.storage)
                    if isinstance(expr, LoopRegion):
                        found.extend(p.type.storage for p in expr.params if p.name == "acc")
        return found

    def stored_types(owner):
        exprs = [
            expr
            for function in owner.functions
            for variant in (function, *function.variants)
            for expr in collect_exprs(variant.body)
        ]
        params = [
            param
            for expr in exprs
            if isinstance(expr, (LoopRegion, MeshRegion))
            for param in expr.params
        ]
        return [(expr, expr.type) for expr in (*exprs, *params)]

    def call_storage(owner, name):
        (function,) = (function for function in owner.functions if function.name == name)
        return function.body.type.storage

    cuda_root = namespace["ChildMatmulRoot"]
    held = tuple(slot.elements[0] for slot in cuda_root.children.values())
    for child in held:
        assert child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
    assert list_call_paths(cuda_root)["direct.direct"].target is cuda_root.target
    assert call_storage(cuda_root, "gemm_on_chip") is StorageKind.RMEM
    before = [stored_types(child) for child in held]

    _CpuRoot = namespace["ChildMatmulCpuRoot"]
    children = tuple(slot.elements[0] for slot in _CpuRoot.children.values())
    for child, original, types in zip(children, held, before, strict=True):
        assert child is original and child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
        assert all(expr.type is stored for expr, stored in types)
    assert call_storage(_CpuRoot, "gemm_on_chip") is StorageKind.SMEM
    assert call_storage(cuda_root, "gemm_on_chip") is StorageKind.RMEM
    callees = [
        expr.target
        for function in _CpuRoot.functions
        for expr in collect_exprs(function.body)
        if isinstance(expr, Call)
    ]
    assert all(any(target is slot for target in callees) for slot in _CpuRoot.children.values())

    imported = reload(as_script(cuda_root), "ChildMatmulRoot")
    for slot in imported.children.values():
        child = slot.elements[0]
        assert child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
    assert list_call_paths(imported)["direct.direct"].target == cuda_root.target
    assert call_storage(imported, "gemm_on_chip") is StorageKind.RMEM

    _Refused = load_authored_ir(str(DSL / "refused_child_matmul.py"))
    parsed = stored_types(_Refused)
    refused = _Refused
    with pytest.raises(VerifyError, match="no nvidia.h200_sxm MMA reads lhs f32 smem"):

        @module(
            entry="call",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 1),),
        )
        class _RefusingRoot:
            child = refused

            @func
            def call(a: Tensor[(M, K), "f32"], b: Tensor[(K, N), "f32"]):
                return child(a, b)  # noqa: F821 -- class-body binding

    assert storages(_Refused) == [StorageKind.UMAT] * 3
    assert all(expr.type is stored for expr, stored in parsed)


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
        assert isinstance(state, Module)
        assert [fn.name for fn in state.functions] == ["dev", "host"]
        assert state.entry == "host"
        assert state.lookup("host").target == CpuTarget()


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
def test_module_rejects_invalid_instance_declarations_and_calls(fixture, refused):
    source = Path(__file__).parents[1] / f"fixtures/diagnostics/parser_module_instance_{fixture}.py"
    with pytest.raises((TypeError, ValueError, VerifyError), match=refused):
        load_authored_ir(str(source))


def test_an_unused_module_without_entry_can_occupy_a_slot_and_be_replaced():
    scaled_modules = Path(__file__).parents[1] / "fixtures/shapes/scaled_modules.py"
    ScaledChild = load_authored_ir(f"{scaled_modules}:ScaledChild")
    slot = ModuleList(dataclasses.replace(ScaledChild, entry=None))
    root = Module("Container", (), None, children={"unused": slot})
    assert dataclasses.replace(root).children["unused"] is slot
    assert slot.elements[0].entry is None


CASES = special(ModuleDecoratorCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_module_decorator_case(case: ModuleDecoratorCase) -> None:
    check_case(case)
