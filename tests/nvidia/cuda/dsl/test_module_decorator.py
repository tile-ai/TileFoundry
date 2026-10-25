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

from tests.core._corpus import reload
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


def test_a_held_child_takes_its_own_owners_target_types():
    """A root types its own calls of a child under its Target; the child keeps its body.

    The children declare no Target, so their matmuls and loop accumulators,
    variants included, keep their stored types wherever they are held. A CUDA
    root's call of the direct child is an f32 register tile and a CPU root's
    call uses shared memory; neither changes the child. A roundtrip keeps the inherited
    Target and the call's storage, and a call the Target refuses fails where the
    root makes it, leaving the child's types as they were.
    """
    source = Path(__file__).parents[1] / "fixtures/logical/child_matmul_inherited_target.py"
    namespace = load_namespace(str(source))[0]
    K_LEN, K, M, N = (namespace[name] for name in ("K_LEN", "K", "M", "N"))

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

    cuda_root = load_authored_ir(f"{source}:ChildMatmulRoot")
    held = tuple(slot.elements[0] for slot in cuda_root.children.values())
    for child in held:
        assert child.target is None
        assert storages(child) == [StorageKind.UMAT] * 3
    assert list_call_paths(cuda_root)["direct.direct"].target is cuda_root.target
    assert call_storage(cuda_root, "gemm_on_chip") is StorageKind.RMEM
    before = [stored_types(child) for child in held]

    @module(entry="gemm", target=CpuTarget(), topologies=(Topology("cta", 1),))
    class _CpuRoot:
        child, direct, staged = held

        @func
        def gemm(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]):
            return child(a, b)  # noqa: F821 -- class-body binding

        @func
        def gemm_on_chip(a: Tensor[(M, K), "bf16"], b: Tensor[(K, N), "bf16"]):
            return direct(a, b)  # noqa: F821 -- class-body binding

        @func
        def gemm_staged(a: Tensor[(M, K_LEN), "bf16"], b: Tensor[(K_LEN, N), "bf16"]):
            return staged(a, b)  # noqa: F821 -- class-body binding

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

    @module(entry="run", topologies=(Topology("cta", 1),))
    class _Refused:
        @func
        def run(a: Tensor[(M, K), "f32"], b: Tensor[(K, N), "f32"]):
            with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
                acc = tf.matmul(
                    tf.reshard(a[:, 0:16], (M, 16), "smem"),
                    tf.reshard(b[0:16, :], (16, N), "smem"),
                )
                for k in tf.tile(16, K, 16):  # noqa: F405 -- authored tile loop
                    acc = acc + tf.matmul(
                        tf.reshard(a[:, k], (M, 16), "smem"),
                        tf.reshard(b[k, :], (16, N), "smem"),
                    )
                return tf.reshard(acc, (M, N), "gmem")

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


def test_prim_func_host_resolves_sibling_device_in_class_body():
    """Test prim func host resolves sibling device in class body.

    A ``@prim_func`` cpu host can ``launch`` a sibling cuda device kernel
    defined above it in the same ``@module`` class body — class-local sibling
    resolution works for prim_func, not only @func.
    """

    @module(entry="host")
    class _Launch:
        @prim_func(target=CudaTarget("nvidia.h200_sxm"))
        def dev(a: Tensor[(8,), "f32"]):  # noqa: ARG001
            with Mesh((Topology("thread", 8),), Layout(shape=(8,), strides=(1,))) as m:
                T.sync(m)

        @prim_func
        def host(a: Tensor[(8,), "f32"]):
            launch(dev, a, grid=(1, 1, 1), block=(8, 1, 1))  # noqa: F821

    assert isinstance(_Launch, Module)
    assert [fn.name for fn in _Launch.functions] == ["dev", "host"]
    assert _Launch.entry == "host"
    assert _Launch.lookup("host").target == CpuTarget()


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
