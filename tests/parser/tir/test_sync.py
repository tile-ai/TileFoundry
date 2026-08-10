"""Pin mesh-scoped ``T.sync`` from parser through codegen text.

The parser emits ``Evaluate(Sync)`` with a compile-time mesh. Full CTAs use
``__syncthreads``; warp and aligned multi-warp subsets use predicated barriers.
Verification rejects missing, unrelated, non-contiguous, or unaligned scopes.
Codegen keeps nonparticipants out and assigns distinct named-barrier ids
([tir §1.5](docs/spec/tir.md#15-sync)).
"""

from __future__ import annotations

import pytest

import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tilefoundry import prim_func
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.ir.tir.sync import Sync, classify
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.target import CudaTarget


def _thread_mesh() -> Mesh:
    """A 128-thread block viewed as (4 warps, 32 lanes)."""
    return Mesh((Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t"))


def _cta_mesh() -> Mesh:
    return Mesh(topologies=(Topology("cta", 128),), layout=Layout(shape=(128,), strides=(1,)))


def _binding(name: str = "m") -> Var:
    return Var(type=TensorType.scalar(DType.i64), name=name)


def _syncs(body) -> list[Sync]:
    """Collect the Sync ops (in order) from a parsed body."""
    out: list[Sync] = []

    def walk(s) -> None:
        if isinstance(s, Sequential):
            for x in s.body:
                walk(x)
        elif isinstance(s, MeshScope):
            walk(s.body)
        elif isinstance(s, Evaluate) and isinstance(s.callable, Sync):
            out.append(s.callable)

    walk(body)
    return out


def test_parse_sync_builds_evaluate_wrapped_op() -> None:
    """``T.sync(m)`` lowers to ``Evaluate(Sync(mesh=m))`` carrying the mesh."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def kernel(a: Tensor[(128,), "f32"]):  # noqa: ARG001 — body-only smoke
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as m:
            T.sync(m)

    mesh_scope = kernel.body.body[0]
    assert isinstance(mesh_scope, MeshScope)
    ev = mesh_scope.body.body[0]
    assert isinstance(ev, Evaluate) and isinstance(ev.callable, Sync)
    assert ev.callable.mesh == mesh_scope.mesh


def test_parse_sync_slice_records_offset_and_extent() -> None:
    """Test parse sync slice records offset and extent.

    ``T.sync(m[1:3, :])`` records the participating sub-box (extents + slice
    origin) in a composed-layout ``layout``; the full sync's ``layout`` is a
    plain ``Layout``.
    """

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def kernel(a: Tensor[(128,), "f32"]):  # noqa: ARG001
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as m:
            T.sync(m)
            T.sync(m[0, :])
            T.sync(m[1:3, :])

    full, warp0, mid = _syncs(kernel.body)
    assert not isinstance(full.mesh.layout, ComposedLayout) and full.mesh.layout.shape == (4, 32)
    assert warp0.mesh.layout.outer.shape == (1, 32) and warp0.mesh.layout.offset == 0
    assert mid.mesh.layout.outer.shape == (2, 32) and mid.mesh.layout.offset == 32


def test_parse_sync_accepts_only_mesh() -> None:
    """A non-mesh ``T.sync`` argument fails to resolve to a mesh."""

    def kernel(a: Tensor[(128,), "f32"]):  # noqa: ARG001
        with Mesh(
            (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
        ) as m:  # noqa: F841
            T.sync(a)

    with pytest.raises(VerifyError):
        prim_func(target=CudaTarget("nvidia.h200_sxm"))(kernel)


def _scoped(mesh: Mesh, sync_mesh: Mesh) -> PrimFunction:
    return PrimFunction(
        name="fn",
        params=(),
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=_binding(),
                    body=Sequential(
                        body=(Evaluate(callable=Sync(mesh=sync_mesh), args=()), Return())
                    ),
                ),
            )
        ),
    )


def test_verify_accepts_full_and_sliced_sync() -> None:
    m = _thread_mesh()
    for sm in (m, m[1:3, :]):
        verify_prim_function(_scoped(m, sm))


def test_verify_rejects_sync_with_no_enclosing_mesh() -> None:
    pf = PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(Evaluate(callable=Sync(mesh=_cta_mesh()), args=()), Return())),
    )
    with pytest.raises(VerifyError, match="enclosing"):
        verify_prim_function(pf)


def test_verify_rejects_non_contiguous_slice() -> None:
    """A lane subset across warps (``m[:, 1:3]``) is not a contiguous thread interval.

    A lane subset across warps (``m[:, 1:3]``) is not a contiguous thread
    interval — rejected, not split into several barriers.
    """
    m = _thread_mesh()
    with pytest.raises(VerifyError, match="contiguous"):
        verify_prim_function(_scoped(m, m[:, 1:3]))


def test_verify_rejects_forged_subbox_exceeding_parent() -> None:
    """Test verify rejects forged subbox exceeding parent.

    A hand-forged slice that is not constructible by ``Mesh.__getitem__``
    (a (1, 64) sub-box of a (4, 32) parent) is rejected — the legal-slice proof
    bounds each sub-extent by the parent shape, not by field equality.
    """
    e = _thread_mesh()
    forged = Mesh(
        topologies=e.topologies,
        layout=ComposedLayout(inner=None, offset=0, outer=Layout((1, 64), (32, 1))),
        names=e.names,
    )
    with pytest.raises(VerifyError, match="enclosing"):
        verify_prim_function(_scoped(e, forged))


def test_verify_rejects_forged_topology_mismatch() -> None:
    """Test verify rejects forged topology mismatch.

    A forged sync mesh that shares the primary topology but differs in the
    full topology tuple is rejected (the proof compares the full tuple).
    """
    e = Mesh(
        topologies=(Topology("warp", 4), Topology("thread", 32)),
        layout=Layout(shape=(4, 32), strides=(32, 1)),
    )
    forged = Mesh(
        topologies=(Topology("warp", 4),),
        layout=ComposedLayout(inner=None, offset=0, outer=Layout((2, 32), (32, 1))),
        names=e.names,
    )
    with pytest.raises(VerifyError, match="enclosing"):
        verify_prim_function(_scoped(e, forged))


def test_verify_rejects_cross_warp_unaligned_slice() -> None:
    """A contiguous but cross-warp-unaligned range (lanes 16..47) is rejected."""
    m = Mesh((Topology("thread", 64),), Layout(shape=(64,), strides=(1,)))
    with pytest.raises(VerifyError, match="warp-aligned"):
        verify_prim_function(_scoped(m, m[16:48]))


def test_classify_rejects_partial_cta_slice() -> None:
    """Only the full cta mesh maps to the grid barrier.

    Only the full cta mesh maps to the grid barrier; a cta slice (a subset of
    CTAs) has no supported barrier and is rejected.
    """
    with pytest.raises(VerifyError, match="partial grid"):
        classify(_cta_mesh()[0:64])


def _emit(*meshes: Mesh) -> str:
    """Emit the syncs for *meshes* under one (fresh) kernel context."""
    ctx = CodegenContext()
    ctx.reset_barrier_ids()
    for mesh in meshes:
        ctx.emit_node(Evaluate(callable=Sync(mesh=mesh), args=()))
    return ctx.source()


def test_codegen_multi_warp_subset_emits_named_bar_sync_under_predicate() -> None:

    src = _emit(_thread_mesh()[1:3, :])
    assert "SyncKind::bar_sync, 32, 64, 0u," in src


def test_codegen_errors_when_named_barriers_exhausted() -> None:
    ctx = CodegenContext()
    ctx.reset_barrier_ids()
    for _ in range(15):
        ctx.alloc_barrier_id()
    with pytest.raises(ValueError, match="too many distinct named barriers"):
        ctx.alloc_barrier_id()
