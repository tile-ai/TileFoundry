"""Mesh-scoped ``T.sync(m)`` — parser → IR → verify → codegen-text.

Covers the contract:
- parser lowers ``T.sync(m)`` / ``T.sync(m[slice])`` to an ``Evaluate``-wrapped
  ``Sync`` op carrying the (possibly sliced) mesh as a compile-time attribute;
- the participant set is derived from the mesh: full CTA → ``__syncthreads()``,
  a single-warp subset → ``__syncwarp(mask)`` under a predicate, a warp-aligned
  multi-warp subset → a named ``bar.sync`` under a predicate;
- verify rejects a sync with no enclosing mesh, a mesh not derived from any
  enclosing scope, a non-contiguous slice, and a cross-warp-unaligned slice;
- codegen emits the predicate so non-participants never run the barrier, every
  participant runs the same id/count, and distinct named barriers get distinct
  ids within a kernel.
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
from tilefoundry.ir.tir.sync import Sync, SyncBarrier, classify
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout


def _thread_mesh() -> Mesh:
    """A 128-thread block viewed as (4 warps, 32 lanes)."""
    return Mesh(Topology("thread", 128), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t"))


def _cta_mesh() -> Mesh:
    return Mesh(topology=Topology("cta", 128), layout=Layout(shape=(128,), strides=(1,)))


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


# --- parser → IR ---------------------------------------------------------


def test_parse_sync_builds_evaluate_wrapped_op() -> None:
    """``T.sync(m)`` lowers to ``Evaluate(Sync(mesh=m))`` carrying the mesh."""

    @prim_func(target="cuda")
    def kernel(a: Tensor[(128,), "f32"]):  # noqa: ARG001 — body-only smoke
        with Mesh(Topology("thread", 128), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")) as m:
            T.sync(m)

    mesh_scope = kernel.body.body[0]
    assert isinstance(mesh_scope, MeshScope)
    ev = mesh_scope.body.body[0]
    assert isinstance(ev, Evaluate) and isinstance(ev.callable, Sync)
    assert ev.callable.mesh == mesh_scope.mesh


def test_parse_sync_slice_records_offset_and_extent() -> None:
    """``T.sync(m[1:3, :])`` records the participating sub-box (extents + slice
    origin) in a composed-layout ``layout``; the full sync's ``layout`` is a
    plain ``Layout``."""

    @prim_func(target="cuda")
    def kernel(a: Tensor[(128,), "f32"]):  # noqa: ARG001
        with Mesh(Topology("thread", 128), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")) as m:
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
        with Mesh(Topology("thread", 128), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")) as m:  # noqa: F841
            T.sync(a)

    with pytest.raises(VerifyError):
        prim_func(target="cuda")(kernel)


# --- verify --------------------------------------------------------------


def _scoped(mesh: Mesh, sync_mesh: Mesh) -> PrimFunction:
    return PrimFunction(
        name="fn",
        params=(),
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=_binding(),
                    body=Sequential(body=(Evaluate(callable=Sync(mesh=sync_mesh), args=()), Return())),
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
    """A lane subset across warps (``m[:, 1:3]``) is not a contiguous thread
    interval — rejected, not split into several barriers."""
    m = _thread_mesh()
    with pytest.raises(VerifyError, match="contiguous"):
        verify_prim_function(_scoped(m, m[:, 1:3]))


def test_verify_rejects_forged_subbox_exceeding_parent() -> None:
    """A hand-forged slice that is not constructible by ``Mesh.__getitem__``
    (a (1, 64) sub-box of a (4, 32) parent) is rejected — the legal-slice proof
    bounds each sub-extent by the parent shape, not by field equality."""
    e = _thread_mesh()  # (4, 32)
    forged = Mesh(
        topology=e.topology,
        layout=ComposedLayout(inner=None, offset=0, outer=Layout((1, 64), (32, 1))),
        names=e.names,
        topologies=e.topologies,
    )
    with pytest.raises(VerifyError, match="enclosing"):
        verify_prim_function(_scoped(e, forged))


def test_verify_rejects_forged_topology_mismatch() -> None:
    """A forged sync mesh that shares the primary topology but differs in the
    full topology tuple is rejected (the proof compares the full tuple)."""
    e = Mesh(
        topology=[Topology("warp", 4), Topology("thread", 32)],
        layout=Layout(shape=(4, 32), strides=(32, 1)),
    )
    forged = Mesh(
        topology=Topology("warp", 4),
        layout=ComposedLayout(inner=None, offset=0, outer=Layout((2, 32), (32, 1))),
        names=e.names,
        topologies=(Topology("warp", 4),),
    )
    with pytest.raises(VerifyError, match="enclosing"):
        verify_prim_function(_scoped(e, forged))


def test_verify_rejects_cross_warp_unaligned_slice() -> None:
    """A contiguous but cross-warp-unaligned range (lanes 16..47) is rejected."""
    # 64-thread block as (2 warps, 32 lanes); slice 16 lanes of warp 0 + 16 of
    # warp 1 → contiguous [16, 48) but not warp-aligned.
    m = Mesh(Topology("thread", 64), Layout(shape=(64,), strides=(1,)))
    with pytest.raises(VerifyError, match="warp-aligned"):
        verify_prim_function(_scoped(m, m[16:48]))


# --- classification ------------------------------------------------------


def test_classify_derives_barrier_from_participants() -> None:
    m = _thread_mesh()
    assert classify(m) is SyncBarrier.SYNCTHREADS          # whole block, 4 warps
    assert classify(m[0, :]) is SyncBarrier.SYNCWARP        # one warp
    assert classify(m[1:3, :]) is SyncBarrier.BAR_SYNC      # 2-warp subset


def test_classify_full_cta_scope_mesh_is_grid_barrier() -> None:
    """A mesh over the ``cta`` topology synchronizes CTAs across the grid — the
    grid-wide software barrier, not a within-block ``__syncthreads``."""
    assert classify(_cta_mesh()) is SyncBarrier.GRID


def test_classify_rejects_partial_cta_slice() -> None:
    """Only the full cta mesh maps to the grid barrier; a cta slice (a subset of
    CTAs) has no supported barrier and is rejected."""
    with pytest.raises(VerifyError, match="partial grid"):
        classify(_cta_mesh()[0:64])


# --- codegen text --------------------------------------------------------


def _emit(*meshes: Mesh) -> str:
    """Emit the syncs for *meshes* under one (fresh) kernel context."""
    ctx = CodegenContext()
    ctx.reset_barrier_ids()
    for mesh in meshes:
        ctx.emit_node(Evaluate(callable=Sync(mesh=mesh), args=()))
    return ctx.source()


def test_codegen_full_cta_emits_syncthreads() -> None:
    assert (
        _emit(_thread_mesh()).strip()
        == "tilefoundry::ops::sync<tilefoundry::ops::SyncKind::syncthreads>();"
    )


def test_codegen_cta_scope_mesh_emits_grid_barrier() -> None:
    assert (
        _emit(_cta_mesh()).strip()
        == "tilefoundry::ops::sync<tilefoundry::ops::SyncKind::grid>"
        "(tilefoundry::tf_grid_bar_state);"
    )


def test_codegen_single_warp_subset_emits_masked_syncwarp_under_predicate() -> None:
    # The uniform entry carries the participant geometry (base 0, count 32, full
    # lane mask) as template parameters; the predicate lives in the runtime.
    src = _emit(_thread_mesh()[0, :])
    assert "SyncKind::syncwarp_masked, 0, 32, 0xffffffffu>();" in src


def test_codegen_multi_warp_subset_emits_named_bar_sync_under_predicate() -> None:
    # Warps 1-2 → base 32, count 64; the named-barrier id + predicate live in the
    # runtime template.
    src = _emit(_thread_mesh()[1:3, :])
    assert "SyncKind::bar_sync, 32, 64, 0u," in src


def test_codegen_allocates_distinct_barrier_ids_per_kernel() -> None:
    """Two distinct multi-warp syncs in one kernel get distinct named ids."""
    m = Mesh(Topology("thread", 128), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t"))
    src = _emit(m[0:2, :], m[2:4, :])
    # The named-barrier id is the last template argument of the uniform entry.
    assert "SyncKind::bar_sync, 0, 64, 0u, 1>();" in src
    assert "SyncKind::bar_sync, 64, 64, 0u, 2>();" in src


def test_codegen_errors_when_named_barriers_exhausted() -> None:
    ctx = CodegenContext()
    ctx.reset_barrier_ids()
    for _ in range(15):
        ctx.alloc_barrier_id()
    with pytest.raises(ValueError, match="too many distinct named barriers"):
        ctx.alloc_barrier_id()
