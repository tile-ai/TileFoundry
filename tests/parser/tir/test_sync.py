"""Pin mesh-scoped ``T.sync`` from parser through codegen text.

The parser emits ``Evaluate(Sync)`` with a compile-time mesh. Full CTAs use
``__syncthreads``; warp and aligned multi-warp subsets use predicated barriers.
The scopes verification must reject — missing, unrelated, non-contiguous, or
unaligned — are rows in ``error_cases.py``. Codegen keeps nonparticipants out
and assigns distinct named-barrier ids ([tir §1.5](docs/spec/tir.md#15-sync)).
"""

from __future__ import annotations

import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tests.parser.error_cases import scoped_sync as _scoped
from tests.parser.error_cases import thread_mesh as _thread_mesh
from tilefoundry import prim_func
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Sequential
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.target import CudaTarget


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


def test_verify_accepts_full_and_sliced_sync() -> None:
    m = _thread_mesh()
    for sm in (m, m[1:3, :]):
        verify_prim_function(_scoped(m, sm))


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
