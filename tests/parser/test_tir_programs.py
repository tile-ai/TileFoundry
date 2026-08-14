"""What the TIR half of the parser must build, asserted on the parsed nodes.

There is no golden here: ``tilefoundry.inspection`` prints HIR and nothing
prints a ``PrimFunction``, so these read the programs in ``programs.py``
directly. What is left out of the programs is left out on purpose — a
hand-forged node fed to the verifier is not something the parser produced, and
those are rows in ``error_cases.py``.
"""

from __future__ import annotations

import tilefoundry.codegen.cuda  # noqa: F401 — trigger emitter autodiscovery
from tests.parser.error_cases import scoped_sync, thread_mesh
from tests.parser.programs import (
    M_CTA,
    tir_atom_fragment_in_a_warp_scope,
    tir_dynamic_device,
    tir_effect_form_selector,
    tir_host_entry,
    tir_param_layout_sugar,
    tir_static_atom_bindings,
    tir_static_device,
    tir_sync_scopes,
)
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.dsl import T
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, MeshScope, Sequential
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout
from tilefoundry.ir.types.shard.layout import ComposedLayout
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind

_ATOM = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)


def _shape_scalars(prim_fn) -> list[str]:
    return [p.name for p in prim_fn.params if "_shape_" in p.name]


def test_a_dynamic_device_dim_injects_a_hidden_shape_scalar() -> None:
    """A device kernel reading a ``DimVar`` axis declares the i32 extent alongside it.

    That is the ABI the HIR-to-TIR lowering appends, so codegen can plumb the
    runtime extent.
    """
    assert _shape_scalars(tir_dynamic_device) == ["a_shape_0"]
    scalar = next(p for p in tir_dynamic_device.params if p.name == "a_shape_0")
    assert isinstance(scalar.type, TensorType)
    assert scalar.type.shape == ()
    assert scalar.type.dtype is DType.i32


def test_a_host_entry_and_a_static_kernel_stay_unpolluted() -> None:
    """A host entry reads its shapes from its tensor argument at launch time.

    A device kernel over only static dims has nothing to plumb, so neither
    grows a hidden scalar.
    """
    assert _shape_scalars(tir_host_entry) == []
    assert [p.name for p in tir_host_entry.params] == ["a"]
    assert _shape_scalars(tir_static_device) == []


def test_a_prim_func_param_takes_the_same_layout_sugar_as_a_func_param() -> None:
    """``8192 @ cta`` canonicalises into ``(128, 64)`` on a device parameter too.

    ``_build_params`` is one walk for both dialects, so this is the TIR twin of
    the HIR ``int-at-single-axis-mesh`` case.
    """
    assert tir_param_layout_sugar.params[0].type == TensorType(
        shape=(1, 8192),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((1, 128, 64), (8192, 64, 1)), attrs=(Split(1),), mesh=M_CTA
        ),
    )


def test_a_trailing_underscore_selects_the_effect_form() -> None:
    """A bare ``copy_(...)`` strips its suffix and resolves ``Copy`` from the T dialect.

    It is unresolved through the closure, so it goes through
    ``dispatch.resolve_callable`` and lands on the same statement ``T.copy``
    would have produced.
    """
    assert isinstance(tir_effect_form_selector.body, Sequential)
    (stmt,) = tir_effect_form_selector.body.body
    assert isinstance(stmt, Evaluate)
    assert isinstance(stmt.callable, Copy)
    assert stmt.args[0].name == "a"
    assert stmt.args[1].name == "b"


def test_an_atom_binding_is_a_compile_time_value_not_a_letstmt() -> None:
    """``op = ...`` and ``atom = ...`` bind statically, leaving the body empty."""
    assert all(not isinstance(s, LetStmt) for s in tir_static_atom_bindings.body.body)
    assert tir_static_atom_bindings.body.body == ()

    atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
    assert isinstance(atom, MmaAtom)
    assert (atom.A, atom.B, atom.C) == (_ATOM.A, _ATOM.B, _ATOM.C)
    assert atom.required_scope is _ATOM.required_scope


def test_a_fragment_alloc_in_a_valid_scope_keeps_the_atom_layout() -> None:
    """A scope that passes the use-point check leaves the fragment layout alone.

    The resolver returns the atom's own object rather than rebinding it to the
    caller's mesh, whatever that mesh names its axes.
    """
    mesh_scope = next(
        s for s in tir_atom_fragment_in_a_warp_scope.body.body if isinstance(s, MeshScope)
    )
    let = next(s for s in mesh_scope.body.body if isinstance(s, LetStmt))
    assert let.var.type.layout is _ATOM.A


def _syncs(body) -> list[Sync]:
    """The Sync ops, in order, from a parsed body."""
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


def test_a_mesh_scoped_sync_records_the_participating_sub_box() -> None:
    """``T.sync(m)`` lowers to ``Evaluate(Sync(mesh=m))`` carrying a compile-time mesh.

    A sliced sync records its extents and slice origin in a composed layout; the
    full sync's layout stays a plain ``Layout``.
    """
    mesh_scope = tir_sync_scopes.body.body[0]
    assert isinstance(mesh_scope, MeshScope)
    first = mesh_scope.body.body[0]
    assert isinstance(first, Evaluate) and isinstance(first.callable, Sync)
    assert first.callable.mesh == mesh_scope.mesh

    full, warp0, mid = _syncs(tir_sync_scopes.body)
    assert not isinstance(full.mesh.layout, ComposedLayout)
    assert full.mesh.layout.shape == (4, 32)
    assert warp0.mesh.layout.outer.shape == (1, 32) and warp0.mesh.layout.offset == 0
    assert mid.mesh.layout.outer.shape == (2, 32) and mid.mesh.layout.offset == 32


def test_verify_accepts_a_full_and_a_sliced_sync() -> None:
    """The scopes the verifier must accept, next to the ones it must refuse."""
    mesh = thread_mesh()
    for participating in (mesh, mesh[1:3, :]):
        verify_prim_function(scoped_sync(mesh, participating))


def test_codegen_puts_a_multi_warp_subset_behind_a_named_barrier() -> None:
    """An aligned multi-warp subset emits a predicated ``bar_sync``, not ``__syncthreads``."""
    ctx = CodegenContext()
    ctx.reset_barrier_ids()
    ctx.emit_node(Evaluate(callable=Sync(mesh=thread_mesh()[1:3, :]), args=()))

    assert "SyncKind::bar_sync, 32, 64, 0u," in ctx.source()
