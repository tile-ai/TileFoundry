"""Platform namespace ``T.cuda.mma`` + ``MmaOpSpec`` / ``MmaAtom``.

Covers:
- ``T.cuda`` resolves to a platform sub-namespace; ``T.cuda.mma.<NAME>`` is an
  ``MmaOpSpec`` and ``T.cuda.mma.atom(op=...)`` an ``MmaAtom`` exposing A/B/C;
- the op / atom descriptors are target-owned in ``ir/tir/cuda/nn`` alongside
  the CUDA instructions + fragments;
- in a ``@prim_func`` body, ``op = ...`` / ``atom = ...`` are compile-time
  static bindings — no ``LetStmt`` is emitted — and a module-level alias is
  the same value.
"""
from __future__ import annotations

import pytest

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.tir.cuda.nn.mma import make_atom
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom, MmaOpSpec
from tilefoundry.ir.tir.stmts import LetStmt, MeshScope
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind

# Module-level pre-instantiated alias (equivalent to building it inline).
_OP = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN
_ATOM = T.cuda.mma.atom(op=_OP)


# --- namespace resolution + descriptors ---------------------------------


def test_cuda_namespace_resolves() -> None:
    assert T.cuda is T.cuda  # stable singleton
    assert hasattr(T.cuda, "mma")


def test_named_op_resolves_to_op_spec() -> None:
    op = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN
    assert isinstance(op, MmaOpSpec)
    assert op.name == "SM80_16x8x16_F32BF16BF16F32_TN"
    assert op.shape_mnk == (16, 8, 16)
    assert (op.dtype_a, op.dtype_b, op.dtype_c) == (DType.bf16, DType.bf16, DType.f32)
    assert op.operand_layout == "TN"


def test_atom_builder_exposes_layout_contract_and_scope() -> None:
    atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
    assert isinstance(atom, MmaAtom)
    # A/B/C are the atom's layout contracts (the pinned fragment shards),
    # returned as-is — not rebound to any caller mesh. A second build shares
    # the same shard objects (the resolver returns the canonical fragments).
    assert atom.A is _ATOM.A
    assert atom.B is _ATOM.B
    assert atom.C is _ATOM.C
    # required_scope is the thread participation contract (32 lanes as (4,8)).
    assert atom.required_scope is _ATOM.required_scope


def test_descriptors_are_target_owned() -> None:
    """The MMA op / atom descriptors are target-owned: MmaOpSpec / MmaAtom live
    under the CUDA target surface in ir/tir/cuda/nn alongside the concrete
    instructions and their fragment layouts."""
    assert MmaOpSpec.__module__ == "tilefoundry.ir.tir.cuda.nn.mma_atom"
    assert MmaAtom.__module__ == "tilefoundry.ir.tir.cuda.nn.mma_atom"


def test_make_atom_rejects_unregistered_op() -> None:
    bogus = MmaOpSpec(
        name="FAKE", shape_mnk=(8, 8, 8),
        dtype_a=DType.f16, dtype_b=DType.f16, dtype_c=DType.f32,
        operand_layout="TN",
    )
    with pytest.raises(KeyError, match="no fragment layouts"):
        make_atom(bogus)


# --- compile-time static binding in a @prim_func body -------------------


def test_infunc_op_and_atom_emit_no_letstmt() -> None:
    """`op = T.cuda.mma.<NAME>` and `atom = T.cuda.mma.atom(op=op)` are
    static bindings: neither lowers to a LetStmt, so a body with only these
    assignments is empty."""

    @prim_func(target="cuda")
    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        op = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN
        atom = T.cuda.mma.atom(op=op)  # noqa: F841

    assert all(not isinstance(s, LetStmt) for s in kernel.body.body)
    assert kernel.body.body == ()


def test_module_level_alias_equals_surface_atom() -> None:
    """A module-level pre-instantiated alias is the same value as building the
    atom inline through the surface."""
    assert _ATOM == make_atom(_OP)
    assert _ATOM == T.cuda.mma.atom(op=_OP)


# --- atom.A use-point scope check (check, not bind) ---------------------
#
# `atom.A` is returned as-is (the atom's layout contract). The parser checks at
# the use point that the enclosing mesh scope can host the atom — structurally,
# independent of binding/axis names. The match is the same predicate `T.mma`
# verify reuses.


def _first_alloc_layout(prim_fn) -> ShardLayout:
    """Pull the ShardLayout off the single alloc'd fragment inside the kernel's
    mesh scope."""
    mesh_scope = next(s for s in prim_fn.body.body if isinstance(s, MeshScope))
    let = next(s for s in mesh_scope.body.body if isinstance(s, LetStmt))
    return let.var.type.layout


def _alloc_frag_kernel(topology, mesh_layout):
    """A kernel that allocs a fragment via `atom.A` inside the given scope."""
    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
        with Mesh(topology, mesh_layout) as warp:  # noqa: F841
            frag = T.alloc_tensor(  # noqa: F841
                TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A,
                           storage=StorageKind.RMEM)
            )
    return kernel


def test_atom_A_in_valid_warp_scope_returns_layout_as_is() -> None:
    """A valid (4,8)/(1,4) thread(32) scope passes the check; `atom.A` is the
    atom's contract layout, returned unchanged (no rebind)."""
    kernel = prim_func(target="cuda")(
        _alloc_frag_kernel(
            Topology("thread", 32), Layout(shape=(4, 8), strides=(1, 4))
        )
    )
    assert _first_alloc_layout(kernel) is _ATOM.A


def test_atom_A_binding_name_irrelevant() -> None:
    """The match is structural on the thread-value layout, not the binding var
    name: the `with Mesh(...) as warp` name is never checked."""
    kernel = prim_func(target="cuda")(
        _alloc_frag_kernel(
            Topology("thread", 32), Layout(shape=(4, 8), strides=(1, 4))
        )
    )
    assert _first_alloc_layout(kernel) is _ATOM.A


def test_atom_A_rejects_flat_32_scope() -> None:
    """A flat (32,) scope cannot host the 2-axis (4,8) fragment → reject."""
    with pytest.raises(VerifyError, match="required thread scope"):
        prim_func(target="cuda")(
            _alloc_frag_kernel(
                Topology("thread", 32), Layout(shape=(32,), strides=(1,))
            )
        )


def test_atom_A_rejects_cta_scope() -> None:
    """A `cta` scope (even with a (4,8) layout) is not a warp scope → reject."""
    with pytest.raises(VerifyError, match="required thread scope"):
        prim_func(target="cuda")(
            _alloc_frag_kernel(
                Topology("cta", 32), Layout(shape=(4, 8), strides=(1, 4))
            )
        )


def test_atom_A_rejects_wrong_thread_count() -> None:
    """A 64-lane thread scope does not match the 32-lane atom → reject."""
    with pytest.raises(VerifyError, match="required thread scope"):
        prim_func(target="cuda")(
            _alloc_frag_kernel(
                Topology("thread", 64), Layout(shape=(8, 8), strides=(1, 8))
            )
        )


def test_atom_A_rejects_inconsistent_thread_mesh() -> None:
    """A thread(64) topology carrying a 32-element layout is malformed → reject."""
    with pytest.raises(VerifyError, match="required thread scope"):
        prim_func(target="cuda")(
            _alloc_frag_kernel(
                Topology("thread", 64), Layout(shape=(4, 8), strides=(1, 4))
            )
        )


def test_atom_A_outside_mesh_scope_is_rejected() -> None:
    """A fragment used with no enclosing mesh scope is rejected."""

    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
        frag = T.alloc_tensor(  # noqa: F841
            TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A,
                       storage=StorageKind.RMEM)
        )

    with pytest.raises(VerifyError, match="must be used inside a `with Mesh"):
        prim_func(target="cuda")(kernel)
