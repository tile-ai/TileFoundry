"""Platform namespace ``T.cuda.mma`` + ``MmaAtom``.

Covers:
- ``T.cuda.mma.atom(op=...)`` builds an ``MmaAtom`` exposing A/B/C layout
  contracts (the pinned fragment shards) and a required thread scope;
- in a ``@prim_func`` body, ``op = ...`` / ``atom = ...`` are compile-time
  static bindings — no ``LetStmt`` is emitted;
- ``atom.A`` (a fragment layout) used to alloc a register tensor is checked at
  the use point against the enclosing mesh scope. That check is
  ``mesh_scope_matches_required_scope``, which matches on thread participation
  (program level + static lane count + the exact required thread-value layout
  shape and strides) and ignores binding-var names, axis names and mesh
  identity; the cases below are its accept/reject categories seen through the
  surface that uses it.
"""
from __future__ import annotations

import pytest

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.tir.stmts import LetStmt, MeshScope
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget

# Module-level pre-instantiated alias (equivalent to building it inline).
_ATOM = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)


def _alloc_frag_kernel(topology, mesh_layout, names=()):
    """A kernel that allocs a fragment via `atom.A` inside the given scope."""
    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
        with Mesh(topology, mesh_layout, names=names) as warp:  # noqa: F841
            frag = T.alloc_tensor(  # noqa: F841
                TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A,
                           storage=StorageKind.RMEM)
            )
    return kernel


def _first_alloc_layout(prim_fn) -> ShardLayout:
    """Pull the ShardLayout off the single alloc'd fragment inside the kernel's
    mesh scope."""
    mesh_scope = next(s for s in prim_fn.body.body if isinstance(s, MeshScope))
    let = next(s for s in mesh_scope.body.body if isinstance(s, LetStmt))
    return let.var.type.layout


def test_a_valid_scope_takes_the_atom_contract_as_is() -> None:
    """A/B/C and ``required_scope`` are the atom's contracts, canonical across
    builds — the resolver returns the same fragment objects rather than rebinding
    them to a caller mesh. A valid (4,8)/(1,4) thread(32) scope passes the
    use-point check whatever its axes are named, and the alloc'd fragment carries
    the atom's own layout unchanged."""
    atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
    assert isinstance(atom, MmaAtom)
    assert (atom.A, atom.B, atom.C) == (_ATOM.A, _ATOM.B, _ATOM.C)
    assert atom.required_scope is _ATOM.required_scope

    kernel = prim_func(target=CudaTarget("nvidia.h200_sxm"))(
        _alloc_frag_kernel(
            Topology("thread", 32), Layout(shape=(4, 8), strides=(1, 4)),
            names=("warp", "lane"),
        )
    )
    assert _first_alloc_layout(kernel) is _ATOM.A


def test_infunc_op_and_atom_emit_no_letstmt() -> None:
    """`op = T.cuda.mma.<NAME>` and `atom = T.cuda.mma.atom(op=op)` are
    static bindings: neither lowers to a LetStmt, so a body with only these
    assignments is empty."""

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        op = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN
        atom = T.cuda.mma.atom(op=op)  # noqa: F841

    assert all(not isinstance(s, LetStmt) for s in kernel.body.body)
    assert kernel.body.body == ()


@pytest.mark.parametrize(
    ("topology", "layout"),
    [
        (Topology("thread", 32), Layout(shape=(32,), strides=(1,))),
        (Topology("thread", 32), Layout(shape=(4, 8), strides=(8, 1))),
        (Topology("cta", 32), Layout(shape=(4, 8), strides=(1, 4))),
        (Topology("thread", 64), Layout(shape=(8, 8), strides=(1, 8))),
        (Topology("thread", 64), Layout(shape=(4, 8), strides=(1, 4))),
    ],
    ids=[
        "flat-32-lanes",
        "wrong-lane-order",
        "cta-not-thread",
        "wrong-lane-count",
        "inconsistent-mesh",
    ],
)
def test_a_scope_that_cannot_host_the_fragment_is_rejected(topology, layout) -> None:
    """Five distinct ways a scope fails the thread-participation match: a flat
    (32,) scope cannot host the 2-axis (4,8) fragment; the right shape in the
    wrong lane order is a different thread-value decomposition; a `cta` scope is
    not a warp scope however it is shaped; 64 lanes do not match a 32-lane atom;
    and a thread(64) topology carrying a 32-element layout is malformed."""
    with pytest.raises(VerifyError, match="required thread scope"):
        prim_func(target=CudaTarget("nvidia.h200_sxm"))(_alloc_frag_kernel(topology, layout))


def test_atom_A_outside_mesh_scope_is_rejected() -> None:
    """A fragment used with no enclosing mesh scope is rejected."""

    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
        frag = T.alloc_tensor(  # noqa: F841
            TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A,
                       storage=StorageKind.RMEM)
        )

    with pytest.raises(VerifyError, match="must be used inside a `with Mesh"):
        prim_func(target=CudaTarget("nvidia.h200_sxm"))(kernel)
