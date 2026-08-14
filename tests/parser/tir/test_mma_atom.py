"""Pin the ``T.cuda.mma`` namespace and ``MmaAtom`` contract.

Atoms expose A/B/C fragment layouts and a required thread scope; atom bindings
are compile-time values, not ``LetStmt`` nodes. Register allocation validates
the enclosing mesh by level, lane count, and exact thread-value layout while
ignoring names and mesh identity ([tir §2.3](docs/spec/tir.md#23-tir-ops)). The
scopes that cannot host a fragment are rows in ``error_cases.py``.
"""

from __future__ import annotations

from tests.parser.error_cases import alloc_frag_kernel as _alloc_frag_kernel
from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.tir.stmts import LetStmt, MeshScope
from tilefoundry.ir.types.shard import Layout, ShardLayout, Topology
from tilefoundry.target import CudaTarget

_ATOM = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)


def _first_alloc_layout(prim_fn) -> ShardLayout:
    """Pull the ShardLayout off the single alloc'd fragment inside the kernel's mesh scope.

    Pull the ShardLayout off the single alloc'd fragment inside the kernel's
    mesh scope.
    """
    mesh_scope = next(s for s in prim_fn.body.body if isinstance(s, MeshScope))
    let = next(s for s in mesh_scope.body.body if isinstance(s, LetStmt))
    return let.var.type.layout


def test_a_valid_scope_takes_the_atom_contract_as_is() -> None:
    """A/B/C and ``required_scope`` are the atom's contracts, canonical across builds.

    A/B/C and ``required_scope`` are the atom's contracts, canonical across
    builds — the resolver returns the same fragment objects rather than rebinding
    them to a caller mesh. A valid (4,8)/(1,4) thread(32) scope passes the
    use-point check whatever its axes are named, and the alloc'd fragment carries
    the atom's own layout unchanged.
    """
    atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
    assert isinstance(atom, MmaAtom)
    assert (atom.A, atom.B, atom.C) == (_ATOM.A, _ATOM.B, _ATOM.C)
    assert atom.required_scope is _ATOM.required_scope

    kernel = prim_func(target=CudaTarget("nvidia.h200_sxm"))(
        _alloc_frag_kernel(
            Topology("thread", 32),
            Layout(shape=(4, 8), strides=(1, 4)),
            names=("warp", "lane"),
        )
    )
    assert _first_alloc_layout(kernel) is _ATOM.A


def test_infunc_op_and_atom_emit_no_letstmt() -> None:
    """`op = T.cuda.mma.<NAME>` and `atom = T.cuda.mma.atom(op=op)` are static bindings.

    `op = T.cuda.mma.<NAME>` and `atom = T.cuda.mma.atom(op=op)` are
    static bindings: neither lowers to a LetStmt, so a body with only these
    assignments is empty.
    """

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def kernel(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
        op = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN
        atom = T.cuda.mma.atom(op=op)  # noqa: F841

    assert all(not isinstance(s, LetStmt) for s in kernel.body.body)
    assert kernel.body.body == ()
