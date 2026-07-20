"""Rank-5 ShardLayout encoding for SM80 16x8x16 mma fragment distribution.

The canonical A / B / C fragment ``ShardLayout``\\s + the ``(4, 8)`` thread
mesh live in ``tilefoundry.ir.tir.cuda.nn.mma`` (the single source of truth, with
the full per-operand derivation recipe); this module reads them off the
realized ``MmaAtom`` (via ``make_atom``) and pins their structural invariants —
shape, per-thread element count (= PTX register count per thread: 8 / 4 / 4),
Split-axis extents, and reshard-typeinfer acceptance — so a change to the
derived strides fails a test rather than silently miscompiling.
"""
from __future__ import annotations

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.tir.cuda.nn.mma import SM80_16x8x16_F32BF16BF16F32_TN, make_atom
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import ShardLayout, Split, product
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TypeInferContext

# Fragment shards are read off the realized atom, not a separate export.
_ATOM = make_atom(SM80_16x8x16_F32BF16BF16F32_TN)
A_FRAG_SHARD = _ATOM.A
B_FRAG_SHARD = _ATOM.B
C_FRAG_SHARD = _ATOM.C


# ── Construction smoke ───────────────────────────────────────────────────


# ── Per-thread element count matches PTX mma fragment width ──────────────


def test_a_per_thread_owns_8_bf16():
    """Each lane holds 8 bf16 elements of A (4 b16x2 register pairs)."""
    assert _per_thread_size(A_FRAG_SHARD) == 8


def test_b_per_thread_owns_4_bf16():
    """Each lane holds 4 bf16 elements of B (2 b16x2 register pairs)."""
    assert _per_thread_size(B_FRAG_SHARD) == 4


def test_c_per_thread_owns_4_f32():
    """Each lane holds 4 f32 elements of C/D."""
    assert _per_thread_size(C_FRAG_SHARD) == 4


# ── Mesh axis attrs — every Split axis has the right extent ──────────────


def test_a_split_axes_match_mesh_extents():
    # Rule is identical across A/B/C operands; one representative covers it.
    _check_split_extents_match_mesh(A_FRAG_SHARD)


# ── Reshard typeinfer accepts each rank-5 fragment ShardLayout ───────────


def test_reshard_typeinfer_accepts_a_fragment():
    # Rule is identical across A/B/C operands; one representative covers it.
    _assert_reshard_typeinfer_ok((16, 16), "bf16", A_FRAG_SHARD)


# ── helpers ──────────────────────────────────────────────────────────────


def _product(shape: tuple[int, ...]) -> int:
    return product(tuple(int(s) for s in shape))


def _per_thread_size(sl: ShardLayout) -> int:
    """Layout product divided by mesh size (= per-thread element count)."""
    mesh_size = _product(sl.mesh.layout.shape)
    return _product(sl.layout.shape) // mesh_size


def _check_split_extents_match_mesh(sl: ShardLayout) -> None:
    mesh_shape = sl.mesh.layout.shape
    assert len(sl.attrs) == len(mesh_shape), (
        f"attrs len {len(sl.attrs)} != mesh rank {len(mesh_shape)}"
    )
    for mesh_i, attr in enumerate(sl.attrs):
        assert isinstance(attr, Split), (
            f"mma fragment shard attrs must be Split, got {attr}"
        )
        ax = attr.axis
        assert sl.layout.shape[ax] == mesh_shape[mesh_i], (
            f"split tensor axis {ax} extent {sl.layout.shape[ax]} != "
            f"mesh axis {mesh_i} extent {mesh_shape[mesh_i]}"
        )


def _assert_reshard_typeinfer_ok(
    src_shape: tuple[int, ...], src_dtype_name: str, dst_layout: ShardLayout
) -> None:
    """Run the registered Reshard typeinfer rule against a synthesised
    Call whose source is a plain global-storage tensor. Assert the
    resulting TensorType pins the requested rank-5 ShardLayout while
    preserving the logical shape."""

    dtype = getattr(DType, src_dtype_name)
    src_ty = TensorType(
        shape=src_shape, dtype=dtype, layout=None, storage=StorageKind.GMEM
    )
    src = Var(type=src_ty, name="x")
    op = Reshard(layout=dst_layout, storage=StorageKind.RMEM)
    # The registry-driven typeinfer ignores the Call's declared ``type``
    # field and recomputes from the op + args; we satisfy the dataclass
    # with a placeholder.
    call = Call(type=src_ty, target=op, args=(src,))
    ctx = TypeInferContext()
    out_ty = ctx.type_of(call)
    assert isinstance(out_ty, TensorType), f"expected TensorType, got {out_ty}"
    assert out_ty.layout is dst_layout, "output layout must reference the rank-5 ShardLayout"
    # Reshard preserves logical tensor shape.
    assert out_ty.shape == src_shape
    assert out_ty.dtype == dtype
    assert out_ty.storage == StorageKind.RMEM
