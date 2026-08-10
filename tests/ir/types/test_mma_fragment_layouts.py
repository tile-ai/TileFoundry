r"""Pin structural invariants of SM80 MMA fragment layouts.

The realized atom supplies A/B/C layouts and its thread mesh. Tests cover shape,
per-thread register counts, Split extents, and reshard acceptance; parser tests
cover required thread scope.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.tir.cuda.nn.mma import SM80_16x8x16_F32BF16BF16F32_TN, make_atom
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import ShardLayout, Split, product
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TypeInferContext

_ATOM = make_atom(SM80_16x8x16_F32BF16BF16F32_TN)
A_FRAG_SHARD = _ATOM.A
B_FRAG_SHARD = _ATOM.B
C_FRAG_SHARD = _ATOM.C


def test_per_thread_element_counts_and_split_extents() -> None:
    """Each lane holds 8 bf16 of A (4 b16x2 register pairs), 4 bf16 of B, and 4 f32 of C/D.

    Each lane holds 8 bf16 of A (4 b16x2 register pairs), 4 bf16 of B, and
    4 f32 of C/D. The Split-axis rule is identical across operands, so A stands
    for all three: every Split axis' tensor extent equals its mesh extent.
    """
    assert _per_thread_size(A_FRAG_SHARD) == 8
    assert _per_thread_size(B_FRAG_SHARD) == 4
    assert _per_thread_size(C_FRAG_SHARD) == 4
    _check_split_extents_match_mesh(A_FRAG_SHARD)


def test_reshard_typeinfer_accepts_a_fragment() -> None:

    _assert_reshard_typeinfer_ok((16, 16), "bf16", A_FRAG_SHARD)


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
        assert isinstance(attr, Split), f"mma fragment shard attrs must be Split, got {attr}"
        ax = attr.axis
        assert sl.layout.shape[ax] == mesh_shape[mesh_i], (
            f"split tensor axis {ax} extent {sl.layout.shape[ax]} != "
            f"mesh axis {mesh_i} extent {mesh_shape[mesh_i]}"
        )


def _assert_reshard_typeinfer_ok(
    src_shape: tuple[int, ...], src_dtype_name: str, dst_layout: ShardLayout
) -> None:
    """Assert reshard typeinfer ok.

    Run the registered Reshard typeinfer rule against a synthesised
    Call whose source is a plain global-storage tensor. Assert the
    resulting TensorType pins the requested rank-5 ShardLayout while
    preserving the logical shape.
    """
    dtype = getattr(DType, src_dtype_name)
    src_ty = TensorType(shape=src_shape, dtype=dtype, layout=None, storage=StorageKind.GMEM)
    src = Var(type=src_ty, name="x")
    op = Reshard(layout=dst_layout, storage=StorageKind.RMEM)

    call = Call(type=src_ty, target=op, args=(src,))
    ctx = TypeInferContext()
    out_ty = ctx.type_of(call)
    assert isinstance(out_ty, TensorType), f"expected TensorType, got {out_ty}"
    assert out_ty.layout is dst_layout, "output layout must reference the rank-5 ShardLayout"

    assert out_ty.shape == src_shape
    assert out_ty.dtype == dtype
    assert out_ty.storage == StorageKind.RMEM
