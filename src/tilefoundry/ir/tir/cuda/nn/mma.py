"""CUDA (NVIDIA) MMA op, instructions, and their fragment layouts.

The CUDA MMA surface: the effect-form ``Mma`` op (``acc += lhs @ rhs``,
``T.mma(acc, a, b, atom=...)``), the concrete named instructions, their A / B /
C fragment ``ShardLayout``\\s, and the ``make_atom`` resolver that binds an
:class:`MmaOpSpec` to its realized :class:`MmaAtom`. The op / atom descriptor
classes live next door in ``mma_atom.py``.
"""
from __future__ import annotations

from tilefoundry.ir.core import Op, VerifyError
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer, register_verify_stmt
from tilefoundry.ir.types import DType, UnitType
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Topology,
)

from .mma_atom import MmaAtom, MmaOpSpec

_FP_ACC_WIDEN = {
    # acc dtype is allowed to widen from input dtype per V1 rule.
    (DType.f16, DType.f32),
    (DType.bf16, DType.f32),
    (DType.f16, DType.f16),
    (DType.bf16, DType.bf16),
    (DType.f32, DType.f32),
}

# operand role → the atom fragment-layout attribute it must match.
_ATOM_ROLE = {"acc": "C", "lhs": "A", "rhs": "B"}


@register_op(category="nn")
class Mma(Op):
    """Matrix-multiply-accumulate: ``acc += lhs @ rhs``."""
    acc = ParamDef(kind="input", pattern=Tensor)
    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
    atom = ParamDef(kind="attribute", annotation=MmaAtom, default=None, optional=True)


@register_typeinfer(Mma)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(Mma)
def _(call: "Call", ctx: "VerifyContext") -> None:
    acc_ty = ctx.type_of(call.args[0])
    lhs_ty = ctx.type_of(call.args[1])
    rhs_ty = ctx.type_of(call.args[2])
    # Per-shard fragment storage may carry rank-N (>2) per-thread shapes
    # — the (M, K) / (K, N) / (M, N) check only applies when operands
    # are still in their logical 2D form (cf. reshard preserves logical
    # shape only when the destination is *not* a fragment ShardLayout).
    if len(lhs_ty.shape) == 2 and len(rhs_ty.shape) == 2 and len(acc_ty.shape) == 2:
        m, k_l = lhs_ty.shape[-2], lhs_ty.shape[-1]
        k_r, n = rhs_ty.shape[-2], rhs_ty.shape[-1]
        if k_l != k_r:
            ctx.error(call, f"Mma K-dim mismatch: {k_l} vs {k_r}")
        if acc_ty.shape[-2] != m or acc_ty.shape[-1] != n:
            ctx.error(call, f"Mma acc shape mismatch: expected (...,{m},{n}), got (...,{acc_ty.shape[-2]},{acc_ty.shape[-1]})")
    if lhs_ty.dtype != rhs_ty.dtype:
        ctx.error(call, f"Mma lhs/rhs dtype mismatch: {lhs_ty.dtype} vs {rhs_ty.dtype}")
    if (lhs_ty.dtype, acc_ty.dtype) not in _FP_ACC_WIDEN:
        ctx.error(call, f"Mma unsupported dtype combo: input {lhs_ty.dtype} acc {acc_ty.dtype}")
    # Atom path: check each operand's fragment layout against the atom
    # (acc→C, lhs→A, rhs→B), and that an enclosing mesh scope (``ctx.mesh_scope``,
    # set by the verify walk) hosts the atom's required thread scope.
    atom = call.target.atom
    if atom is not None:
        for role, ty, want in (
            ("acc", acc_ty, atom.C), ("lhs", lhs_ty, atom.A), ("rhs", rhs_ty, atom.B),
        ):
            if getattr(ty, "layout", None) != want:
                ctx.error(
                    call,
                    f"Mma {role} fragment layout does not match atom "
                    f"{_ATOM_ROLE[role]}",
                )
        from tilefoundry.ir.types.shard.scope_match import (  # noqa: PLC0415
            mesh_scope_matches_required_scope,
        )
        if not any(
            mesh_scope_matches_required_scope(s, atom.required_scope)
            for s in ctx.mesh_scope
        ):
            raise VerifyError(
                "T.mma: no enclosing mesh scope hosts the atom's required thread "
                f"scope (topology {atom.required_scope.topology.name!r}, "
                f"{atom.required_scope.topology.size} lanes)"
            )


# ── Thread mesh: 32 lanes laid out as (x=4, y=8), strides (1, 4) ─────────
#
# The SM80 atom's ``Shape<_4,_8>`` thread layout. Strides ``(1, 4)`` make
# axis 0 ('x', size 4) the fastest-varying lane coord. Wrong strides silently
# break numerics, so this is pinned, not derived at use sites.
_SM80_THREAD_MESH = Mesh(
    topology=Topology("thread", 32),
    layout=Layout(shape=(4, 8), strides=(1, 4)),
)

# Fragment derivation recipe (per operand):
#   1. Extract the MMA_Atom thread-value layout from cutlass's
#      ``atom/mma_traits_sm80.hpp``.
#   2. Reorder axes by stride (ascending) under tilefoundry's row-major
#      interpretation of the (M,K)/(K,N)/(M,N) tensor.
#   3. Locate the mesh-axis stride positions; attach Split to the corresponding
#      tensor axes. The rest are per-thread *value* axes (no Split).
# per_thread element count = layout-product / 32 must match the PTX register
# count per thread (8 / 4 / 4 for A / B / C).

# ── A-fragment: (M=16, K=16) bf16, row-major ─────────────────────────────
# cutlass ALayout (col-major source):
#   Layout<Shape<Shape<_4,_8>, Shape<_2,_2,_2>>,
#          Stride<Stride<_32,_1>, Stride<_16,_8,_128>>>
# Per-thread map (lane t=(tx∈[0,4), ty∈[0,8)), values v0/v1/v2):
#   M = ty + 8*v1 ; K = 2*tx + v0 + 8*v2
# Row-major (idx = M*16 + K): ty→16, tx→2, v0→1, v1→128, v2→8.
# Reordered ascending by stride → (v0, tx, v2, ty, v1).
A_FRAG_LAYOUT = Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128))
_A_FRAG_SHARD = ShardLayout(
    layout=A_FRAG_LAYOUT,
    attrs=(Split(1), Split(3)),  # x → axis 1 (size 4); y → axis 3 (size 8)
    mesh=_SM80_THREAD_MESH,
)

# ── B-fragment: (K=16, N=8) bf16, row-major ──────────────────────────────
# cutlass BLayout (col-major source):
#   Layout<Shape<Shape<_4,_8>, Shape<_2,_2>>, Stride<Stride<_16,_1>, Stride<_8,_64>>>
# Per-thread map (lane t=(tx,ty), values v0/v1):
#   N = ty ; K = 2*tx + v0 + 8*v1
# Row-major (idx = K*8 + N): ty→1, tx→16, v0→8, v1→64.
# Reordered ascending → (ty, v0, tx, v1).
B_FRAG_LAYOUT = Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64))
_B_FRAG_SHARD = ShardLayout(
    layout=B_FRAG_LAYOUT,
    attrs=(Split(2), Split(0)),  # x → axis 2 (size 4); y → axis 0 (size 8)
    mesh=_SM80_THREAD_MESH,
)

# ── C/D-fragment: (M=16, N=8) f32, row-major ─────────────────────────────
# cutlass CLayout = SM80_16x8_Row (col-major source):
#   Layout<Shape<Shape<_4,_8>, Shape<_2,_2>>, Stride<Stride<_32,_1>, Stride<_16,_8>>>
# Per-thread map (lane t=(tx,ty), values v0/v1):
#   M = ty + 8*v1 ; N = 2*tx + v0
# Row-major (idx = M*8 + N): ty→8, tx→2, v0→1, v1→64.
# Reordered ascending → (v0, tx, ty, v1).
C_FRAG_LAYOUT = Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64))
_C_FRAG_SHARD = ShardLayout(
    layout=C_FRAG_LAYOUT,
    attrs=(Split(1), Split(2)),  # x → axis 1 (size 4); y → axis 2 (size 8)
    mesh=_SM80_THREAD_MESH,
)


# The first supported instruction: SM80 mma.sync m16n8k16, bf16 inputs, f32
# accumulate, A row-major / B col-major (TN).
SM80_16x8x16_F32BF16BF16F32_TN = MmaOpSpec(
    name="SM80_16x8x16_F32BF16BF16F32_TN",
    shape_mnk=(16, 8, 16),
    dtype_a=DType.bf16,
    dtype_b=DType.bf16,
    dtype_c=DType.f32,
    operand_layout="TN",
)

# op → (A, B, C fragment ShardLayouts, required_scope). One entry today; new
# instructions add a row here (and a codegen handler) — no change to the
# atom / T.mma API.
_ATOM_TABLE: dict[
    MmaOpSpec, tuple[ShardLayout, ShardLayout, ShardLayout, Mesh]
] = {
    SM80_16x8x16_F32BF16BF16F32_TN: (
        _A_FRAG_SHARD, _B_FRAG_SHARD, _C_FRAG_SHARD, _SM80_THREAD_MESH,
    ),
}


def make_atom(op: MmaOpSpec) -> MmaAtom:
    """Build the :class:`MmaAtom` for ``op`` (cutlass ``make_tiled_mma`` analog).

    Raises ``KeyError`` (with a clear message) for an instruction that has no
    registered fragment layouts yet.
    """
    if not isinstance(op, MmaOpSpec):
        raise TypeError(
            f"mma atom(op=...) expects an MmaOpSpec, got {type(op).__name__}"
        )
    entry = _ATOM_TABLE.get(op)
    if entry is None:
        raise KeyError(
            f"no fragment layouts registered for MMA op {op.name!r}; "
            f"add an entry to ir.tir.cuda.nn.mma._ATOM_TABLE"
        )
    a, b, c, scope = entry
    return MmaAtom(op=op, A=a, B=b, C=c, required_scope=scope)


__all__ = [
    "Mma",
    "make_atom",
    "SM80_16x8x16_F32BF16BF16F32_TN",
]
