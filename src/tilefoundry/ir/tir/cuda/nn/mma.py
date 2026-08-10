r"""Define CUDA MMA effects, instructions, and fixed fragment layouts.

Fragment constants encode the CuTe thread-value maps in row-major order;
``make_atom`` binds a named instruction to those layouts and its required mesh.
The descriptor records live in ``mma_atom.py``.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op, VerifyError
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, UnitType
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Topology,
)
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

from .mma_atom import MmaAtom, MmaOpSpec

_FP_ACC_WIDEN = {
    (DType.f16, DType.f32),
    (DType.bf16, DType.f32),
    (DType.f16, DType.f16),
    (DType.bf16, DType.bf16),
    (DType.f32, DType.f32),
}


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

    if len(lhs_ty.shape) == 2 and len(rhs_ty.shape) == 2 and len(acc_ty.shape) == 2:
        m, k_l = lhs_ty.shape[-2], lhs_ty.shape[-1]
        k_r, n = rhs_ty.shape[-2], rhs_ty.shape[-1]
        if k_l != k_r:
            ctx.error(call, f"Mma K-dim mismatch: {k_l} vs {k_r}")
        if acc_ty.shape[-2] != m or acc_ty.shape[-1] != n:
            ctx.error(
                call,
                f"Mma acc shape mismatch: expected (...,{m},{n}), got (...,{acc_ty.shape[-2]},{acc_ty.shape[-1]})",
            )
    if lhs_ty.dtype != rhs_ty.dtype:
        ctx.error(call, f"Mma lhs/rhs dtype mismatch: {lhs_ty.dtype} vs {rhs_ty.dtype}")
    if (lhs_ty.dtype, acc_ty.dtype) not in _FP_ACC_WIDEN:
        ctx.error(call, f"Mma unsupported dtype combo: input {lhs_ty.dtype} acc {acc_ty.dtype}")

    atom = call.target.atom
    if atom is not None:
        for role, ty, want in (
            ("acc", acc_ty, atom.C),
            ("lhs", lhs_ty, atom.A),
            ("rhs", rhs_ty, atom.B),
        ):
            if getattr(ty, "layout", None) != want:
                ctx.error(
                    call,
                    f"Mma {role} fragment layout does not match atom {_ATOM_ROLE[role]}",
                )
        from tilefoundry.ir.types.shard.scope_match import (  # noqa: PLC0415
            mesh_scope_matches_required_scope,
        )

        if not any(
            mesh_scope_matches_required_scope(s, atom.required_scope) for s in ctx.mesh_scope
        ):
            raise VerifyError(
                "T.mma: no enclosing mesh scope hosts the atom's required thread "
                f"scope (topology {atom.required_scope.topologies[0].name!r}, "
                f"{atom.required_scope.topologies[0].size} lanes)"
            )


_SM80_THREAD_MESH = Mesh(
    topologies=(Topology("thread", 32),),
    layout=Layout(shape=(4, 8), strides=(1, 4)),
)


A_FRAG_LAYOUT = Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128))
_A_FRAG_SHARD = ShardLayout(
    layout=A_FRAG_LAYOUT,
    attrs=(Split(1), Split(3)),
    mesh=_SM80_THREAD_MESH,
)


B_FRAG_LAYOUT = Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64))
_B_FRAG_SHARD = ShardLayout(
    layout=B_FRAG_LAYOUT,
    attrs=(Split(2), Split(0)),
    mesh=_SM80_THREAD_MESH,
)


C_FRAG_LAYOUT = Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64))
_C_FRAG_SHARD = ShardLayout(
    layout=C_FRAG_LAYOUT,
    attrs=(Split(1), Split(2)),
    mesh=_SM80_THREAD_MESH,
)


SM80_16x8x16_F32BF16BF16F32_TN = MmaOpSpec(
    name="SM80_16x8x16_F32BF16BF16F32_TN",
    shape_mnk=(16, 8, 16),
    dtype_a=DType.bf16,
    dtype_b=DType.bf16,
    dtype_c=DType.f32,
    operand_layout="TN",
)


_ATOM_TABLE: dict[MmaOpSpec, tuple[ShardLayout, ShardLayout, ShardLayout, Mesh]] = {
    SM80_16x8x16_F32BF16BF16F32_TN: (
        _A_FRAG_SHARD,
        _B_FRAG_SHARD,
        _C_FRAG_SHARD,
        _SM80_THREAD_MESH,
    ),
}


def make_atom(op: MmaOpSpec) -> MmaAtom:
    """Build the :class:`MmaAtom` for ``op`` (CuTe ``make_tiled_mma`` analog).

    Raises ``KeyError`` (with a clear message) for an instruction that has no
    registered fragment layouts yet.
    """
    if not isinstance(op, MmaOpSpec):
        raise TypeError(f"mma atom(op=...) expects an MmaOpSpec, got {type(op).__name__}")
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
