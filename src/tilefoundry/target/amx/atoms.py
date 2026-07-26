"""``candidate_atoms(op, target) -> list[AtomFact]`` -- bridge one HIR
compute op to the AMX atom catalogue it could run on. It only *lists*
candidates (a hard filter over shape/dtype/layout); it never picks one,
that ranking is the schedule layer's CP-SAT job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tilefoundry.analysis.atom_facts import AtomFact
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.target import Target, resolve_target
from tilefoundry.target.amx.target import AmxTarget


@dataclass(frozen=True)
class AmxOpSpec:
    """A named, fully-specified AMX matrix instruction."""

    name: str
    shape_mnk: tuple[int, int, int]
    dtype_a: DType
    dtype_b: DType
    dtype_c: DType


@dataclass(frozen=True)
class AmxAtom:
    """Realized AMX atom -- op plus the register bytes each operand holds."""

    op: AmxOpSpec
    x_bytes: int
    y_bytes: int
    z_bytes: int


# One FMA32 multiplies a 64-byte X operand by a 64-byte Y operand, so its shape
# is a 16x16 f32 rank-one outer product: K is the atom's own extent, 1.
AMX_FMA32_16x16x1_F32 = AmxOpSpec(
    name="AMX_FMA32_16x16x1_F32",
    shape_mnk=(16, 16, 1),
    dtype_a=DType.f32,
    dtype_b=DType.f32,
    dtype_c=DType.f32,
)

# The AMX atom catalogue this bridge searches; add an AmxOpSpec here to extend
# it. FMA16 and FMA64 exist in the instruction set and are not modelled.
_AMX_OP_CATALOG: tuple[AmxOpSpec, ...] = (AMX_FMA32_16x16x1_F32,)


def _dense_bytes(shape: tuple[int, ...], dtype: DType) -> int:
    """Bytes for one dense ``shape``/``dtype`` operand."""
    return math.ceil(math.prod(shape) * dtype.bit_width / 8)


def make_atom(op: AmxOpSpec) -> AmxAtom:
    """Realize ``op``: the X/Y operand and Z accumulator bytes its shape needs."""
    m, n, k = op.shape_mnk
    return AmxAtom(
        op=op,
        x_bytes=_dense_bytes((m, k), op.dtype_a),
        y_bytes=_dense_bytes((k, n), op.dtype_b),
        z_bytes=_dense_bytes((m, n), op.dtype_c),
    )


def _roofline_duration_ns(atom: AmxAtom, target: AmxTarget) -> tuple[float, float]:
    """Nominal roofline estimate (ns) for *one* atom instance, as
    ``(duration, compute_only)``.

    Compute is the atom's own MNK flops over one AMX unit's measured f32
    throughput; memory is its X+Y+Z bytes over unified-memory bandwidth. The
    compute-only half is returned separately for a consumer that accounts the
    surrounding traffic itself and would otherwise charge memory twice. This is
    a nominal estimate to rank against, not a claim of accuracy.
    """
    m, n, k = atom.op.shape_mnk
    flops = 2 * m * n * k
    moved_bytes = atom.x_bytes + atom.y_bytes + atom.z_bytes
    device = target.device
    compute_ns = flops * 1_000_000_000 / device.throughput_for(atom.op.dtype_a)
    memory_ns = (
        moved_bytes * 1_000_000_000 / device.unified_memory_bandwidth_bytes_per_second
    )
    return max(compute_ns, memory_ns, 1.0), compute_ns


def _operands_layout_ok(lhs: TensorType, rhs: TensorType) -> bool:
    """Layout hard filter: the X/Y operand packing is derived for dense,
    unsharded operands. A ShardLayout-carrying operand may need a repack step to
    feed this atom, which is an agent-filled hole rather than a candidate here.
    """
    return lhs.layout is None and rhs.layout is None


def _static_positive(*dims: object) -> bool:
    return all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in dims)


def candidate_atoms(op: Call, target: Target | str | None = None) -> list[AtomFact]:
    """List every AMX atom that could execute ``op`` (an HIR ``MatMul``
    ``Call``) on ``target`` (a default :class:`AmxTarget` when omitted; a
    backend name string is resolved via ``resolve_target``).

    Hard filter only -- no ranking, no CP-SAT: an atom is a candidate iff

    1. ``op``'s M/N/K are all static and evenly divisible by the atom's
       ``shape_mnk``;
    2. ``op``'s lhs/rhs dtypes match the atom's ``dtype_a``/``dtype_b``; and
    3. the operands' layouts are compatible (``_operands_layout_ok``).

    Returns ``[]`` when no registered atom clears the filter (a legitimate "no
    candidates" outcome, not an error). Raises ``NotImplementedError`` for an
    unsupported op kind or target.
    """
    target = AmxTarget() if target is None else resolve_target(target)
    if not isinstance(op, Call) or not isinstance(op.target, MatMul):
        got = type(op).__name__
        if isinstance(op, Call):
            got += f" (target={type(op.target).__name__})"
        raise NotImplementedError(
            f"candidate_atoms: only a MatMul Call is supported, got {got}"
        )
    if not isinstance(target, AmxTarget):
        raise NotImplementedError(
            "candidate_atoms: only AmxTarget is supported, got "
            f"{type(target).__name__}"
        )

    lhs_type, rhs_type = op.args[0].type, op.args[1].type
    m, k = lhs_type.shape[-2], lhs_type.shape[-1]
    n = rhs_type.shape[-1]
    if not _static_positive(m, n, k) or not _operands_layout_ok(lhs_type, rhs_type):
        return []

    facts: list[AtomFact] = []
    for amx_op in _AMX_OP_CATALOG:
        atom_m, atom_n, atom_k = amx_op.shape_mnk
        if m % atom_m != 0 or n % atom_n != 0 or k % atom_k != 0:
            continue
        if lhs_type.dtype != amx_op.dtype_a or rhs_type.dtype != amx_op.dtype_b:
            continue
        atom = make_atom(amx_op)
        duration, compute_duration = _roofline_duration_ns(atom, target)
        facts.append(
            AtomFact(
                shape=amx_op.shape_mnk,
                dtype=(amx_op.dtype_a, amx_op.dtype_b, amx_op.dtype_c),
                duration=duration,
                compute_duration=compute_duration,
                storage={
                    "x_bytes": atom.x_bytes,
                    "y_bytes": atom.y_bytes,
                    "z_bytes": atom.z_bytes,
                    "register_bytes": atom.x_bytes + atom.y_bytes + atom.z_bytes,
                },
                # AMX instructions issue in order on the coprocessor pipe.
                resource={"amx": 1},
                is_async=False,
                atom=atom,
            )
        )
    return facts


__all__ = ["AMX_FMA32_16x16x1_F32", "AmxAtom", "AmxOpSpec", "candidate_atoms", "make_atom"]
