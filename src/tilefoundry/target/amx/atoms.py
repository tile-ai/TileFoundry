"""``candidate_atoms(op, target) -> list[AtomFact]`` -- bridge one HIR
compute op to the atom catalogue of the AMX target, which spans two
execution units: the AMX coprocessor and the core's own NEON SIMD pipes.
It only *lists* candidates (a hard filter over shape, dtype, layout and
operand storage); it never picks one, that choice is the schedule layer's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.target import Target, resolve_target
from tilefoundry.target.amx.spec import installed_architecture
from tilefoundry.target.amx.target import AmxTarget


@dataclass(frozen=True)
class StorageLevel:
    """Where an atom's operands sit while it executes: per operand role, the
    bytes that role has to fit into. A level backed by a larger store only
    streams its operands through, so it budgets no role and holds anything."""

    name: str
    budget: tuple[tuple[str, int], ...] = ()

    def holds(self, operand_bytes: dict[str, int]) -> bool:
        """Whether every budgeted role fits -- vacuously so when none is."""
        return all(operand_bytes[role] <= limit for role, limit in self.budget)


# The X/Y/Z register files are AMX ISA geometry, not a per-part figure, so the
# level they form is read once off the installed architecture.
_ISA = installed_architecture()

# An AMX operand is addressed as a register: A in X, B in Y, C in the Z
# accumulator file, and one instance never reaches outside them.
AMX_REGISTERS = StorageLevel(
    name="amx_xyz_registers",
    budget=(
        ("a_bytes", _ISA.staging_bytes),
        ("b_bytes", _ISA.staging_bytes),
        ("c_bytes", _ISA.accumulator_bytes),
    ),
)

# NEON loads and stores its operands through the core's caches, which unified
# memory backs: an operand too big to stay resident streams instead of being
# rejected, so this level budgets nothing.
CORE_CACHE = StorageLevel(name="core_cache")


@dataclass(frozen=True)
class AmxOpSpec:
    """A named, fully-specified matrix instruction: which execution unit issues
    it, and which storage level has to hold the operands it is handed."""

    name: str
    unit: str
    level: StorageLevel
    shape_mnk: tuple[int, int, int]
    dtype_a: DType
    dtype_b: DType
    dtype_c: DType


@dataclass(frozen=True)
class AmxAtom:
    """Realized atom -- op plus the bytes each of its own operands occupies."""

    op: AmxOpSpec
    a_bytes: int
    b_bytes: int
    c_bytes: int


# One FMA32 multiplies a 64-byte X operand by a 64-byte Y operand, so its shape
# is a 16x16 f32 rank-one outer product: K is the atom's own extent, 1.
AMX_FMA32_16x16x1_F32 = AmxOpSpec(
    name="AMX_FMA32_16x16x1_F32",
    unit="amx",
    level=AMX_REGISTERS,
    shape_mnk=(16, 16, 1),
    dtype_a=DType.f32,
    dtype_b=DType.f32,
    dtype_c=DType.f32,
)

# `FMLA vD.4s, vN.4s, vM.s[i]` accumulates a 4-lane f32 vector times one
# broadcast lane of another; four of them over four accumulator registers are
# the smallest whole f32 outer product NEON issues, so K is 1 as for FMA32.
NEON_FMLA_4x4x1_F32 = AmxOpSpec(
    name="NEON_FMLA_4x4x1_F32",
    unit="neon",
    level=CORE_CACHE,
    shape_mnk=(4, 4, 1),
    dtype_a=DType.f32,
    dtype_b=DType.f32,
    dtype_c=DType.f32,
)

# The catalogue this bridge searches; add an AmxOpSpec here to extend it. AMX
# FMA16 and FMA64 exist in the instruction set and are not modelled.
_AMX_OP_CATALOG: tuple[AmxOpSpec, ...] = (AMX_FMA32_16x16x1_F32, NEON_FMLA_4x4x1_F32)


def _dense_bytes(shape: tuple[int, ...], dtype: DType) -> int:
    """Bytes for one dense ``shape``/``dtype`` operand."""
    return math.ceil(math.prod(shape) * dtype.bit_width / 8)


def _operand_bytes(shape_mnk: tuple[int, int, int], op: AmxOpSpec) -> dict[str, int]:
    """Bytes each of a ``shape_mnk`` matmul's operand roles has to hold at
    ``op``'s dtypes.

    A and B are *staged* one reduction step at a time, so their roles hold one
    column and one row rather than the whole K extent. C accumulates, so its
    role holds the entire M by N block for as long as the reduction runs --
    which is what makes a wide matmul unable to sit in a register file that a
    narrow one fits exactly.
    """
    m, n, _ = shape_mnk
    return {
        "a_bytes": _dense_bytes((m,), op.dtype_a),
        "b_bytes": _dense_bytes((n,), op.dtype_b),
        "c_bytes": _dense_bytes((m, n), op.dtype_c),
    }


def make_atom(op: AmxOpSpec) -> AmxAtom:
    """Realize ``op``: the bytes its own three operand roles need."""
    return AmxAtom(op=op, **_operand_bytes(op.shape_mnk, op))


def _roofline_duration_ns(atom: AmxAtom, target: AmxTarget) -> tuple[float, float]:
    """Nominal roofline estimate (ns) for *one* atom instance, as
    ``(duration, compute_only)``.

    Compute is the atom's own MNK flops over the measured f32 throughput of
    the unit that issues it; memory is its three operands' bytes over
    unified-memory bandwidth. The compute-only half is returned separately for
    a consumer that accounts the surrounding traffic itself and would otherwise
    charge memory twice. A nominal estimate to rank against, not a claim of
    accuracy. Both halves are positive, so neither needs a floor -- one that
    rounded up to a nanosecond would swallow a sub-ns SIMD atom whole.
    """
    m, n, k = atom.op.shape_mnk
    flops = 2 * m * n * k
    moved_bytes = atom.a_bytes + atom.b_bytes + atom.c_bytes
    device = target.device
    compute_ns = flops * 1_000_000_000 / device.throughput_for(atom.op.unit, atom.op.dtype_a)
    memory_ns = (
        moved_bytes * 1_000_000_000 / device.unified_memory_bandwidth_bytes_per_second
    )
    return max(compute_ns, memory_ns), compute_ns


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

    Hard filter only -- no ranking: an atom is a candidate iff

    1. ``op``'s M/N/K are all static and evenly divisible by the atom's
       ``shape_mnk``;
    2. ``op``'s lhs/rhs dtypes match the atom's ``dtype_a``/``dtype_b``;
    3. the operands' layouts are compatible (``_operands_layout_ok``); and
    4. ``op``'s own three operands fit the atom's storage level -- which is
       what separates a register-resident unit from one streaming through
       cache, and so what an untiled whole-tensor statement fails.

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
        if not amx_op.level.holds(_operand_bytes((m, n, k), amx_op)):
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
                    "a_bytes": atom.a_bytes,
                    "b_bytes": atom.b_bytes,
                    "c_bytes": atom.c_bytes,
                    "operand_bytes": atom.a_bytes + atom.b_bytes + atom.c_bytes,
                },
                # Both units issue their atoms in order; neither has an async form.
                resource={amx_op.unit: 1},
                is_async=False,
                atom=atom,
            )
        )
    return facts


__all__ = [
    "AMX_FMA32_16x16x1_F32",
    "AMX_REGISTERS",
    "CORE_CACHE",
    "NEON_FMLA_4x4x1_F32",
    "AmxAtom",
    "AmxOpSpec",
    "StorageLevel",
    "candidate_atoms",
    "make_atom",
]
