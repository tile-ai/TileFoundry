"""``candidate_atoms(op, target) -> list[AtomFact]``.

``candidate_atoms(op, target) -> list[AtomFact]`` -- bridge one HIR
compute op to the atom catalogue of the AMX target, which spans two
execution units: the AMX coprocessor and the core's own NEON SIMD pipes.
It only *lists* candidates (a hard filter over shape, dtype, layout and
operand storage); it never picks one, that choice is the schedule layer's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.nn.matmul import MatMul, matmul_axes
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.target import Target
from tilefoundry.target.amx.spec import APPLE_AMX_ID
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.base import target_instance


@dataclass(frozen=True)
class StorageLevel:
    """Where an atom's operands sit while it executes.

    Where an atom's operands sit while it executes: per operand role, the
    bytes that role has to fit into. A level backed by a larger store only
    streams its operands through, so it budgets no role and holds anything.
    """

    name: str
    budget: tuple[tuple[str, int], ...] = ()

    def holds(self, operand_bytes: dict[str, int]) -> bool:
        """Whether every budgeted role fits -- vacuously so when none is."""
        return all(operand_bytes[role] <= limit for role, limit in self.budget)


_ISA = AmxTarget.hardware.resolve(APPLE_AMX_ID).value



AMX_REGISTERS = StorageLevel(
    name="amx_xyz_registers",
    budget=(
        ("a_bytes", _ISA.staging_bytes),
        ("b_bytes", _ISA.staging_bytes),
        ("c_bytes", _ISA.accumulator_bytes),
    ),
)




CORE_CACHE = StorageLevel(name="core_cache")


@dataclass(frozen=True)
class AmxOpSpec:
    """A named, fully-specified matrix instruction: which execution unit issues it.

    A named, fully-specified matrix instruction: which execution unit issues
    it, and which storage level has to hold the operands it is handed.
    """

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




AMX_FMA32_16x16x1_F32 = AmxOpSpec(
    name="AMX_FMA32_16x16x1_F32",
    unit="amx",
    level=AMX_REGISTERS,
    shape_mnk=(16, 16, 1),
    dtype_a=DType.f32,
    dtype_b=DType.f32,
    dtype_c=DType.f32,
)




NEON_FMLA_4x4x1_F32 = AmxOpSpec(
    name="NEON_FMLA_4x4x1_F32",
    unit="neon",
    level=CORE_CACHE,
    shape_mnk=(4, 4, 1),
    dtype_a=DType.f32,
    dtype_b=DType.f32,
    dtype_c=DType.f32,
)



_AMX_OP_CATALOG: tuple[AmxOpSpec, ...] = (AMX_FMA32_16x16x1_F32, NEON_FMLA_4x4x1_F32)


def _dense_bytes(shape: tuple[int, ...], dtype: DType) -> int:
    """Bytes for one dense ``shape``/``dtype`` operand."""
    return math.ceil(math.prod(shape) * dtype.bit_width / 8)


def _operand_bytes(shape_mnk: tuple[int, int, int], op: AmxOpSpec) -> dict[str, int]:
    """Bytes each of a ``shape_mnk`` matmul's operand roles has to hold at ``op``'s dtypes.

    Bytes each of a ``shape_mnk`` matmul's operand roles has to hold at
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
    """Estimate nominal compute and traffic time for one AMX atom.

    Compute uses MNK flops and measured unit throughput; memory uses operand
    bytes and unified bandwidth. Return the maximum and compute-only time in
    nanoseconds so callers that account for traffic do not charge it twice.
    This ranks candidates rather than predicting measured performance. Values
    remain sub-nanosecond instead of applying a one-nanosecond floor.
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
    """Layout hard filter: the X/Y operand packing is derived for dense, unsharded operands.

    Layout hard filter: the X/Y operand packing is derived for dense,
    unsharded operands. A ShardLayout-carrying operand may need a repack step to
    feed this atom, which is an agent-filled hole rather than a candidate here.
    """
    return lhs.layout is None and rhs.layout is None


def _static_positive(*dims: object) -> bool:
    return all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in dims)


def candidate_atoms(op: Call, target: Target | None = None) -> list[AtomFact]:
    """List AMX atoms eligible for an HIR ``MatMul`` Call.

    This is a hard filter: static M/N/K divisibility, input dtypes, layouts, and
    three operands fitting the atom storage level. The storage test separates a
    register-resident unit from cache streaming. ``[]`` is valid; unsupported
    operation kinds and Targets raise.
    """
    target = AmxTarget() if target is None else target
    target = target_instance(target)
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
    a_m, a_k, b_n, _b_k = matmul_axes(op.target)
    m, k = lhs_type.shape[a_m], lhs_type.shape[a_k]
    n = rhs_type.shape[b_n]
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
