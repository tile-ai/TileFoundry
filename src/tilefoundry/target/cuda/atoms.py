"""``candidate_atoms(op, target) -> list[AtomFact]``.

``candidate_atoms(op, target) -> list[AtomFact]`` -- bridge one HIR
compute op to the TIR MMA atom catalogue it could run on. It only *lists*
candidates (a hard filter over shape/dtype/layout); it never picks one,
that ranking is the schedule layer's CP-SAT job. See each helper's
docstring below for exactly where its numbers come from.
"""
from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.nn.matmul import MatMul, matmul_axes
from tilefoundry.ir.tir.cuda.nn.mma import SM80_16x8x16_F32BF16BF16F32_TN, make_atom
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaOpSpec
from tilefoundry.ir.types import DType, TensorType, tensor_bytes
from tilefoundry.ir.types.shard import ShardLayout
from tilefoundry.ir.types.shard.shard_layout import shard_layout_local_shape
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.target import Target, default_target
from tilefoundry.target.base import target_instance
from tilefoundry.target.cuda.target import CudaTarget

_MMA_OP_CATALOG: tuple[MmaOpSpec, ...] = (SM80_16x8x16_F32BF16BF16F32_TN,)


def _is_async_op(op: MmaOpSpec) -> bool:
    """wgmma-family instructions issue asynchronously; the sole registered op today is synchronous.

    wgmma-family instructions (SM90+) issue asynchronously; the sole
    registered op today (SM80 ``mma.sync``) is synchronous. No wgmma
    ``MmaOpSpec`` is registered yet in ``ir.tir.cuda.nn.mma._ATOM_TABLE``, so
    this substring check is forward-compatible naming, not a real dispatch
    -- it always returns ``False`` today.
    """
    return "wgmma" in op.name.lower()


def _dense_bytes(shape: tuple[int, ...], dtype: DType) -> int:
    """Bytes for one dense (unsharded) ``shape``/``dtype`` tile.

    Bytes for one dense (unsharded) ``shape``/``dtype`` tile -- reuses
    ``target.cuda.cost.tensor_bytes`` (the exact helper the ``MatMul`` cost
    evaluator uses for its own HBM-traffic accounting), so op-level and
    atom-level byte counts share one formula. ``storage``/``layout`` are
    irrelevant to a byte count, so a placeholder ``GMEM`` is used.
    """
    type = TensorType(shape=shape, dtype=dtype, layout=None, storage=StorageKind.GMEM)
    return tensor_bytes(type)


def _fragment_reg_bytes(fragment: ShardLayout, dtype: DType) -> int:
    """Per-thread register bytes for one A/B/C fragment.

    Per-thread register bytes for one A/B/C fragment: divide the
    fragment's global layout shape down to its per-thread local shape
    (``shard_layout_local_shape`` -- quotients out the Split-bound lane
    axes; see ``mma.py``'s fragment derivation comments for the 8/4/4
    elements-per-thread this yields for A/B/C), then reuse ``tensor_bytes``
    for the numel*bitwidth->bytes conversion (the same formula as
    ``_dense_bytes``, at per-thread instead of whole-tile granularity).
    """
    local_shape = shard_layout_local_shape(fragment)
    type = TensorType(shape=local_shape, dtype=dtype, layout=None, storage=StorageKind.RMEM)
    return tensor_bytes(type)


def _roofline_duration_ns(op: MmaOpSpec, target: CudaTarget) -> tuple[float, float]:
    """Estimate nominal compute and traffic time for one CUDA MMA atom.

    Compute uses the atom's MNK flops, device peak throughput, and SM count;
    memory uses dense A+B+C bytes and HBM bandwidth. Return the maximum with a
    one-nanosecond floor and the compute-only time so callers that account for
    traffic do not charge it twice. This ranks candidates rather than
    predicting measured performance.
    """
    m, n, k = op.shape_mnk
    flops = 2 * m * n * k
    moved_bytes = (
        _dense_bytes((m, k), op.dtype_a)
        + _dense_bytes((k, n), op.dtype_b)
        + _dense_bytes((m, n), op.dtype_c)
    )
    device = target.device
    compute_ns = flops * 1_000_000_000 * device.sm_count / device.peak_for(op.dtype_a)
    memory_ns = moved_bytes * 1_000_000_000 / device.hbm_bandwidth_bytes_per_second
    return max(compute_ns, memory_ns, 1.0), compute_ns


def _operands_layout_ok(lhs: TensorType, rhs: TensorType) -> bool:
    """V1 layout hard filter.

    V1 layout hard filter: the registered fragment layouts
    (``ir.tir.cuda.nn.mma``) are derived for dense, unsharded row-major
    operands -- the sole ``operand_layout="TN"`` convention today. A
    ShardLayout-carrying operand may need a pack/repack step to feed this
    atom; V1 defers that to an agent-filled hole (out of scope for this
    bridge) and simply excludes such operands from candidacy here, rather
    than erroring.
    """
    return lhs.layout is None and rhs.layout is None


def _static_positive(*dims: object) -> bool:
    return all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in dims)


def candidate_atoms(op: Call, target: Target | None = None) -> list[AtomFact]:
    """List CUDA MMA atoms eligible for an HIR ``MatMul`` Call.

    This is a hard filter, not ranking: static M/N/K must divide ``shape_mnk``;
    input dtypes and operand layouts must agree. ``dtype_c`` is the atom's
    internal accumulator, so narrowing output is an epilogue concern. ``[]`` is
    a valid outcome; unsupported operation kinds and Targets raise.
    """
    target = default_target() if target is None else target
    target = target_instance(target)
    if not isinstance(op, Call) or not isinstance(op.target, MatMul):
        got = type(op).__name__
        if isinstance(op, Call):
            got += f" (target={type(op.target).__name__})"
        raise NotImplementedError(
            f"candidate_atoms: only a MatMul Call is supported, got {got}"
        )
    if not isinstance(target, CudaTarget):
        raise NotImplementedError(
            "candidate_atoms: only CudaTarget is supported, got "
            f"{type(target).__name__}"
        )

    lhs_type, rhs_type = op.args[0].type, op.args[1].type
    a_m, a_k, b_n, _b_k = matmul_axes(op.target)
    m, k = lhs_type.shape[a_m], lhs_type.shape[a_k]
    n = rhs_type.shape[b_n]
    if not _static_positive(m, n, k) or not _operands_layout_ok(lhs_type, rhs_type):
        return []

    facts: list[AtomFact] = []
    for mma_op in _MMA_OP_CATALOG:
        atom_m, atom_n, atom_k = mma_op.shape_mnk
        if m % atom_m != 0 or n % atom_n != 0 or k % atom_k != 0:
            continue
        if lhs_type.dtype != mma_op.dtype_a or rhs_type.dtype != mma_op.dtype_b:
            continue
        atom = make_atom(mma_op)
        a_bytes = _fragment_reg_bytes(atom.A, mma_op.dtype_a)
        b_bytes = _fragment_reg_bytes(atom.B, mma_op.dtype_b)
        c_bytes = _fragment_reg_bytes(atom.C, mma_op.dtype_c)
        duration, compute_duration = _roofline_duration_ns(mma_op, target)
        facts.append(
            AtomFact(
                shape=mma_op.shape_mnk,
                dtype=(mma_op.dtype_a, mma_op.dtype_b, mma_op.dtype_c),
                duration=duration,
                compute_duration=compute_duration,
                storage={
                    "a_reg_bytes": a_bytes,
                    "b_reg_bytes": b_bytes,
                    "c_reg_bytes": c_bytes,
                    "reg_bytes": a_bytes + b_bytes + c_bytes,
                },
                resource={"lane": atom.required_scope.topologies[0].size},
                is_async=_is_async_op(mma_op),
                atom=atom,
            )
        )
    return facts


__all__ = ["candidate_atoms"]
