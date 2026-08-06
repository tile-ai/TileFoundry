"""``candidate_atoms(op, target) -> list[AtomFact]`` -- the CUDA target's
own candidate enumeration: HIR ``MatMul`` op + target -> the MMA atom
candidates it could run on (a hard filter over shape/dtype/layout; no
CP-SAT ranking, which is the solver's own subject).

Builds a bf16 gemm HIR function -- bf16 being the sole dtype the one registered
SM80 atom accepts -- and checks the listed ``AtomFact`` against that atom's
real, known numbers, not just non-empty/non-zero placeholders.
"""
from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.ir.tir.cuda.nn.mma import SM80_16x8x16_F32BF16BF16F32_TN
from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom
from tilefoundry.ir.types import DType
from tilefoundry.schedule.facts import AtomFact
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.target.cuda.atoms import candidate_atoms


@func(target=CudaTarget("nvidia.h200_sxm"))
def bf16_gemm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)
    return h


@func(target=CudaTarget("nvidia.h200_sxm"))
def f32_gemm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)
    return h


@func(target=CudaTarget("nvidia.h200_sxm"))
def odd_shape_bf16_gemm(
    x: Tensor[(15, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
) -> Tensor[(15, 64), "bf16"]:
    h = matmul(x, w)
    return h


@func
def gemm_rmsnorm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def test_bf16_gemm_lists_the_sm80_atom_with_real_numbers():
    """The sole registered atom (SM80 16x8x16, bf16 x bf16 -> f32) is a
    candidate for a bf16 gemm whose M/N/K (64, 64, 128) all divide its
    (16, 8, 16) shape; every ``AtomFact`` field is checked against the
    atom's own known real numbers."""
    facts = candidate_atoms(bf16_gemm.entry_function().body, bf16_gemm.resolve_target())

    print("\n=== candidate AtomFacts (bf16 gemm, M=64 N=64 K=128) ===")
    for fact in facts:
        print(fact)

    assert len(facts) == 1
    fact = facts[0]
    assert isinstance(fact, AtomFact)
    assert fact.shape == (16, 8, 16)
    assert fact.dtype == (DType.bf16, DType.bf16, DType.f32)
    assert fact.duration > 0
    assert isinstance(fact.atom, MmaAtom)
    assert fact.atom.op is SM80_16x8x16_F32BF16BF16F32_TN

    # storage: per-thread fragment register bytes -- 8/4/4 elements per
    # thread for A/B/C, per mma.py's own fragment-derivation comments.
    assert fact.storage == {
        "a_reg_bytes": 16,  # 8 bf16 elements * 2 bytes
        "b_reg_bytes": 8,   # 4 bf16 elements * 2 bytes
        "c_reg_bytes": 16,  # 4 f32 elements * 4 bytes
        "reg_bytes": 40,
    }
    assert fact.resource == {"lane": 32}
    assert fact.is_async is False


def test_a_gemm_the_atom_cannot_run_lists_no_candidate():
    """Both halves of the hard filter, each an empty list rather than an error.

    The SM80 atom is bf16 x bf16 -> f32, so an all-f32 gemm's operand dtypes do
    not match it at all; and a gemm whose M is 15 does not divide the atom's M of
    16 even though its dtypes do match. An empty list is the answer a caller can
    act on -- an error here would make "this atom does not apply" indistinguishable
    from "this op cannot be asked about".
    """
    assert candidate_atoms(f32_gemm.entry_function().body, f32_gemm.resolve_target()) == []
    assert (
        candidate_atoms(
            odd_shape_bf16_gemm.entry_function().body,
            odd_shape_bf16_gemm.resolve_target(),
        )
        == []
    )


def test_non_matmul_op_raises():
    """V1 supports a MatMul Call only; any other op (here RMSNorm, from
    ``gemm_rmsnorm``'s body) raises a clear ``NotImplementedError`` rather
    than silently returning ``[]``."""
    with pytest.raises(NotImplementedError):
        candidate_atoms(gemm_rmsnorm.body)


def test_non_cuda_target_raises():
    """V1 supports ``CudaTarget`` only -- no per-atom device facts
    (sm_count / hbm_bandwidth / peak_for) exist for a CPU target."""
    with pytest.raises(NotImplementedError):
        candidate_atoms(bf16_gemm.entry_function().body, target=CpuTarget())
