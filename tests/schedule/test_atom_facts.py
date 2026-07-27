"""``candidate_atoms(op, target) -> list[AtomFact]`` -- the CUDA target's
own candidate enumeration: HIR ``MatMul`` op + target -> the MMA atom
candidates it could run on (a hard filter over shape/dtype/layout; no
CP-SAT ranking, that is ``test_solve.py``'s subject).

Builds a bf16 gemm HIR function (mirrors ``test_poly_model.py``'s
construction, dtype swapped to bf16 -- the sole dtype the one registered
SM80 atom accepts) and checks the listed ``AtomFact`` against that atom's
real, known numbers -- not just non-empty/non-zero placeholders.
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
from tilefoundry.target import default_target
from tilefoundry.target.cuda.atoms import candidate_atoms


@func(target="cuda")
def bf16_gemm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)
    return h


@func(target="cuda")
def f32_gemm(
    x: Tensor[(64, 128), "f32"],
    w: Tensor[(128, 64), "f32"],
) -> Tensor[(64, 64), "f32"]:
    h = matmul(x, w)
    return h


@func(target="cuda")
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


def test_target_none_defaults_to_default_target():
    """``target=None`` resolves via ``default_target()`` -- the same
    result as passing an equivalent target explicitly."""
    explicit = candidate_atoms(bf16_gemm.entry_function().body, default_target())
    implicit = candidate_atoms(bf16_gemm.entry_function().body)
    assert implicit == explicit


def test_target_accepts_a_backend_name_string():
    """A ``target=`` string resolves via ``resolve_target``, matching
    ``@func``'s own ``target=`` surface (``func(target="cuda")``)."""
    by_string = candidate_atoms(bf16_gemm.entry_function().body, "cuda")
    by_object = candidate_atoms(bf16_gemm.entry_function().body, default_target())
    assert by_string == by_object


def test_f32_gemm_has_no_candidates_dtype_mismatch():
    """The SM80 atom is bf16 x bf16 -> f32; an all-f32 gemm's lhs/rhs
    dtype does not match (dtype_a=dtype_b=bf16), so the hard filter
    excludes it -- an empty list, not an error."""
    facts = candidate_atoms(f32_gemm.entry_function().body, f32_gemm.resolve_target())
    assert facts == []


def test_odd_shape_bf16_gemm_has_no_candidates_indivisible_mnk():
    """M=15 does not divide the atom's M=16 -- hard-filtered out even
    though dtype matches."""
    facts = candidate_atoms(odd_shape_bf16_gemm.entry_function().body, odd_shape_bf16_gemm.resolve_target())
    assert facts == []


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
        candidate_atoms(bf16_gemm.entry_function().body, target="cpu")
