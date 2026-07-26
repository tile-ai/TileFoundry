"""``extract(HIR) -> TileGraph`` for a minimal gemm+rmsnorm HIR
``Function`` -- the polyhedral model plus the dependences derived from it,
independent of the existing HIR -> TIR -> CUDA path (``compile.py`` /
``passes`` / ``codegen`` / ``ir`` are untouched by this module and this
test). The isl schedule computed over these facts is covered by
``tests/schedule/test_kernel_schedule.py``.

Extraction is plain element granularity (no tiling), so matmul's M/N/K
(2, 2, 4) *are* the iteration domain directly -- chosen at this small size
so the expected ``deps`` stay a direct, hand-checkable transcription of the
M1 de-risk probe (an isl ``union_access_info``/``compute_flow`` script over
gemm+rmsnorm reads/writes).
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def test_extract_deps_include_k_carry_and_mm_to_rn():
    """One TileUnit per compute op (MM, RN), and ``deps`` auto-inferred
    from access relations alone (isl.union_access_info.compute_flow, per
    m1_deps_probe.py) reproduces exactly the two dependences that PoC
    hand-wrote: matmul's k-reduction carry and the MM -> RN RAW-on-h
    fusion edge -- nothing more, nothing less."""
    tg = extract(gemm_rmsnorm)
    assert isinstance(tg, TileGraph)

    names_by_op = {u.name: type(u.op.target).__name__ for u in tg.units}
    assert names_by_op == {"MM": "MatMul", "RN": "RMSNorm"}

    print("\n=== domain ===")
    print(tg.domain)
    print("=== reads ===")
    print(tg.reads)
    print("=== writes ===")
    print(tg.writes)
    print("=== deps (auto-inferred) ===")
    print(tg.deps)

    k_carry = isl.map("{ MM[i,j,k] -> MM[i,j,k+1] : 0<=i<2 and 0<=j<2 and 0<=k<3 }")
    mm_to_rn = isl.map("{ MM[i,j,3] -> RN[i] : 0<=i<2 and 0<=j<2 }")
    assert k_carry.is_subset(tg.deps)
    assert mm_to_rn.is_subset(tg.deps)

    expected_total = isl.union_map("{}").union(k_carry).union(mm_to_rn)
    assert tg.deps.is_equal(expected_total)
