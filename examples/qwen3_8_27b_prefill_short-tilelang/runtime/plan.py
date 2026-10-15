"""Which kernel configuration serves which prompt length.

This table is where the segmentation and per-operator dimension decisions live
(BENCH.md states why each boundary is where it is; WORKLOG.md has the measurements).
Every length gets statically shaped kernels (M = T is a compile-time constant); a
plan row says which tile/split configuration each linear uses at that length.
"""
from dataclasses import dataclass, field

from runtime.kernels.gemm import GemmConfig
from runtime.kernels.gemm_ws import WS_CONFIGS


@dataclass(frozen=True)
class Plan:
    name: str
    gemm: dict = field(default_factory=dict)
    norm_rows: int = 4          # rmsnorm rows (= warps) per CTA
    head_rows: int = 16
    head_threads: int = 256


# v3: SOL pass (WORKLOG "SOL"): tile widths chosen so the CTA count fills whole waves of
# the 132 SMs (gate_up HB 176 = 6.0 waves at T=1024, 3.0 at 512; HB 144 at 128/256);
# v2 values were the per-length best of the step-3 sweep.
# Keys: gdn_in / att_in = packed input projections, gate_up = two-source SwiGLU
# (BN = width of one half), o_proj / down = residual-epilogue projections.
_PLANS = {
    128: Plan(
        name="T128: memory-bound tiles (whole M in one tile, deep rings, split-K for N=5120)",
        norm_rows=1,
        gemm={
            "gdn_in": GemmConfig(BM=128, BN=128, BK=64, stages=6, swizzle=0),
            "att_in": GemmConfig(BM=128, BN=128, BK=64, stages=6, swizzle=0),
            "gate_up": GemmConfig(BM=128, BN=144, BK=128, stages=2, swizzle=0),
            "o_proj": GemmConfig(BM=128, BN=80, BK=64, stages=8, swizzle=0, split_k=2),
            "down": GemmConfig(BM=128, BN=80, BK=64, stages=8, swizzle=0, split_k=2),
        },
    ),
    256: Plan(
        name="T256: compute tiles, split-K 2 for down (64 tiles fill half the SMs), o_proj unsplit with 80-wide tiles",
        norm_rows=2,
        gemm={
            "gdn_in": GemmConfig(BM=128, BN=256, BK=64, stages=4),
            "att_in": GemmConfig(BM=128, BN=224, BK=64, stages=4),
            "gate_up": GemmConfig(BM=128, BN=144, BK=64, stages=4),
            "o_proj": GemmConfig(BM=128, BN=80, BK=64, stages=6),
            "down": GemmConfig(BM=128, BN=160, BK=64, stages=4, split_k=2),
        },
    ),
    512: Plan(
        name="T512: compute tiles",
        gemm={
            "gdn_in": GemmConfig(BM=128, BN=256, BK=64, stages=4),
            "att_in": GemmConfig(BM=128, BN=224, BK=64, stages=4),
            "gate_up": GemmConfig(BM=128, BN=176, BK=64, stages=3),
            "o_proj": GemmConfig(BM=128, BN=160, BK=64, stages=4),
            "down": GemmConfig(BM=128, BN=160, BK=64, stages=4),
        },
    ),
    1024: Plan(
        name="T1024: compute tiles",
        gemm={
            "gdn_in": GemmConfig(BM=128, BN=256, BK=64, stages=4),
            "att_in": GemmConfig(BM=128, BN=224, BK=64, stages=4),
            "gate_up": GemmConfig(BM=128, BN=176, BK=64, stages=3),
            "o_proj": GemmConfig(BM=128, BN=160, BK=64, stages=4),
            "down": GemmConfig(BM=128, BN=160, BK=64, stages=4),
        },
    ),
}


def for_length(T: int) -> Plan:
    if T not in _PLANS:
        raise ValueError(f"no plan for T={T}; planned lengths {sorted(_PLANS)}")
    return _PLANS[T]


def ws_choice(T: int, linear: str):
    """(WsConfig, epilogue) of the hand warp-specialized kernel (runtime/kernels/gemm_ws.py)
    that serves `linear` at length T, or None: the plan's gemm_kernel config serves it.
    WS_CONFIGS lists a kernel where it measured >= 1 % faster and bitwise equal
    (GEMM_WS_NOTES.md). Not used: the cluster split-K variant (csk), whose two CTAs
    exchange f32 partials through distributed shared memory -- TileFoundry cannot state a
    cross-CTA exchange (WORKLOG, gdn_scan), so its chain would hold a TIR without an order
    between the pair; it saved only ~0.17 ms per forward at T=128."""
    import os
    if os.environ.get("NO_GEMM_WS") == "1":     # A/B switch for experiments
        return None
    c = WS_CONFIGS.get(T, {}).get(linear)
    if c is None or c[0].csk:
        return None
    return c
