"""Dense BF16 GEMM schedules with hand-checkable first-iteration footprints."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")

TILE_M = 256
TILE_N = 256
TILE_K = 128

WAVE_C = 132
WAVE_G = 12
WAVE_OTHER = 11
WAVE_BM = 64
WAVE_BN = 128
WAVE_BK = 32
WAVE_M = 132 * WAVE_BM
WAVE_N = 132 * WAVE_BN
WAVE_K = 64

RESIDENT_M = 4096
RESIDENT_N = 4096
RESIDENT_BM = 128
RESIDENT_BN = 128
RESIDENT_BK = 64
RESIDENT_X = 12
RESIDENT_Y = 11
RESIDENT_K_FITS = 4096
RESIDENT_K_OVER = 16384

DEEPGEMM_X = 17
DEEPGEMM_Y = 8
DEEPGEMM_M = DEEPGEMM_X * RESIDENT_BM
DEEPGEMM_N = DEEPGEMM_Y * RESIDENT_BN
DEEPGEMM_K = 8192


@module(entry="gemm", target=_H200, topologies=(Topology("cta", 1),))
class GemmTile64:
    """A 64-square tile reaches two 64-square bf16 operand tiles.

    One K iteration reaches ``2 * 64 * 64 * 2 = 16384 B``. The full schedule
    reloads operands across the M/N tile loops, giving a different traffic
    total from the 128-square schedule below.
    """

    @func
    def gemm(a: Tensor[(TILE_M, TILE_K), "bf16"], b: Tensor[(TILE_K, TILE_N), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            result = tf.zeros(Tensor[(64, 64), "bf16", (64, 64), "rmem"])
            for m in tile(TILE_M, 64):
                for n in tile(TILE_N, 64):
                    for k in tile(TILE_K, 64):
                        lhs = tf.reshard(a[m, k], (64, 64), "smem")
                        rhs = tf.reshard(b[k, n], (64, 64), "smem")
                        result = tf.reshard(tf.matmul(lhs, rhs), (64, 64), "rmem")
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", 1),))
class GemmTile128:
    """A 128-square tile reaches two 128-square bf16 operand tiles.

    One K iteration reaches ``2 * 128 * 128 * 2 = 65536 B``, four times the
    64-square footprint, while the larger tiles change whole-Function traffic.
    """

    @func
    def gemm(a: Tensor[(TILE_M, TILE_K), "bf16"], b: Tensor[(TILE_K, TILE_N), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            result = tf.zeros(Tensor[(128, 128), "bf16", (128, 128), "rmem"])
            for m in tile(TILE_M, 128):
                for n in tile(TILE_N, 128):
                    for k in tile(TILE_K, 128):
                        lhs = tf.reshard(a[m, k], (128, 128), "smem")
                        rhs = tf.reshard(b[k, n], (128, 128), "smem")
                        result = tf.reshard(tf.matmul(lhs, rhs), (128, 128), "rmem")
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", WAVE_C),))
class GemmNaiveWave:
    """A row-major wave reuses A once and takes 132 distinct B tiles.

    The first K iteration reaches
    ``(1 * 64 + 132 * 128) * 32 * 2 = 1085440 B``.
    """

    @func
    def gemm(a: Tensor[(WAVE_M, WAVE_K), "bf16"], b: Tensor[(WAVE_K, WAVE_N), "bf16"]):
        with Mesh(("cta",), layout=(1, WAVE_C), names=("x", "y")) as cta:
            result = tf.zeros(Tensor[(WAVE_BM, WAVE_BN), "bf16", (WAVE_BM, WAVE_BN), "rmem"])
            b_columns = tf.reshape(b, (WAVE_K, WAVE_C, WAVE_BN))
            for yi in range(cta.y, WAVE_C, WAVE_C):
                for mi in tile(WAVE_M, WAVE_BM):
                    for ki in tile(WAVE_K, WAVE_BK):
                        lhs = tf.reshard(a[mi, ki], (WAVE_BM, WAVE_BK), "smem")
                        rhs = tf.reshard(
                            b_columns[ki, yi, :], (WAVE_BK, WAVE_BN), "smem"
                        )
                        result = tf.reshard(
                            tf.matmul(lhs, rhs), (WAVE_BM, WAVE_BN), "rmem"
                        )
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", WAVE_C),))
class GemmReuseAWave:
    """DeepGEMM's reuse-A candidate has 11 A tiles and 12 B tiles.

    The first K iteration reaches
    ``(ceil(132 / 12) * 64 + 12 * 128) * 32 * 2 = 143360 B``.
    Each CTA still covers an equal chunk of the same dense GEMM as reuse-B.
    """

    @func
    def gemm(a: Tensor[(WAVE_M, WAVE_K), "bf16"], b: Tensor[(WAVE_K, WAVE_N), "bf16"]):
        with Mesh(
            ("cta",),
            layout=(WAVE_OTHER, WAVE_G),
            names=("x", "y"),
        ) as cta:
            result = tf.zeros(Tensor[(WAVE_BM, WAVE_BN), "bf16", (WAVE_BM, WAVE_BN), "rmem"])
            a_groups = tf.reshape(a, (WAVE_OTHER, WAVE_M // WAVE_OTHER, WAVE_K))
            b_groups = tf.reshape(b, (WAVE_K, WAVE_G, WAVE_N // WAVE_G))
            for xi in range(cta.x, WAVE_OTHER, WAVE_OTHER):
                for yi in range(cta.y, WAVE_G, WAVE_G):
                    for mi in tile(WAVE_M // WAVE_OTHER, WAVE_BM):
                        for ni in tile(WAVE_N // WAVE_G, WAVE_BN):
                            for ki in tile(WAVE_K, WAVE_BK):
                                lhs = tf.reshard(
                                    a_groups[xi, mi, ki],
                                    (WAVE_BM, WAVE_BK),
                                    "smem",
                                )
                                rhs = tf.reshard(
                                    b_groups[ki, yi, ni],
                                    (WAVE_BK, WAVE_BN),
                                    "smem",
                                )
                                result = tf.reshard(
                                    tf.matmul(lhs, rhs),
                                    (WAVE_BM, WAVE_BN),
                                    "rmem",
                                )
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", WAVE_C),))
class GemmReuseBWave:
    """DeepGEMM's reuse-B candidate has 12 A tiles and 11 B tiles.

    The first K iteration reaches
    ``(12 * 64 + ceil(132 / 12) * 128) * 32 * 2 = 139264 B``.
    This is the smaller of the two official scheduler candidates.
    """

    @func
    def gemm(a: Tensor[(WAVE_M, WAVE_K), "bf16"], b: Tensor[(WAVE_K, WAVE_N), "bf16"]):
        with Mesh(
            ("cta",),
            layout=(WAVE_G, WAVE_OTHER),
            names=("x", "y"),
        ) as cta:
            result = tf.zeros(Tensor[(WAVE_BM, WAVE_BN), "bf16", (WAVE_BM, WAVE_BN), "rmem"])
            a_groups = tf.reshape(a, (WAVE_G, WAVE_M // WAVE_G, WAVE_K))
            b_groups = tf.reshape(b, (WAVE_K, WAVE_OTHER, WAVE_N // WAVE_OTHER))
            for xi in range(cta.x, WAVE_G, WAVE_G):
                for yi in range(cta.y, WAVE_OTHER, WAVE_OTHER):
                    for mi in tile(WAVE_M // WAVE_G, WAVE_BM):
                        for ni in tile(WAVE_N // WAVE_OTHER, WAVE_BN):
                            for ki in tile(WAVE_K, WAVE_BK):
                                lhs = tf.reshard(
                                    a_groups[xi, mi, ki],
                                    (WAVE_BM, WAVE_BK),
                                    "smem",
                                )
                                rhs = tf.reshard(
                                    b_groups[ki, yi, ni],
                                    (WAVE_BK, WAVE_BN),
                                    "smem",
                                )
                                result = tf.reshard(
                                    tf.matmul(lhs, rhs),
                                    (WAVE_BM, WAVE_BN),
                                    "rmem",
                                )
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", RESIDENT_X * RESIDENT_Y),))
class GemmResidentFits:
    """A persistent wave gives A and B distinct nested reuse windows.

    One panel is ``128 * 4096 * 2 = 1048576 B``. A's inner ``ni`` window holds
    12 A panels and 11 B panels: ``(12 + 11) MB = 24117248 B`` (48.2% of the
    50000000 B L2), and saves ``(3 * 11 - 1) * 12 MB = 402653184 B``. B's
    outer ``mi`` window contains the full nested ``ni`` sweep, so it holds 12 A
    panels plus all 32 B panels: ``(12 + 32) MB = 46137344 B`` (92.3%), and
    saves ``(3 * 12 - 1) * 32 MB = 1174405120 B``.
    """

    @func
    def gemm(
        a: Tensor[(RESIDENT_M, RESIDENT_K_FITS), "bf16"],
        b: Tensor[(RESIDENT_K_FITS, RESIDENT_N), "bf16"],
    ):
        with Mesh(
            ("cta",),
            layout=(RESIDENT_X, RESIDENT_Y),
            names=("x", "y"),
        ) as cta:
            result = tf.zeros(
                Tensor[
                    (RESIDENT_BM, RESIDENT_BN),
                    "bf16",
                    (RESIDENT_BM, RESIDENT_BN),
                    "rmem",
                ]
            )
            for mi in range(
                cta.x * RESIDENT_BM,
                RESIDENT_M,
                RESIDENT_X * RESIDENT_BM,
            ):
                for ni in range(
                    cta.y * RESIDENT_BN,
                    RESIDENT_N,
                    RESIDENT_Y * RESIDENT_BN,
                ):
                    for ki in tile(RESIDENT_K_FITS, RESIDENT_BK):
                        lhs = tf.reshard(
                            a[mi : mi + RESIDENT_BM, ki],
                            (RESIDENT_BM, RESIDENT_BK),
                            "smem",
                        )
                        rhs = tf.reshard(
                            b[ki, ni : ni + RESIDENT_BN],
                            (RESIDENT_BK, RESIDENT_BN),
                            "smem",
                        )
                        result = tf.reshard(
                            tf.matmul(lhs, rhs),
                            (RESIDENT_BM, RESIDENT_BN),
                            "rmem",
                        )
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", RESIDENT_X * RESIDENT_Y),))
class GemmResidentOver:
    """The same persistent wave exceeds L2 when K grows to 16384.

    One panel is ``128 * 16384 * 2 = 4194304 B``. A's inner ``ni`` window
    holds ``(12 + 11) * 4 MB = 96468992 B`` (92 MB), and saves
    ``(3 * 11 - 1) * 48 MB = 1610612736 B``. B's outer ``mi`` window holds
    ``48 MB + 128 MB = 184549376 B`` (176 MB), and saves
    ``(3 * 12 - 1) * 128 MB = 4697620480 B``. Both exceed the 50000000 B L2,
    so the two distinct windows each state one finding.
    """

    @func
    def gemm(
        a: Tensor[(RESIDENT_M, RESIDENT_K_OVER), "bf16"],
        b: Tensor[(RESIDENT_K_OVER, RESIDENT_N), "bf16"],
    ):
        with Mesh(
            ("cta",),
            layout=(RESIDENT_X, RESIDENT_Y),
            names=("x", "y"),
        ) as cta:
            result = tf.zeros(
                Tensor[
                    (RESIDENT_BM, RESIDENT_BN),
                    "bf16",
                    (RESIDENT_BM, RESIDENT_BN),
                    "rmem",
                ]
            )
            for mi in range(
                cta.x * RESIDENT_BM,
                RESIDENT_M,
                RESIDENT_X * RESIDENT_BM,
            ):
                for ni in range(
                    cta.y * RESIDENT_BN,
                    RESIDENT_N,
                    RESIDENT_Y * RESIDENT_BN,
                ):
                    for ki in tile(RESIDENT_K_OVER, RESIDENT_BK):
                        lhs = tf.reshard(
                            a[mi : mi + RESIDENT_BM, ki],
                            (RESIDENT_BM, RESIDENT_BK),
                            "smem",
                        )
                        rhs = tf.reshard(
                            b[ki, ni : ni + RESIDENT_BN],
                            (RESIDENT_BK, RESIDENT_BN),
                            "smem",
                        )
                        result = tf.reshard(
                            tf.matmul(lhs, rhs),
                            (RESIDENT_BM, RESIDENT_BN),
                            "rmem",
                        )
            return result


@module(entry="gemm", target=_H200, topologies=(Topology("cta", DEEPGEMM_X * DEEPGEMM_Y),))
class GemmDeepGemmWave:
    """Write DeepGEMM's 132-SM block-index decode as a two-axis mesh.

    For 128-square output tiles, candidates 8 and 16 both have usage 3200, so
    DeepGEMM chooses the first. One 132-CTA wave covers ``ceil(132 / 8) = 17``
    M blocks and all 8 N blocks: the first 132 positions of this ``(17, 8)``
    mesh. Its space-only window is one near-lockstep K iteration and holds
    ``(17 + 8) * 128 * 64 * 2 = 409600 B`` (400.00 KB). This proves the same
    tile set as DeepGEMM's rasterization; its ``usage`` chooses a group size,
    not a cache-capacity estimate.
    """

    @func
    def gemm(
        a: Tensor[(DEEPGEMM_M, DEEPGEMM_K), "bf16"],
        b: Tensor[(DEEPGEMM_N, DEEPGEMM_K), "bf16"],
    ):
        with Mesh(
            ("cta",),
            layout=(DEEPGEMM_X, DEEPGEMM_Y),
            names=("x", "y"),
        ) as cta:
            result = tf.zeros(
                Tensor[
                    (RESIDENT_BM, RESIDENT_BN),
                    "bf16",
                    (RESIDENT_BM, RESIDENT_BN),
                    "rmem",
                ]
            )
            for mi in range(
                cta.x * RESIDENT_BM,
                (cta.x + 1) * RESIDENT_BM,
                RESIDENT_BM,
            ):
                for ni in range(
                    cta.y * RESIDENT_BN,
                    (cta.y + 1) * RESIDENT_BN,
                    RESIDENT_BN,
                ):
                    for ki in tile(DEEPGEMM_K, RESIDENT_BK):
                        lhs = tf.reshard(
                            a[mi : mi + RESIDENT_BM, ki],
                            (RESIDENT_BM, RESIDENT_BK),
                            "smem",
                        )
                        rhs = tf.reshard(
                            b[ni : ni + RESIDENT_BN, ki],
                            (RESIDENT_BN, RESIDENT_BK),
                            "smem",
                        )
                        result = tf.reshard(
                            tf.matmul(lhs, rhs, b_layout="NK"),
                            (RESIDENT_BM, RESIDENT_BN),
                            "rmem",
                        )
            return result


__all__ = [
    "GemmDeepGemmWave",
    "GemmNaiveWave",
    "GemmResidentFits",
    "GemmResidentOver",
    "GemmReuseAWave",
    "GemmReuseBWave",
    "GemmTile64",
    "GemmTile128",
    "WAVE_BK",
    "WAVE_BM",
    "WAVE_BN",
    "WAVE_C",
    "WAVE_G",
    "WAVE_OTHER",
]
