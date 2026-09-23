"""Small placed programs whose analysis results fit in handwritten arithmetic."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

S, K, N = 8, 4, 6
BM, BK, BN = 4, 2, 2

_H200 = CudaTarget("nvidia.h200_sxm")
_ONE_MIB = 1024 * 1024


@module(entry="reuse", target=_H200, topologies=(Topology("cta", 1),))
class InvariantReuse:
    """Expose an ``n``-invariant read without making it dead work.

    One ``x`` tile is ``BM * BK * sizeof(bf16) = 4 * 2 * 2 = 16 B``.
    The authored nest executes it ``(S/BM) * (N/BN) * (K/BK) = 2 * 3 * 2``
    times, so its traffic is 192 B.  Its logical account omits the invariant
    ``n`` replication and is ``2 * 2 * 16 = 64 B``.
    """

    @func
    def reuse(
        x: Tensor[(S, K), "bf16"],
    ):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            result = tf.zeros(Tensor[(BM, BK), "bf16", (BM, BK), "smem"])
            for m in tile(S, BM):  # noqa: F405
                for n in tile(N, BN):  # noqa: F405
                    for k in tile(K, BK):  # noqa: F405
                        result = tf.reshard(x[m, k], (BM, BK), "smem")
            return result


@module(entry="read", target=_H200, topologies=(Topology("cta", 1),))
class OverlappingReads:
    """Two overlapping reads count their union, not their sum.

    The ranges ``[0, 8)`` and ``[4, 12)`` contain 12 distinct bf16 elements,
    so ``x`` occupies ``12 * 2 = 24 B`` rather than ``16 * 2 = 32 B``.
    """

    @func
    def read(x: Tensor[(16,), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            left = tf.reshard(x[0:8], (8,), "rmem")
            right = tf.reshard(x[4:12], (8,), "rmem")
            return left + right


@module(entry="read", target=_H200, topologies=(Topology("cta", 1),))
class SlicedView:
    """A view chain is counted in its final source coordinates and width.

    ``x[2:10]`` reaches 8 elements of the original f32 ``x``. Reshaping the
    view changes neither source nor width, so its footprint is ``8 * 4 = 32 B``.
    """

    @func
    def read(x: Tensor[(16,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            viewed = tf.reshape(x[2:10], (2, 4))
            return tf.reshard(viewed, (2, 4), "smem")


@module(entry="store", target=_H200, topologies=(Topology("cta", 1),))
class StoreOnly:
    """A store occupies the cache even when no global boundary is read.

    The result stores 8 bf16 elements to global memory, so its write-only
    footprint is ``8 * 2 = 16 B``.
    """

    @func
    def store():
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            local = tf.zeros(Tensor[(8,), "bf16", (8,), "rmem"])
            return tf.reshard(local, (8,), "gmem")


@module(entry="read", target=_H200, topologies=(Topology("cta", 1),))
class PackedDtype:
    """Packed element bits are summed before rounding to whole bytes.

    Nine f4e2m1 elements occupy ``ceil(9 * 4 / 8) = 5 B``. Rounding each
    element separately would incorrectly report 9 B.
    """

    @func
    def read(x: Tensor[(9,), "f4e2m1"]):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            return tf.reshard(x, (9,), "rmem")


@module(entry="read", target=_H200, topologies=(Topology("cta", 256),))
class WaveTruncation:
    """Only the resident CTA wave contributes, and every CTA has distinct data.

    CTA ``i`` reads four bf16 values beginning at ``4 * i``. H200 runs 132 CTAs
    at once, so one wave reaches ``132 * 4 * 2 = 1056 B``; a target able to run
    all 256 declared CTAs reaches ``256 * 4 * 2 = 2048 B``.
    """

    @func
    def read(x: Tensor[(1024,), "bf16"]):
        with Mesh(("cta",), layout=(256,), names=("i",)) as cta:
            result = tf.zeros(Tensor[(4,), "bf16", (4,), "rmem"])
            for i in tile(cta.i * 4, (cta.i + 1) * 4, 4):  # noqa: F405
                result = tf.reshard(x[i], (4,), "rmem")
            return result


_TIGHT_L2 = CudaTarget(
    replace(_H200.device, l2_capacity_bytes=_ONE_MIB),
    architecture=_H200.architecture,
)


@module(entry="read", target=_TIGHT_L2, topologies=(Topology("cta", 1),))
class CapacityExceeded:
    """A known working set produces a hand-checkable occupancy and error.

    The input holds 786432 bf16 elements, or 1572864 B = 1.50 MiB. Against a
    1048576 B = 1.00 MiB L2 that is exactly 150.0%, so it exceeds capacity.
    """

    @func
    def read(x: Tensor[(786432,), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("cta",)) as _cta:
            return tf.reshard(x, (786432,), "rmem")


__all__ = [
    "InvariantReuse",
    "CapacityExceeded",
    "WaveTruncation",
    "OverlappingReads",
    "PackedDtype",
    "SlicedView",
    "StoreOnly",
]
