"""Small placed programs whose analysis results fit in handwritten arithmetic."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

S, K, N = 8, 4, 6
BM, BK, BN = 4, 2, 2

_H200 = CudaTarget("nvidia.h200_sxm")


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


__all__ = ["InvariantReuse"]
