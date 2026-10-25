"""A plain bf16 window round-tripped through thread register fragments."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class _RegisterRoundTrip:
    """The optimize tutorial's register round trip, at its own mesh and shard.

    A (2, 5120) bf16 window on cta 1 x thread (2, 4, 32): the row axis over
    ``x``, the 40 blocks over ``y``, the 128-wide block over ``t`` -- a
    40-element fragment per thread, and both crossings of the plain/shard
    boundary, since the tutorial's output copy keeps its window plain.
    """

    @prim_func(target=_CUDA)
    def round_trip(a: Tensor[(2, 5120), "bf16"], out: Tensor[(2, 5120), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("tile",)) as _cta:
            with Mesh(
                (Topology("thread", 256),), layout=(2, 4, 32), names=("x", "y", "t")
            ) as scope:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (2, 5120),
                        "bf16",
                        (2 @ scope.x, 40 @ scope.y, 128 @ scope.t),
                        "rmem",
                    ]
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value),
                    layout=(2 @ scope.x, 40 @ scope.y, 128 @ scope.t),
                    shape=(2, 5120),
                )
                T.copy(a, dst_view)
                src_view = T.tensor_view(
                    T.ptr_of(value),
                    layout=(2 @ scope.x, 40 @ scope.y, 128 @ scope.t),
                    shape=(2, 5120),
                )
                T.copy(src_view, out)


@module(entry="round_trip_host", target=_CUDA)
class RegisterRoundTrip:
    """Launch the one CTA the shard is written for."""

    round_trip = _RegisterRoundTrip

    @prim_func(target=CpuTarget())
    def round_trip_host(a: Tensor[(2, 5120), "bf16"], out: Tensor[(2, 5120), "bf16"]):
        launch(  # noqa: F821
            round_trip.round_trip,  # noqa: F821
            a,
            out,
            grid=(1, 1, 1),
            block=(256, 1, 1),
        )
