"""Each CTA casts its own window of rows and clears a parameter in place."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")

_CTAS, _THREADS = 4, 8
_ROWS = _CTAS * _THREADS


@module(topologies=(Topology("cta", _CTAS), Topology("thread", _THREADS)))
class CastWindow:
    """Each CTA narrows its own window and clears a parameter in place.

    The window start is mesh-coordinate arithmetic, as a multi-CTA schedule
    writes it; ``(c // 2) * 16 + (c % 2) * 8`` is ``c * 8``.
    """

    @prim_func(target=_CUDA)
    def cast_window_device(
        a: Tensor[(_ROWS,), "f32"], out: Tensor[(_ROWS,), "f16"], filled: Tensor[(_ROWS,), "f32"]
    ):
        with Mesh((Topology("cta", _CTAS),), layout=(_CTAS,), names=("d0",)) as cta:
            with Mesh((Topology("thread", _THREADS),), layout=(_THREADS,), names=("d0",)) as thread:
                row = ShardLayout(Layout((_THREADS,), (1,)), (Split(0),), thread)
                source = T.tensor_view(
                    T.ptr_of(a[(cta.d0 // 2) * 16 + (cta.d0 % 2) * 8:(cta.d0 // 2) * 16 + (cta.d0 % 2) * 8 + 1]),
                    layout=row,
                    shape=(_THREADS,),
                )
                written = T.tensor_view(
                    T.ptr_of(out[(cta.d0 // 2) * 16 + (cta.d0 % 2) * 8:(cta.d0 // 2) * 16 + (cta.d0 % 2) * 8 + 1]),
                    layout=row,
                    shape=(_THREADS,),
                )
                held = T.alloc_tensor(Tensor[(_THREADS,), "f32", row, "rmem"])
                narrowed = T.alloc_tensor(Tensor[(_THREADS,), "f16", row, "rmem"])
                T.copy(source, held)
                T.cast(held, narrowed, dtype="f16")
                T.copy(narrowed, written)
                T.fill(filled, 0.5)


@module(entry="cast_window_host", target=_CUDA)
class CastWindowHost:
    """Launch one CTA per window."""

    window = CastWindow

    @prim_func(target=CpuTarget())
    def cast_window_host(
        a: Tensor[(_ROWS,), "f32"], out: Tensor[(_ROWS,), "f16"], filled: Tensor[(_ROWS,), "f32"]
    ):
        launch(  # noqa: F821
            window.cast_window_device,  # noqa: F821
            a,
            out,
            filled,
            grid=(_CTAS, 1, 1),
            block=(_THREADS, 1, 1),
        )
