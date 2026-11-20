"""One tensor's rows split three ways: across cards, CTAs, and threads.

One mesh names all three levels, so each cuts what the one above it left and
the rows a thread touches are the intersection. A card is told which it is, a
CTA reads ``blockIdx`` and a thread reads ``threadIdx``, so what each card
ends up writing says whether the id it was told reached the device at all.
TIR rather than HIR: the claim is about what a launch is told, not about
lowering.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

GPUS, CTAS, THREADS = 2, 4, 16
ROWS, COLS = GPUS * CTAS * THREADS, 4
ROWS_PER_CARD = ROWS // GPUS
_CUDA = cuda.Target("nvidia.h200_sxm", device_count=GPUS)


_TOPOLOGIES = (Topology("gpu", GPUS), Topology("cta", CTAS), Topology("thread", THREADS))


@module(entry="copy_rows_host", topologies=_TOPOLOGIES, target=_CUDA)
class GpuPlacedRows:
    """A copy whose source and destination are both one thread's share."""

    @prim_func(target=_CUDA)
    def copy_rows_device(a: Tensor[(ROWS, COLS), "f32"], out: Tensor[(ROWS, COLS), "f32"]):
        with Mesh(
            (Topology("gpu", GPUS), Topology("cta", CTAS), Topology("thread", THREADS)),
            layout=(GPUS, CTAS, THREADS),
            names=("g", "c", "t"),
        ) as m:
            source = T.tensor_view(T.ptr_of(a), layout=((ROWS @ m.g @ m.c @ m.t, COLS), (COLS, 1)))
            written = T.tensor_view(T.ptr_of(out), layout=((ROWS @ m.g @ m.c @ m.t, COLS), (COLS, 1)))
            T.copy(source, written)

    @prim_func(target=cpu.Target())
    def copy_rows_host(a: Tensor[(ROWS, COLS), "f32"], out: Tensor[(ROWS, COLS), "f32"]):
        launch(
            copy_rows_device, a, out, grid=(CTAS, 1, 1), block=(THREADS, 1, 1)
        )
