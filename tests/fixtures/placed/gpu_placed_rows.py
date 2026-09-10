"""One tensor's rows split three ways: across cards, CTAs, and threads.

One mesh names all three levels, so each cuts what the one above it left and
the rows a thread touches are the intersection. A card is told which it is, a
CTA reads ``blockIdx`` and a thread reads ``threadIdx``, so what each card
ends up writing says whether the id it was told reached the device at all.
TIR rather than HIR: the claim is about what a launch is told, not about
lowering.
"""

from __future__ import annotations

import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.target import CpuTarget, CudaTarget

GPUS, CTAS, THREADS = 2, 4, 16
ROWS, COLS = GPUS * CTAS * THREADS, 4
ROWS_PER_CARD = ROWS // GPUS
_CUDA = CudaTarget("nvidia.h200_sxm", device_count=GPUS)


_TOPOLOGIES = (Topology("gpu", GPUS), Topology("cta", CTAS), Topology("thread", THREADS))


def _split_rows(mesh) -> ShardLayout:
    """The rows one instance of *mesh* holds: each level cuts what is left."""
    return ShardLayout(
        layout=Layout(shape=(ROWS, COLS), strides=(COLS, 1)),
        attrs=(Split(0), Split(0), Split(0)),
        mesh=mesh,
    )


@module(entry="copy_rows_host", topologies=_TOPOLOGIES)
class GpuPlacedRows:
    """A copy whose source and destination are both one thread's share."""

    @prim_func(target=_CUDA)
    def copy_rows_device(a: Tensor[(ROWS, COLS), "f32"], out: Tensor[(ROWS, COLS), "f32"]):
        with Mesh(
            (Topology("gpu", GPUS), Topology("cta", CTAS), Topology("thread", THREADS)),
            Layout(shape=(GPUS, CTAS, THREADS), strides=(CTAS * THREADS, THREADS, 1)),
            ("g", "c", "t"),
        ) as m:
            rows = _split_rows(m)
            source = T.tensor_view(a, layout=rows)
            written = T.tensor_view(out, layout=rows)
            T.copy(source, written)

    @prim_func(target=CpuTarget())
    def copy_rows_host(a: Tensor[(ROWS, COLS), "f32"], out: Tensor[(ROWS, COLS), "f32"]):
        launch(  # noqa: F821
            copy_rows_device, a, out, grid=(CTAS, 1, 1), block=(THREADS, 1, 1)
        )
