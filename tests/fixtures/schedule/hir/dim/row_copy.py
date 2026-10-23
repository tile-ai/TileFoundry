"""Finalize an explicitly scheduled copy after binding its row extent."""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget

ROWS = DimVar("rows", 1, 256)


@module(
    entry="copy_rows",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class RowCopy:
    @func
    def copy_rows(x: Tensor[(ROWS,), "f32"]) -> Tensor[(ROWS,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                dst = tf.empty(Tensor[(ROWS,), "f32", (ROWS,), "rmem"])
                return tf.schedule((x, dst), op=T.copy())
