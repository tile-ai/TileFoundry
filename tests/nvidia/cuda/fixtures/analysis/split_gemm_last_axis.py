"""A 128-way split gemm, its weight split along the result's last axis."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_SPLIT_GRID = 128
_SPLIT_HIDDEN = 2048
_SPLIT_OUT = 12288
_SPLIT_BLOCK = 128
_H200 = CudaTarget("nvidia.h200_sxm")


@module(entry="last_axis", target=_H200, topologies=(Topology("cta", _SPLIT_GRID),))
class _SplitLastAxis:
    """The weight's N on the mesh, so the result's last axis is the split one."""

    @func
    def last_axis(
        x: Tensor[(1, _SPLIT_BLOCK, _SPLIT_HIDDEN), "bf16"],
        w: ConstTensor[(_SPLIT_HIDDEN, _SPLIT_OUT), "bf16"],
    ):
        with Mesh(("cta",), layout=(_SPLIT_GRID,), names=("unit",)) as mesh:
            rows = tf.reshard(x[:, :, 0:_SPLIT_BLOCK], (1, _SPLIT_BLOCK, _SPLIT_BLOCK), "smem")
            strip = tf.reshard(w[0:_SPLIT_BLOCK, :], (_SPLIT_BLOCK, _SPLIT_OUT @ mesh.unit), "smem")
            return tf.matmul(rows, strip, out_dtype="f32")
