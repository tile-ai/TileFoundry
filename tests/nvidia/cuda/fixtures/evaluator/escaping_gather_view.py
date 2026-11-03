"""A gathered allocation remains live through an escaping slice."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,), (1,)), ("dp",))
split = ShardLayout(Layout((8, 8)), (Split(0),), mesh)
full = ShardLayout(Layout((8, 8)), (Broadcast(),), mesh)

@module(entry="run", target=CudaTarget("nvidia.h200_sxm", device_count=2), topologies=(gpu,))
class EscapingView:
    @func
    def stage(w: Tensor[(8, 8), "f32", split]):
        gathered = tf.reshard(w, layout=full, storage="gmem")
        return tf.slice(gathered, (0, 0), sizes=(4, 8), strides=(1, 1))

    @func
    def run(w: Tensor[(8, 8), "f32", split]):
        with Mesh(("gpu",), layout=(2,), names=("dp",)) as _dp:
            view = stage(w)  # noqa: F821
            other = stage(w)  # noqa: F821
            return view + other
