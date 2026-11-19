"""A split value resharded onto mesh slices that share a unit with its own."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=4)
gpu = Topology("gpu", 4)
mesh = Mesh((gpu,), Layout((4,)), ("g",))


def on(units):
    return TensorType((4, 8), DType.f32, ShardLayout.from_logical(
        (4, 8), units, (Split(0),), storage="gmem", owners=target.storage_owners), "gmem")


MIDDLE = on(mesh[1:3])


@module(entry="run", target=target, topologies=(gpu,))
class SplitOverlap:
    @func
    def run(x: on(mesh[0:2])):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, MIDDLE.layout, "gmem")
