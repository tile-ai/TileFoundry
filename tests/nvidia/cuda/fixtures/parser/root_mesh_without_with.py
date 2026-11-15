"""A function mesh with no `with`: its body may build a Mesh and name an external layout."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

LANES = Mesh((Topology("thread", 8),), layout=(8,), names=("lane",))
SUMMED = ShardLayout(Layout((8, 16), (16, 1)), (Partial("sum"),), LANES)


@module(entry="f", target=cuda.Target("nvidia.h200_sxm"))
class Model:
    @func(mesh=LANES)
    def f(x: Tensor[(8, 16), "f32", SUMMED, "rmem"]):
        lanes = Mesh((Topology("thread", 8),), layout=(8,), names=("lane",))
        return tf.reshard(x, SUMMED, "rmem")
