"""A bitcast re-lays a Varying window under a plain layout; plain is Broadcast, so it is refused."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="run", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 2), Topology("thread", 32)))
class BitcastPlainEnd:
    @func
    def run(x: Tensor[(32, 8), "f32"]) -> Tensor[(8, 16), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("n",)) as cta:
            with Mesh(("thread",), layout=(32,), names=("t",)) as _threads:
                rows = x[cta.n * 16:cta.n * 16 + 16, :]
                return tf.bitcast(rows, Layout((8, 16), (1, 8)))
