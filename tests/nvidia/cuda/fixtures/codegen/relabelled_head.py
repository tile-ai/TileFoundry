"""The first 16 of 32 threads, selected and given a layout of their own, copy x."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="run",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class RelabelledHead:
    @func
    def run(x: Tensor[(16,), "f32"]) -> Tensor[(16,), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("t",)) as threads:
                with Mesh(threads[0:16], layout=(16,), names=("head",)) as head:
                    split = ShardLayout(layout=Layout((16,), (1,)), attrs=(S(0),), mesh=head)
                    return tf.schedule((tf.reshard(x, split), tf.empty(Tensor[(16,), "f32"])), op=T.copy())
