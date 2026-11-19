"""Expert parallel on two GPUs, one expert each: bucket locally, all-to-all, apply, all-to-all back.

Each GPU holds its own tokens and routes `n` of them to every expert. Swapping the
split axis of the bucketed tokens is the all-to-all: before it a GPU holds what it
sends, after it what its expert receives.
"""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

P, NL, C, F = 2, 8, 8, 16
n = DimVar("n", 1, NL)
target = cpu.Target()
gpu = Topology("gpu", P)
mesh = Mesh((gpu,), Layout((P,)), ("ep",))


def placed(shape, attr, dtype=DType.f32):
    return TensorType(shape, dtype, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


Tokens = placed((P, NL, C), Split(0))
Route = placed((P * n,), Varying(), DType.i32)
Expert = placed((C, F), Varying())
Received = placed((P, P * n, C), Split(1))
Returned = placed((P, P * n, F), Split(0))


@module(entry="forward", target=target, topologies=(gpu,))
class ExpertParallel:
    @func
    def forward(tokens: Tokens, route: Route, w: Expert):
        with Mesh(("gpu",), layout=(P,), names=("ep",)) as _ep:
            sent = tf.index_select(tokens, route, dim=1)
            received = tf.reshard(sent, layout=Received.layout, storage="gmem")
            applied = tf.matmul(received, w)
            return tf.reshard(applied, layout=Returned.layout, storage="gmem")
