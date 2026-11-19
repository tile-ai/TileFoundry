"""A function that calls a function that calls a function, on two H200s.

Differentiating it saves each callee's values inside its caller's, so the
generated Forward returns a tuple three levels deep.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("dp",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


X, W = placed((8, 16), Split(0)), placed((16, 16), Broadcast())


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class CallChain:
    @func
    def inner(x: X, w: W):
        return tf.silu(tf.matmul(x, w))

    @func
    def middle(x: X, w: W):
        return inner(x, w) * inner(x * x, w)

    @func
    def forward(x: X, w: W):
        return tf.reduce(middle(x, w) + x, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
