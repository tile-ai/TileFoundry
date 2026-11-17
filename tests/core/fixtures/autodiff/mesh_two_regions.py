"""Two Mesh regions that both read x and w: their gradients are added inside the mesh."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("data",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu,))
class MeshTwoRegions:
    @func
    def forward(x: placed((M, K), Split(0)), w: placed((K, N), Broadcast())):
        with Mesh(("gpu",), layout=(2,), names=("data",)) as _a:
            h = tf.matmul(x, w)
        with Mesh(("gpu",), layout=(2,), names=("data",)) as _b:
            return h * tf.exp(h) + tf.matmul(x, w)
