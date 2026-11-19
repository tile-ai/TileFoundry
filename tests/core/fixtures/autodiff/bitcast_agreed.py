"""A Bitcast that keeps the layout states a value's copies are equal before a Broadcast reader."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K = 4, 6
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("dp",))
AGREED = TensorType((M, K), DType.f32, ShardLayout.from_logical(
    (M, K), mesh, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu,))
class BitcastAgreed:
    @func
    def forward(x: Tensor[(M, K), "f32", "gmem"], w: AGREED):
        h = x * x
        agreed = tf.bitcast(h, AGREED.layout)
        return agreed * w
