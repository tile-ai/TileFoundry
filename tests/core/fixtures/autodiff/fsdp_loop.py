"""FSDP over three layers in a loop: each layer's slice of the split stacked weights is gathered."""
from tilefoundry.dsl import *
from tilefoundry.target import cpu

L, T, C, F = 3, 8, 4, 6
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("shard",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


A = placed((T, C), Split(0))
Up, Down = placed((L, F, C), Split(1)), placed((L, C, F), Split(2))
FullUp, FullDown = placed((F, C), Broadcast()), placed((C, F), Broadcast())


@module(entry="forward", target=target, topologies=(gpu,))
class FsdpLayers:
    @func
    def layer(h: A, w1: FullUp, w2: FullDown):
        return h + tf.matmul(tf.silu(tf.matmul(h, w1, b_layout="NK")), w2, b_layout="NK")

    @func
    def forward(x: A, up: Up, down: Down):
        h = x
        for t in range(L):
            index = tf.reshape(t, new_shape=(1,))
            w1 = tf.reshard(tf.reshape(tf.index_select(up, index, dim=0), new_shape=(F, C)),
                            layout=FullUp.layout, storage="gmem")
            w2 = tf.reshard(tf.reshape(tf.index_select(down, index, dim=0), new_shape=(C, F)),
                            layout=FullDown.layout, storage="gmem")
            h = layer(h, w1, w2)
        return h * h
