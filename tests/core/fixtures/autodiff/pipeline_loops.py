"""Four microbatches through two pipeline stages, scheduled two ways.

OneLoop hands each microbatch to the second stage as soon as the first stage
makes it. TwoLoops runs every microbatch through the first stage, moves the
whole stack, and then runs the second stage, so the stack is held there at once.

The loop index counts from literal bounds, so it is the same on every unit and
what it picks or writes stays Broadcast
([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)).
"""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, T, C = 4, 4, 8
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("pp",))


def on(stage, shape, attr=Broadcast()):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, stage, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


H1 = on(mesh[1], (T, C))
STACK0, STACK1 = on(mesh[0], (M, T, C)), on(mesh[1], (M, T, C))


@module(entry="forward", target=target, topologies=(gpu,))
class OneLoop:
    @func
    def forward(x: on(mesh[0], (M, T, C)), w0: on(mesh[0], (C, C)), w1: on(mesh[1], (C, C)),
                acc0: on(mesh[1], (T, C))):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            acc = acc0
            for t in range(M):
                index = tf.reshape(t, new_shape=(1,))
                with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                    h = tf.matmul(tf.reshape(tf.index_select(x, index, dim=0), new_shape=(T, C)), w0)
                moved = tf.reshard(h, H1.layout, "gmem")
                with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                    acc = acc + tf.matmul(moved, w1)
            return acc


@module(entry="forward", target=target, topologies=(gpu,))
class TwoLoops:
    @func
    def forward(x: on(mesh[0], (M, T, C)), w0: on(mesh[0], (C, C)), w1: on(mesh[1], (C, C)),
                acc0: on(mesh[1], (T, C))):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                held = tf.reshard(x * 0.0, STACK0.layout, "gmem")
                for t in range(M):
                    index = tf.reshape(t, new_shape=(1,))
                    h = tf.matmul(tf.reshape(tf.index_select(x, index, dim=0), new_shape=(T, C)), w0)
                    held = tf.insert_slice(held, tf.reshape(h, new_shape=(1, T, C)), (t, 0, 0))
            moved = tf.reshard(held, STACK1.layout, "gmem")
            with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                acc = acc0
                for t in range(M):
                    index = tf.reshape(t, new_shape=(1,))
                    picked = tf.reshape(tf.index_select(moved, index, dim=0), new_shape=(T, C))
                    acc = acc + tf.matmul(picked, w1)
            return acc
