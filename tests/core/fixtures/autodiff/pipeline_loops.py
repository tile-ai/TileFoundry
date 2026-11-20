"""Four microbatches through two pipeline stages, scheduled two ways.

OneLoop hands each microbatch to the second stage as soon as the first stage
makes it. TwoLoops runs every microbatch through the first stage, moves the
whole stack, and then runs the second stage, so the stack is held there at once.

The loop index counts from literal bounds, so it is the same on every unit and
what it picks or writes stays Broadcast
([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)).
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, T, C = 4, 4, 8
target = cpu.Target()
gpu = Topology("gpu", 2)


@module(entry="forward", target=target, topologies=(gpu,))
class OneLoop:
    @func(mesh=Mesh(("gpu",), (2,), names=("pp",)))
    def forward(
        x: Tensor[(M, T, C), "f32", (M, T, C) @ mesh[0]],
        w0: Tensor[(C, C), "f32", (C, C) @ mesh[0]],
        w1: Tensor[(C, C), "f32", (C, C) @ mesh[1]],
        acc0: Tensor[(T, C), "f32", (T, C) @ mesh[1]],
    ):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            acc = acc0
            for t in range(M):
                index = tf.reshape(t, new_shape=(1,))
                with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                    h = tf.matmul(tf.reshape(tf.index_select(x, index, dim=0), new_shape=(T, C)), w0)
                moved = tf.reshard(h, (T, C) @ stages[1], "gmem")
                with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                    acc = acc + tf.matmul(moved, w1)
            return acc


@module(entry="forward", target=target, topologies=(gpu,))
class TwoLoops:
    @func(mesh=Mesh(("gpu",), (2,), names=("pp",)))
    def forward(
        x: Tensor[(M, T, C), "f32", (M, T, C) @ mesh[0]],
        w0: Tensor[(C, C), "f32", (C, C) @ mesh[0]],
        w1: Tensor[(C, C), "f32", (C, C) @ mesh[1]],
        acc0: Tensor[(T, C), "f32", (T, C) @ mesh[1]],
    ):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                held = tf.reshard(x * 0.0, (M, T, C) @ stages[0], "gmem")
                for t in range(M):
                    index = tf.reshape(t, new_shape=(1,))
                    h = tf.matmul(tf.reshape(tf.index_select(x, index, dim=0), new_shape=(T, C)), w0)
                    held = tf.insert_slice(held, tf.reshape(h, new_shape=(1, T, C)), (t, 0, 0))
            moved = tf.reshard(held, (M, T, C) @ stages[1], "gmem")
            with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                acc = acc0
                for t in range(M):
                    index = tf.reshape(t, new_shape=(1,))
                    picked = tf.reshape(tf.index_select(moved, index, dim=0), new_shape=(T, C))
                    acc = acc + tf.matmul(picked, w1)
            return acc
