"""Collectives retain their isolated context and participant view when printed."""

from tilefoundry.dsl import Layout, Mesh, Tensor, Topology, func, module, tf
from tilefoundry.target import cpu

GPU = Topology("gpu", 8)
PEERS = Mesh((GPU,), Layout((2, 2, 2)), ("replica", "shard", "tp"))


@module(entry="gradients", target=cpu.Target(), topologies=(GPU,))
class Communication:
    @func
    def gradients(x: Tensor[(4, 2), "f32"]):
        first = tf.collective(x, participants=PEERS[1, :, 0], context="gradient", kind="all_reduce", reduction="sum")
        return tf.collective(first, participants=PEERS[1, :, 0], context="publish", kind="broadcast", root=1)
