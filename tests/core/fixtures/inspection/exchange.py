"""Matched exchanges preserve uneven counts and isolated contexts when printed."""

from tilefoundry.dsl import Layout, Mesh, Tensor, Topology, func, module, tf
from tilefoundry.target import cpu

GPU = Topology("gpu", 8)
PEERS = Mesh((GPU,), Layout((2, 2, 2)), ("replica", "shard", "tp"))


@module(entry="dispatch", target=cpu.Target(), topologies=(GPU,))
class Communication:
    @func
    def dispatch(x: Tensor[(4,), "f32"]):
        return tf.exchange(x, participants=PEERS[1, :, 0], context="dispatch", counts=((1, 3), (2, 0)))
