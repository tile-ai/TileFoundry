"""Four SIMD32 groups; reduction results broadcast within each group."""
from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry_apple import MetalTarget


@module(entry="collectives", target=MetalTarget("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 2), Topology("thread", 128)))
class Program:
    @func
    def collectives(
        x: Tensor[(4,32), "f32", ShardLayout(Layout((4, 32), (0, 0)), (Split(0), Split(1)), Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)))), "rmem"],
        integer: Tensor[(4,32), "i32", ShardLayout(Layout((4, 32), (0, 0)), (Split(0), Split(1)), Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)))), "rmem"],
        ordinary: Tensor[(4,32), "f32"],
    ):
        with Mesh(("threadgroup",), layout=(2,), names=("tile",)) as _blocks:
            with Mesh(("thread",), layout=Layout((4,32),(32,1)), names=("group","lane")) as _threads:
                total = tf.reduce(x, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                maximum = tf.reduce(x, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                minimum = tf.reduce(integer, axes=(1,), keepdim=True, kind=ReduceKind.MIN)
                mean = tf.reduce(x, axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)
                ordinary_sum = tf.reduce(ordinary, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                complete_total = tf.reshard(total, ShardLayout(Layout((4,1),(0,0)), (Split(0),Broadcast()), Mesh((Topology("thread",128),),Layout((4,32),(32,1)))), "rmem")
                complete_maximum = tf.reshard(maximum, ShardLayout(Layout((4,1),(0,0)), (Split(0),Broadcast()), Mesh((Topology("thread",128),),Layout((4,32),(32,1)))), "rmem")
                complete_minimum = tf.reshard(minimum, ShardLayout(Layout((4,1),(0,0)), (Split(0),Broadcast()), Mesh((Topology("thread",128),),Layout((4,32),(32,1)))), "rmem")
                return total, maximum, minimum, mean, ordinary_sum, complete_total, complete_maximum, complete_minimum
