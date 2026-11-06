"""Matrix load/store and incompatible physical layouts."""
from tilefoundry.dsl import *
from tilefoundry_apple import MetalTarget

@module(entry="memory", target=MetalTarget("apple.m2_pro.gpu"), topologies=(Topology("threadgroup", 2), Topology("thread", 64)))
class Program:
    @func
    def memory(
        memory: Tensor[(2,8,8), "f32", ((2,8,8),(64,8,1)), "gmem"],
        shared: Tensor[(2,8,8), "f32", ((2,8,8),(64,8,1)), "smem"],
        registers: Tensor[(2,8,8), "f32", ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 64),), Layout((2, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem"],
        bad_stride: Tensor[(2,8,8), "f32", ((2,8,8),(128,16,2)), "gmem"],
    ):
        with Mesh(("threadgroup",), layout=(2,), names=("tile",)) as blocks:
            with Mesh(("thread",), layout=Layout((2,32),(32,1)), names=("group","lane")) as threads:
                load = tf.reshard(memory, ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 64),), Layout((2, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem")
                load_shared = tf.reshard(shared, ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 64),), Layout((2, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem")
                store = tf.reshard(registers, ((2,8,8),(64,8,1)), "gmem")
                bad_load = tf.reshard(bad_stride, ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 64),), Layout((2, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem")
                return load, load_shared, store, bad_load
