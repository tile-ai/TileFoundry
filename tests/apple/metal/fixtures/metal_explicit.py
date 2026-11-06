"""Hand-authored TIR; no native backend or code generation required."""
from tilefoundry.dsl import *
from tilefoundry_apple import MetalTarget

@module(entry="explicit", target=MetalTarget("apple.m2_pro.gpu"), topologies=(Topology("threadgroup",2), Topology("thread",128)))
class Program:
    @func
    def explicit(
        acc: Tensor[(4,8,8), "f32", ShardLayout(Layout((4, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 128),), Layout((4, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem"],
        a: Tensor[(4,8,8), "f32", ShardLayout(Layout((4, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 128),), Layout((4, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem"],
        b: Tensor[(4,8,8), "f32", ShardLayout(Layout((4, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (Split(0), Split(4), Split(2), Split(3), Split(1)), Mesh((Topology("thread", 128),), Layout((4, 2, 4, 2, 2), (32, 1, 2, 8, 16)))), "rmem"],
    ):
        with Mesh(("threadgroup",), layout=(2,), names=("tile",)) as blocks:
            with Mesh(("thread",), layout=Layout((4,32),(32,1)), names=("group","lane")) as threads:
                product = tf.schedule((acc, a, b), op=T.metal.matmul())
                result = tf.schedule((product, a, b, product), op=T.metal.mma_m8n8k8())
                return result
