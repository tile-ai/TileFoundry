"""Eight complete quads within a SIMD32 group."""
from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry_apple import MetalTarget

@module(entry="quad",target=MetalTarget("apple.m2_pro.gpu"),topologies=(Topology("threadgroup",2),Topology("thread",32)))
class Program:
    @func
    def quad(x: Tensor[(8,4),"f32",ShardLayout(Layout((8,4),(0,0)),(Split(0),Split(1)),Mesh((Topology("thread",32),),Layout((8,4),(4,1)))),"rmem"]):
        with Mesh(("threadgroup",),layout=(2,),names=("tile",)) as blocks:
            with Mesh(("thread",),layout=Layout((8,4),(4,1)),names=("quad","lane")) as threads:
                total=tf.reduce(x,axes=(-1,),keepdim=True,kind=ReduceKind.SUM)
                return total
