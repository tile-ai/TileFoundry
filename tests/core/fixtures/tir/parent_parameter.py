from tilefoundry.dsl import *
from tilefoundry.target import cpu

N = DimVar("N", 1, 8)


@module(entry="parent", target=cpu.Target())
class ParentParameter:
    @prim_func
    def parent(x: Tensor[(N,), "f32"]):
        pass

    @parent.specialize(pattern.Range("N", 1, 4))
    def small(x: Tensor[(N,), "f32"]):
        pass
