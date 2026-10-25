from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

N = DimVar("N", 1, 8)


@module(entry="parent", target=CpuTarget())
class ParentParameter:
    @prim_func
    def parent(x: Tensor[(N,), "f32"]):
        pass

    @parent.specialize(RangePattern("N", 1, 4))
    def small(x: Tensor[(N,), "f32"]):
        pass
