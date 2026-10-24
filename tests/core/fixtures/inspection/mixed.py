from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module()
class Mixed:
    @func
    def h(a: Tensor[(1,), 'f32']):
        return a

    @prim_func(target=CpuTarget())
    def t(a: Tensor[(1,), 'f32']):
        return
