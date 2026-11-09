from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(target=cpu.Target())
class Mixed:
    @func
    def h(a: Tensor[(1,), 'f32']):
        return a

    @prim_func(target=cpu.Target())
    def t(a: Tensor[(1,), 'f32']):
        return
