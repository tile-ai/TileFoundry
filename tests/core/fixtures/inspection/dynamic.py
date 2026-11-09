from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry='dynamic', target=cpu.Target())
class DynamicLoop:
    @prim_func(target=cpu.Target())
    def dynamic(n: Tensor[(), 'i64']):
        for i in range(n):
            return
