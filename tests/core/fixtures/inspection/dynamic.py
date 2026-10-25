from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry='dynamic', target=CpuTarget())
class DynamicLoop:
    @prim_func(target=CpuTarget())
    def dynamic(n: Tensor[(), 'i64']):
        for i in range(n):
            return
