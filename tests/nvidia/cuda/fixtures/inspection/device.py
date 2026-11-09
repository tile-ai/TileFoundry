from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry='device', target=cuda.Target('nvidia.h200_sxm'), topologies=(Topology('thread', 32),))
class Device:
    @prim_func(target=cuda.Target('nvidia.h200_sxm'))
    def device(a: Tensor[(64,), 'f32'], out: Tensor[(64,), 'f32']):
        with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:
            for i in range(0, 2, 1):
                if i < 1:
                    T.sync(thread)
                else:
                    T.sync(thread[:])
