from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@prim_func(target=CudaTarget('nvidia.h200_sxm'))
def device(a: Tensor[(64,), 'f32'], out: Tensor[(64,), 'f32']):
    with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:
        for i in range(0, 2, 1):
            if i < 1:
                T.sync(thread)
            else:
                T.sync(thread[:])
