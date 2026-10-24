from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(entry='host', topologies=(Topology('thread', 32),))
class Launched:
    @prim_func(target=CudaTarget('nvidia.h200_sxm'))
    def device(a: Tensor[(32,), 'f32']):
        pass

    @prim_func(target=CpuTarget())
    def host(a: Tensor[(32,), 'f32']):
        launch(device, a, grid=(1, 1, 1), block=(32, 1, 1), cluster=(1, 1, 1),
               dynamic_smem=512)
