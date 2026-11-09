from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda


@module(entry='host', target=cuda.Target('nvidia.h200_sxm'), topologies=(Topology('thread', 32),))
class Launched:
    @prim_func(target=cuda.Target('nvidia.h200_sxm'))
    def device(a: Tensor[(32,), 'f32']):
        pass

    @prim_func(target=cpu.Target())
    def host(a: Tensor[(32,), 'f32']):
        launch(device, a, grid=(1, 1, 1), block=(32, 1, 1), cluster=(1, 1, 1),
               dynamic_smem=512)
