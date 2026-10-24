from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@prim_func(target=CpuTarget())
def control(a: Tensor[(1,), 'f32']):
    with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:  # noqa: F841
        for i in range(2):
            if i < 1:
                T.abort(message='stop')
