from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry='control', target=cpu.Target(), topologies=(Topology('thread', 32),))
class Control:
    @prim_func(target=cpu.Target())
    def control(a: Tensor[(1,), 'f32']):
        with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:
            for i in range(2):
                if i < 1:
                    T.abort(message='stop')
