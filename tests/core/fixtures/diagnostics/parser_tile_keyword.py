from tilefoundry import module
from tilefoundry.dsl import Tensor, func, tf
from tilefoundry.target import cpu


@module(entry="main", target=cpu.Target())
class Bad:
    @func
    def main(x: Tensor[(8,), "f32"]):
        for i in tf.tile(8, step=2):
            tf.relu(x)
