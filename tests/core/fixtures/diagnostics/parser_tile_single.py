from tilefoundry import module
from tilefoundry.dsl import Tensor, func, tf
from tilefoundry.target import CpuTarget


@module(entry="main", target=CpuTarget())
class Bad:
    @func
    def main(x: Tensor[(8,), "f32"]):
        for i in tf.tile(8):
            y = tf.relu(x)  # noqa: F841
