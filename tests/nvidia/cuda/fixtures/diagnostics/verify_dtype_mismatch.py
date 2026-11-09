from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="main", target=cuda.Target("nvidia.h200_sxm"))
class Bad:
    @func
    def main(x: Tensor[(8,), "f32"]):
        wrong = tf.add(x, tf.cast(x, "i32"))
        return wrong
