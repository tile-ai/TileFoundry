"""Each element of a ModuleList calls a child slot of its own: a path through two slots."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward")
class Layer:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.matmul(x, w)
        return h * tf.exp(-h)


@module(entry="forward")
class Block:
    inner = Layer

    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        return inner(x, w)


@module(entry="forward", target=cpu.Target())
class Stack:
    blocks = ModuleList(Block, 2)

    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        return blocks[1](blocks[0](x, w), w)
