"""Programs for core evaluation: a scalar operand, a call, and loop carries."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="add_scalar", target=cpu.Target())
class AddScalar:
    @func
    def add_scalar(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(x, 2.0)


@module(entry="caller", target=cpu.Target())
class CallerCallee:
    @func
    def callee(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(a, b)

    @func
    def caller(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.mul(callee(a, b), b)


@module(entry="carry_sum", target=cpu.Target())
class CarrySum:
    @func
    def carry_sum(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        acc = a
        for i in range(3):
            acc = tf.add(acc, b)
        return acc


@module(entry="carry_two", target=cpu.Target())
class CarryTwo:
    @func
    def carry_two(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        p = a
        q = b
        for i in range(2):
            p = tf.add(p, b)
            q = tf.add(q, a)
        return tf.add(p, q)
