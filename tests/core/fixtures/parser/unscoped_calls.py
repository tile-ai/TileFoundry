"""A logical HIR function and a TIR function: neither needs a mesh scope around its calls."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="logical", target=CpuTarget())
class Logical:
    @func
    def logical(x: Tensor[(4,), "f32"]):
        return tf.add(x, x)


@module(entry="tir", target=CpuTarget())
class Tir:
    @prim_func
    def tir(x: Tensor[(4,), "f32"]):
        _ptr = T.ptr_of(x)
        with Mesh((Topology("thread", 1),), (1,), names=("unit",)) as _mesh:
            _value = T.ptr_of(x)
        _ptr2 = T.ptr_of(x)
