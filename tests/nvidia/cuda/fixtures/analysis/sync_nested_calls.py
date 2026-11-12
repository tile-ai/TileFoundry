"""A root calls a wrapper instance, whose own body calls its inner instance.

The two roots differ only in the inner leaf: ``Plain``'s holds no shared memory,
``Staged``'s stages its input through shared memory. ``Twice`` calls the staged
wrapper twice in a row.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

TARGET = cuda.Target("nvidia.rtx_6000_ada")
TOPOLOGIES = (Topology("cta", 1), Topology("thread", 32))


def _plain_leaf():
    @module(entry="run", target=TARGET, topologies=TOPOLOGIES)
    class Leaf:
        @func
        def run(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                with Mesh(("thread",), layout=(32,), names=("lane",)) as _thread:
                    value_dst = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
                    value = tf.schedule((x, value_dst), op=T.copy())
                    out = tf.zeros(Tensor[(32,), "f32", None, "gmem"])
                    return tf.insert_slice(out, value, (0,))

    return Leaf


def _staged_leaf():
    @module(entry="run", target=TARGET, topologies=TOPOLOGIES)
    class Leaf:
        @func
        def run(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                with Mesh(("thread",), layout=(32,), names=("lane",)) as thread:
                    stage = tf.empty(Tensor[(32,), "f32", Layout((32,), (1,)), "smem"])
                    with thread[0:1] as _writer:
                        staged = tf.schedule((x, stage), op=T.copy())
                    value_dst = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
                    value = tf.schedule((staged, value_dst), op=T.copy())
                    out = tf.zeros(Tensor[(32,), "f32", None, "gmem"])
                    return tf.insert_slice(out, value, (0,))

    return Leaf


def _wrapper(Leaf):
    @module(entry="run", target=TARGET, topologies=TOPOLOGIES)
    class Wrapper:
        inner = ModuleList(Leaf, 1)

        @func
        def run(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                return inner[0](x)

    return Wrapper


def _twice(Leaf):
    Wrapper = _wrapper(Leaf)

    @module(entry="gemm", target=TARGET, topologies=TOPOLOGIES)
    class Twice:
        layers = ModuleList(Wrapper, 2)

        @func
        def gemm(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                first = layers[0](x)
                return layers[1](first)

    return Twice


def _root(Leaf):
    Wrapper = _wrapper(Leaf)

    @module(entry="gemm", target=TARGET, topologies=TOPOLOGIES)
    class Root:
        layers = ModuleList(Wrapper, 1)

        @func
        def gemm(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                return layers[0](x)

    return Root


Plain = _root(_plain_leaf())
Staged = _root(_staged_leaf())
Twice = _twice(_staged_leaf())
