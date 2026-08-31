#!/usr/bin/env python
"""Two ways to state a dispatch, and neither is reachable from a real entry.

`tilefoundry tutorial authoring` puts the variants on the module's entry. That
works when the entry returns one tensor. A decode step that returns its logits
*and* the state every layer produced returns a tuple, and a dispatch prototype
needs a return annotation, whose grammar is `tensor | scalar-type` -- so the
entry of such a model cannot carry variants at all (`ToEntry` below).

The other placement -- variants on a callee, entry calls it -- parses and types,
and then fails the moment anything binds a dimension:

    specialising through 'pick': the callee dispatches on its own variants,
    which this rebuild does not choose

which is every `check --dim` and every `analyze --dim`. `ToCallee` below is the
smallest program that shows it; `Direct` is the same module with the entry
calling one variant's body instead of the prototype, and it runs.

    $ tilefoundry check repro/specialize_through_call.py:Direct   --inputs random \\
          --dim n=64 --out output --fn nan_inf          # PASS
    $ tilefoundry check repro/specialize_through_call.py:ToCallee --inputs random \\
          --dim n=64 --out output --fn nan_inf          # the rebuild error
    $ python repro/specialize_through_call.py           # the return-annotation one
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, DimVarRangePat, Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

D, W, BOUND = 64, 4, 128
N = DimVar("n", 1, 1024)
_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", W)


@module(entry="run", target=_H200, topologies=(_CTA,))
class ToCallee:
    """The dispatch is on a callee; the entry calls the prototype."""

    @func
    def pick(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        pass

    @pick.specialize(DimVarRangePat("n", 1, BOUND))
    def pick_small(x: Tensor[(1, D), "f32"],
                   k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @pick.specialize(DimVarRangePat("n", BOUND, 1024))
    def pick_big(x: Tensor[(1, D), "f32"],
                 k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @func
    def run(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        return pick(x, k)


@module(entry="run", target=_H200, topologies=(_CTA,))
class Direct:
    """The same module with the entry calling a variant's body: this one runs."""

    @func
    def pick(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        pass

    @pick.specialize(DimVarRangePat("n", 1, BOUND))
    def pick_small(x: Tensor[(1, D), "f32"],
                   k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @pick.specialize(DimVarRangePat("n", BOUND, 1024))
    def pick_big(x: Tensor[(1, D), "f32"],
                 k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @func
    def run(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        return pick_big(x, k)


def to_entry():
    """A dispatch on an entry that returns more than one tensor."""

    @module(entry="run", target=_H200, topologies=(_CTA,))
    class ToEntry:
        @func
        def run(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]):
            pass

        @run.specialize(DimVarRangePat("n", 1, BOUND))
        def run_small(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]):
            with Mesh(("cta",), layout=(W,), names=("w",)) as m:
                xs = tf.reshard(x, (1, D @ m.w), "smem")
                y = tf.reshard(xs + xs, (1, D), "gmem")
                return y, y

    return ToEntry


if __name__ == "__main__":
    try:
        to_entry()
        print("ToEntry: built")
    except Exception as error:
        print("ToEntry:", str(error).strip().splitlines()[-1][:300])
