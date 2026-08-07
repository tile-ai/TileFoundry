"""A reshard back to global memory keeps its numerics when the store vectorizes.

``copy_fragment`` takes a 128-bit path when the register-side source is a
statically contiguous run and the destination is global. That path packs whole
``uint4`` groups and copies the remainder one element at a time, so a run length
that is *not* a multiple of four f32 is the interesting case: an off-by-one in
the tail loop drops or duplicates the last elements of every thread's run, which
no aligned length can reveal.

Each kernel below is a distinct top-level function on purpose. The compiled
artifact is cached per function, so a factory producing several same-named
kernels would hand every size the first one that was compiled.
"""

from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

_ROWS = 64


# 4 f32 == exactly one 128-bit group, no tail.
@func(topologies=(Topology("cta", _ROWS),))
def square_4(a: Tensor[(_ROWS, 4), "f32"]) -> Tensor[(_ROWS, 4), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        reg = tf.reshard(a, layout=(_ROWS @ cta, 4), storage=rmem)
        return tf.reshard(tf.mul(reg, reg), layout=(_ROWS @ cta, 4), storage=gmem)


# One full group plus a 2-element tail.
@func(topologies=(Topology("cta", _ROWS),))
def square_6(a: Tensor[(_ROWS, 6), "f32"]) -> Tensor[(_ROWS, 6), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        reg = tf.reshard(a, layout=(_ROWS @ cta, 6), storage=rmem)
        return tf.reshard(tf.mul(reg, reg), layout=(_ROWS @ cta, 6), storage=gmem)


# Three full groups plus a 1-element tail.
@func(topologies=(Topology("cta", _ROWS),))
def square_13(a: Tensor[(_ROWS, 13), "f32"]) -> Tensor[(_ROWS, 13), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        reg = tf.reshard(a, layout=(_ROWS @ cta, 13), storage=rmem)
        return tf.reshard(tf.mul(reg, reg), layout=(_ROWS @ cta, 13), storage=gmem)


# A long run — 64 full groups plus a 3-element tail.
@func(topologies=(Topology("cta", _ROWS),))
def square_259(a: Tensor[(_ROWS, 259), "f32"]) -> Tensor[(_ROWS, 259), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        reg = tf.reshard(a, layout=(_ROWS @ cta, 259), storage=rmem)
        return tf.reshard(tf.mul(reg, reg), layout=(_ROWS @ cta, 259), storage=gmem)


def _check(kernel, cols: int) -> None:
    rm = tilefoundry.compile(kernel, target="cuda")
    torch.manual_seed(0)
    # Away from zero so a dropped tail element cannot coincidentally match, and
    # the output starts as NaN so an unwritten element never looks right.
    x = torch.randn(_ROWS, cols, dtype=torch.float32, device="cuda") + 2.0
    out = torch.full_like(x, float("nan"))
    rm(x, out)
    torch.cuda.synchronize()

    expected = x * x
    assert torch.allclose(out, expected, rtol=0, atol=0), (
        f"cols={cols}: {int((out != expected).sum())} of {out.numel()} wrong "
        f"(tail of {cols % 4} elements past the last full 128-bit group)"
    )


def test_a_store_of_exactly_one_vector_group_is_exact() -> None:
    _check(square_4, 4)


def test_a_store_with_a_two_element_tail_is_exact() -> None:
    _check(square_6, 6)


def test_a_store_with_a_one_element_tail_is_exact() -> None:
    _check(square_13, 13)


def test_a_long_store_with_a_three_element_tail_is_exact() -> None:
    _check(square_259, 259)
