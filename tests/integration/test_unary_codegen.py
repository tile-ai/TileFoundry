"""Every unary kind the DSL exposes compiles and matches its torch oracle.

`tf.exp`, `tf.log`, `tf.abs` and friends type-infer and evaluate, but only a
handful of `UnaryKind` members had a `_UNARY_TAG` row and a runtime functor, so
the rest could be written and checked and then failed to emit.

Each kernel here is a separate top-level function: the compiled artifact is
cached per function, so a factory producing several same-named kernels would
hand every op whichever one compiled first.

Exactness matters for two of these. `round` must break halfway cases to even to
agree with torch — `roundf` would send them away from zero instead — so the
input grid below lands exactly on `.5`. The transcendentals use the precise
`<math.h>` entries rather than the `__`-prefixed intrinsics, which is why they
are compared at a tight tolerance rather than a loose one.
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

_ROWS, _COLS = 64, 32


@func(topologies=(Topology("cta", _ROWS),))
def u_exp(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.exp(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


@func(topologies=(Topology("cta", _ROWS),))
def u_exp2(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.exp2(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


@func(topologies=(Topology("cta", _ROWS),))
def u_log(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.log(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


@func(topologies=(Topology("cta", _ROWS),))
def u_log2(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.log2(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


@func(topologies=(Topology("cta", _ROWS),))
def u_abs(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.abs(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


@func(topologies=(Topology("cta", _ROWS),))
def u_ceil(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.ceil(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


@func(topologies=(Topology("cta", _ROWS),))
def u_round(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(topology="cta", layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        r = tf.reshard(a, layout=(_ROWS @ cta, _COLS), storage=rmem)
        return tf.reshard(tf.round(r), layout=(_ROWS @ cta, _COLS), storage=gmem)


def _positive() -> torch.Tensor:
    """Strictly positive input, for the ops whose domain needs it."""
    torch.manual_seed(0)
    return torch.rand(_ROWS, _COLS, dtype=torch.float32, device="cuda") * 8.0 + 0.25


def _signed() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(_ROWS, _COLS, dtype=torch.float32, device="cuda") * 4.0


def _halves() -> torch.Tensor:
    """A grid landing exactly on .5 boundaries, where rounding mode shows."""
    n = _ROWS * _COLS
    return (
        (torch.arange(n, dtype=torch.float32, device="cuda") - n // 2) * 0.5
    ).reshape(_ROWS, _COLS)


@pytest.mark.parametrize(
    "kernel,oracle,make_input",
    [
        (u_exp, torch.exp, _signed),
        (u_exp2, torch.exp2, _signed),
        (u_log, torch.log, _positive),
        (u_log2, torch.log2, _positive),
        (u_abs, torch.abs, _signed),
        (u_ceil, torch.ceil, _signed),
        (u_round, torch.round, _halves),
    ],
    ids=["exp", "exp2", "log", "log2", "abs", "ceil", "round"],
)
def test_a_unary_kernel_matches_its_torch_oracle(kernel, oracle, make_input) -> None:
    rm = tilefoundry.compile(kernel, target="cuda")
    x = make_input()
    out = torch.full_like(x, float("nan"))
    rm(x, out)
    torch.cuda.synchronize()

    expected = oracle(x)
    # abs / ceil / round are exact; the transcendentals are compared at single
    # -precision ulp scale, which the precise math entries meet and the
    # ``__``-prefixed intrinsics would not.
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
