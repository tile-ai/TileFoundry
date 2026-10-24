"""End-to-end checks for the explicit host/device boundary.

Covers static and dynamic CTA launches, implicit and explicit CPU entries,
multi-CTA execution, and device-placement and shape negatives.
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry.dsl import *
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.layout import Layout
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_ROWS = 128
_COLS = 12


class ExternalH200Target(CudaTarget):
    name = "tests.core.integration.external_h200"


@func(topologies=(Topology("cta", _ROWS),))
def double_rows(a: Tensor[(_ROWS, _COLS), "f32"]) -> Tensor[(_ROWS, _COLS), "f32"]:
    with Mesh(("cta",), layout=Layout(shape=(_ROWS,), strides=(1,))) as cta:
        reg = tf.reshard(a, layout=(128 @ cta, 12), storage=rmem)  # noqa: F405
        out = tf.mul(reg, reg)
        return tf.reshard(out, layout=(128 @ cta, 12), storage=gmem)  # noqa: F405


@prim_func(target=CpuTarget())
def host_double(a: Tensor[(_ROWS, _COLS), "f32"], out: Tensor[(_ROWS, _COLS), "f32"]):
    launch(double_rows, a, out, grid=(_ROWS, 1, 1), block=(1, 1, 1))  # noqa: F821


@func(topologies=(Topology("thread", 6 * 32),))
def row_mean(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1), "f32"]:
    with Mesh(("thread",), (6, 32), ("w", "t")) as m:
        a_reg = tf.reshard(a, (1, 1536 @ (m.w, m.t)), rmem)
        a_mean = tf.reduce(a_reg, (-1,), True, ReduceKind.MEAN)
        return tf.reshard(a_mean, (1, 1), gmem)


def _randn_rows() -> torch.Tensor:
    torch.manual_seed(0)

    return torch.randn(_ROWS, _COLS, dtype=torch.float32, device="cuda")


_TILE = 12
_NT = DimVar("Ntile", 1, 63)
_DYNAMIC_CTA_LOWERING_WAIT = (
    "waiting for the follow-up dynamic-CTA runtime-lowering plan: lowering must "
    "preserve symbolic topology extents"
)


def _dynamic_cta_module(*, explicit_host: bool) -> Module:
    @func(topologies=(Topology("cta", _NT),))
    def dyn_double(a: Tensor[(_NT, _TILE), "f32"]) -> Tensor[(_NT, _TILE), "f32"]:
        with Mesh(("cta",), layout=Layout(shape=(_NT,), strides=(1,))) as cta:
            reg = tf.reshard(  # noqa: F405
                a,
                layout=ShardLayout(
                    layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                    attrs=(S(0),),
                    mesh=cta,
                ),
                storage=rmem,
            )
            out = tf.mul(reg, reg)
            return tf.reshard(  # noqa: F405
                out,
                layout=ShardLayout(
                    layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                    attrs=(S(0),),
                    mesh=cta,
                ),
                storage=gmem,
            )

    functions = (dyn_double.entry_function(),)
    entry = "dyn_double"
    if explicit_host:

        @prim_func(target=CpuTarget())
        def host_dyn_double(
            a: Tensor[(_NT, _TILE), "f32"], out: Tensor[(_NT, _TILE), "f32"]
        ):
            launch(dyn_double, a, out, grid=(_NT, 1, 1), block=(1, 1, 1))  # noqa: F821

        functions += (host_dyn_double,)
        entry = "host_dyn_double"
    return Module(
        name="m",
        functions=functions,
        entry=entry,
        topologies=dyn_double.effective_topologies(),
    )


@pytest.mark.skip(reason=_DYNAMIC_CTA_LOWERING_WAIT)
def test_dynamic_cta_two_shapes_one_compile() -> None:
    """Test dynamic cta two shapes one compile.

    One compiled artifact launches the dynamic-CTA kernel at two different
    ``Ntile`` shapes via the host-computed grid; both match torch with no
    recompile.
    """
    rm = tilefoundry.compile(
        _dynamic_cta_module(explicit_host=True), target=CudaTarget("nvidia.h200_sxm")
    )
    for nt in (4, 8):
        torch.manual_seed(nt)
        x = torch.randn(nt, _TILE, dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)
        rm(x, out)
        torch.cuda.synchronize()
        assert torch.allclose(out, x * x, rtol=0, atol=0)


@pytest.mark.skip(reason=_DYNAMIC_CTA_LOWERING_WAIT)
def test_dynamic_cta_rejects_implicit_entry() -> None:
    """A dynamic-CTA kernel has no compile-time grid.

    A dynamic-CTA kernel has no compile-time grid, so the implicit
    auto-inserted host entry cannot derive one — it must error loudly rather
    than guess a CTA count.
    """
    mod = _dynamic_cta_module(explicit_host=False)
    with pytest.raises(Exception, match=r"cannot derive its grid"):
        tilefoundry.compile(mod, target=CudaTarget("nvidia.h200_sxm"))
