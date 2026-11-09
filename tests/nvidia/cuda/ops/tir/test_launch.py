"""Verify host-to-device launch contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tests.nvidia.cuda.ops.tir.targets import CUDA_TARGETS, retarget
from tilefoundry import cli, verifier
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr, tir, types
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda


def test_launch_rejects_forwarded_type_mismatch() -> None:
    device_type = Tensor[(8,), 'f32', None, 'gmem']
    host_type = Tensor[(8,), "f32", types.Layout(shape=(8,), strides=(2,)), "gmem"]
    device_arg = expr.Var(type=device_type, name="device_arg")
    device = core.PrimFunction(
        name="device",
        params=(device_arg,),
        body=core.Sequential(body=()),
        target=cuda.Target("nvidia.h200_sxm"),
    )

    host_arg = expr.Var(type=host_type, name="host_arg")
    extent_type = Tensor[(), "i64", None, "rmem"]
    one = expr.Constant(type=extent_type, value=1)
    ref = core.SymbolRef(name=device.name, type=types.callable_type_for_prim_function(device))
    launch = core.Evaluate(callable=tir.launch.Launch(), args=(ref, one, one, one, one, one, one, host_arg))
    host = core.PrimFunction(
        name="host",
        params=(host_arg,),
        body=core.Sequential(body=(launch,)),
        target=cpu.Target(),
    )

    with pytest.raises(expr.VerifyError, match=r"forwarded arg\[0\] type"):
        verifier.verify_module((host, device))


OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = cuda.Target("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refused_launch_raises() -> None:
    OverAskingLaunch = cli.load_authored_ir(f"{OPS / 'over_asking_launch.py'}:OverAskingLaunch")
    runtime = tilefoundry.compile(OverAskingLaunch, target=_CUDA)
    with pytest.raises(RuntimeError, match=r"tilefoundry: launch of \w+ failed: "):
        runtime(torch.ones(4, 32, device="cuda"))


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_launch_lifts_the_dynamic_shared_memory_default(target: cuda.Target) -> None:
    """An 80 KiB stage runs once its launch asks for 80 KiB and lifts CUDA's 48 KiB default."""
    WideSharedStage = cli.load_authored_ir(f"{OPS / 'wide_shared_stage.py'}:WideSharedStage")
    runtime = tilefoundry.compile(retarget(WideSharedStage, target), target=target)
    torch.manual_seed(5)
    source = torch.randn(128, 160, dtype=torch.float32, device="cuda")
    out = torch.empty_like(source)
    runtime(source, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, source, rtol=0, atol=0)
