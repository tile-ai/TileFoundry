"""Verify host-to-device launch contracts."""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry.dsl import *
from tilefoundry.ir.core import Constant, Var, VerifyError
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import Layout
from tilefoundry.ir.types.callable_type import callable_type_for_prim_function
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia import CudaTarget


def test_launch_rejects_forwarded_type_that_differs_from_device_param() -> None:
    device_type = Tensor[(8,), 'f32', None, 'gmem']
    host_type = Tensor[(8,), "f32", Layout(shape=(8,), strides=(2,)), "gmem"]
    device_arg = Var(type=device_type, name="device_arg")
    device = PrimFunction(
        name="device",
        params=(device_arg,),
        body=Sequential(body=()),
        target=CudaTarget("nvidia.h200_sxm"),
    )

    host_arg = Var(type=host_type, name="host_arg")
    extent_type = Tensor[(), "i64", None, "rmem"]
    one = Constant(type=extent_type, value=1)
    ref = SymbolRef(name=device.name, type=callable_type_for_prim_function(device))
    launch = Evaluate(callable=Launch(), args=(ref, one, one, one, one, one, one, host_arg))
    host = PrimFunction(
        name="host",
        params=(host_arg,),
        body=Sequential(body=(launch,)),
        target=CpuTarget(),
    )

    with pytest.raises(VerifyError, match=r"forwarded arg\[0\] type"):
        verify_module((host, device))


_CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="over_asking_host", target=_CUDA, topologies=(Topology("thread", 128),))
class OverAskingLaunch:
    """One kernel launched with 1 MiB of dynamic shared memory.

    No CUDA device grants a block that much, so every device refuses the launch.
    """

    @prim_func(target=_CUDA)
    def over_asking_device(a: Tensor[(4, 32), "f32"]):
        with Mesh((Topology("thread", 128),), layout=(4, 32), names=("w", "t")) as m:
            view = T.tensor_view(T.ptr_of(a), layout=((4 @ m.w, 32 @ m.t), (32, 1)))
            reg = T.alloc_tensor(
                tensor_type=Tensor[(4, 32), "f32", ((4 @ m.w, 32 @ m.t), (32, 1)), "rmem"]
            )
            T.copy(view, reg)
            T.copy(reg, view)

    @prim_func(target=CpuTarget())
    def over_asking_host(a: Tensor[(4, 32), "f32"]):
        launch(over_asking_device, a, grid=(1, 1, 1), block=(128, 1, 1), dynamic_smem=1048576)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_launch_the_device_refuses_raises() -> None:
    runtime = tilefoundry.compile(OverAskingLaunch, target=_CUDA)
    with pytest.raises(RuntimeError, match=r"tilefoundry: launch of \w+ failed: "):
        runtime(torch.ones(4, 32, device="cuda"))
