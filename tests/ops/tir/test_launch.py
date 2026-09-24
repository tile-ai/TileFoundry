"""Verify host-to-device launch argument contracts."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import Tensor
from tilefoundry.ir.core import Constant, Var, VerifyError
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.ir.types import (
    Layout,
    callable_type_for_prim_function,
)
from tilefoundry.target import CpuTarget, CudaTarget


def test_launch_rejects_forwarded_type_that_differs_from_device_param() -> None:
    device_type = Tensor[(8,), 'f32', None, 'gmem']
    host_type = Tensor[(8,), 'f32', Layout(shape=(8,), strides=(1,)), 'gmem']
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
