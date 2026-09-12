"""Which variant runs is decided on the card, not before the launch.

The variants of a prototype cannot differ in launch geometry -- a translation
unit specializes ``program_dim`` once for every kernel in it -- so there is
nothing for the host to gain by choosing between them. One symbol, one kernel,
one branch on the extent the call already carries.
"""

from __future__ import annotations

from tests.fixtures.tir.square import TirSquare
from tilefoundry.codegen.cpu.context import CpuCodegenContext
from tilefoundry.codegen.cpu.module import emit_host_module
from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.module import emit_cuda_module
from tilefoundry.codegen.signature import symbol_table
from tilefoundry.codegen.topology import launch_geometry
from tilefoundry.target import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


def _device_source() -> str:
    ctx = CudaCodegenContext(
        symbols=symbol_table(TirSquare, _CUDA),
        target=_CUDA,
        launches=launch_geometry(TirSquare),
    )
    kernels = tuple(fn for fn in TirSquare.functions if fn.target == _CUDA)
    return emit_cuda_module(TirSquare, kernels, _CUDA, ctx).source


def _host_source() -> str:
    entry = TirSquare.entry_function()
    ctx = CpuCodegenContext(symbols=symbol_table(TirSquare, _CUDA), target=entry.target)
    return emit_host_module(TirSquare, (entry,), entry.target, ctx).source


def test_a_prototype_becomes_one_kernel_and_one_shim() -> None:
    source = _device_source()
    assert source.count("__global__") == 1
    assert source.count('extern "C" void tilefoundry_square_device_launch(') == 2
    assert "square_small" not in source
    assert "square_large" not in source


def test_the_kernel_branches_on_the_extent_the_call_carries() -> None:
    """The subject is a parameter, because the type said that axis was open."""
    source = _device_source()
    assert "if ((1 <= x_shape_0) && (x_shape_0 <= 127))" in source
    assert "} else if ((128 <= x_shape_0) && (x_shape_0 <= 255))" in source
    assert "__trap();" in source


def test_the_host_makes_one_call_and_chooses_nothing() -> None:
    """No ``if`` / ``else`` over N shims: the host writes the one call the table names."""
    source = _host_source()
    assert source.count("tilefoundry_square_device_launch(") == 2
    assert "else" not in source
