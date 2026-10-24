"""Which variant runs is decided on the card, not before the launch.

The variants of a prototype cannot differ in launch geometry -- a translation
unit specializes ``program_dim`` once for every kernel in it -- so there is
nothing for the host to gain by choosing between them. One symbol, one kernel,
one branch on the extent the call already carries.
"""

from __future__ import annotations

from tests.fixtures.tir.square import TirSquare
from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.context_builder import build_codegen_context, mangle
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.codegen.abi import launcher_symbol

_CUDA = CudaTarget("nvidia.h200_sxm")


def _view(target):
    root = build_codegen_context(TirSquare)
    group = next(group for group in root.groups if group.target == target)
    return root, group, root.for_group(group)


def _device_source() -> str:
    _root, group, view = _view(_CUDA)
    return _CUDA.resolve(LinkableModule, TirSquare, group.functions, _CUDA.resolve(EmitContext, view)).source


def _host_source() -> str:
    entry = TirSquare.entry_function()
    _root, group, view = _view(entry.target)
    return entry.target.resolve(LinkableModule, TirSquare, (entry,), entry.target.resolve(EmitContext, view)).source


def test_a_prototype_becomes_one_kernel_and_one_shim() -> None:
    source = _device_source()
    assert source.count("__global__") == 1
    assert (
        source.count(
            f'extern "C" const char* {launcher_symbol(mangle(TirSquare, TirSquare.square_device))}('
        )
        == 2
    )
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
    assert source.count(f"{launcher_symbol(mangle(TirSquare, TirSquare.square_device))}(") == 2
    assert "else" not in source
