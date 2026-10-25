"""Which variant runs is decided on the card, not before the launch.

The variants of a prototype cannot differ in launch geometry -- a translation
unit specializes ``program_dim`` once for every kernel in it -- so there is
nothing for the host to gain by choosing between them. One symbol, one kernel,
one branch on the extent the call already carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.context_builder import build_codegen_context, mangle
from tilefoundry.codegen.linkable_module import LinkableModule
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.codegen.abi import launcher_symbol

_CUDA = CudaTarget("nvidia.h200_sxm")
SQUARE = Path(__file__).parents[1] / "fixtures" / "tir" / "square.py"


def _view(square, target):
    root = build_codegen_context(square)
    group = next(group for group in root.groups if group.target == target)
    return root, group, root.for_group(group)


def _device_source(square) -> str:
    _root, group, view = _view(square, _CUDA)
    return _CUDA.resolve(
        LinkableModule, square, group.functions, _CUDA.resolve(EmitContext, view)
    ).source


def _host_source(square) -> str:
    entry = square.entry_function()
    _root, group, view = _view(square, entry.target)
    return entry.target.resolve(
        LinkableModule, square, (entry,), entry.target.resolve(EmitContext, view)
    ).source


class DeviceDispatchCase(Case):
    """The device and host sources emitted for one shape-dispatched program."""


@case
class SquareDispatch(DeviceDispatchCase):
    path: Path = SQUARE

    def guard(self, state, problems):
        device = _device_source(state)
        launcher = launcher_symbol(mangle(state, state.square_device))
        problems.run("one kernel and one shim", self.one_kernel_one_shim, device, launcher)
        problems.run("kernel branches on the extent", self.kernel_branches_on_extent, device)
        problems.run("host makes one call", self.host_makes_one_call, _host_source(state), launcher)

    @staticmethod
    def one_kernel_one_shim(source, launcher):
        assert source.count("__global__") == 1
        assert source.count(f'extern "C" const char* {launcher}(') == 2
        assert "square_small" not in source
        assert "square_large" not in source

    @staticmethod
    def kernel_branches_on_extent(source):
        """The subject is a parameter, because the type said that axis was open."""
        assert "if ((1 <= x_shape_0) && (x_shape_0 <= 127))" in source
        assert "} else if ((128 <= x_shape_0) && (x_shape_0 <= 255))" in source
        assert "__trap();" in source

    @staticmethod
    def host_makes_one_call(source, launcher):
        """No ``if`` / ``else`` over N shims: the host writes the one call the table names."""
        assert source.count(f"{launcher}(") == 2
        assert "else" not in source


CASES = special(DeviceDispatchCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_device_dispatch_case(case: DeviceDispatchCase) -> None:
    check_case(case)
