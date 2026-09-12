"""One codegen registry: who answers, from which side of a call, about what.

A target is a dimension of the key rather than a word in a registry's name, so
a caller of one target asks the callee's target how a call to it is written
without naming anything of that target's.
"""

from __future__ import annotations

import pytest

from tests.fixtures.tir.square import TirSquare
from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.signature import (
    LAUNCH_ABI,
    ProgramIdSignature,
    ProgramMetaSignature,
    TensorSignature,
    UnitSignature,
    called_as,
)
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.visitor_registry.registries import Role, codegen_registry

_TARGET = CudaTarget("nvidia.h200_sxm")
_TENSOR = TensorSignature(
    name="x",
    type=TensorType(
        shape=(4, DimVar("S", 1, 8)),
        dtype=DType.f32,
        layout=None,
        storage=StorageKind.GMEM,
    ),
)


def _ctx() -> CudaCodegenContext:
    return CudaCodegenContext(target=_TARGET)


def test_a_node_is_written_by_the_target_being_written() -> None:
    """Emission keys on the same three things as every other question."""
    assert _ctx().handler_for(Sequential) is codegen_registry.lookup(
        (CudaTarget, Role.EMIT, Sequential)
    )


def test_a_tensor_declares_as_a_pointer_and_the_axes_its_type_leaves_open() -> None:
    """A static extent costs the convention nothing; an open one travels along."""
    assert _ctx().declare(_TENSOR) == ("float* x", "int x_shape_1")


def test_what_a_convention_adds_declares_in_the_type_it_states() -> None:
    ctx = _ctx()
    gpu_id = ProgramIdSignature(name="tilefoundry_gpu_program_id", topology_level="gpu")
    assert ctx.declare(LAUNCH_ABI[0]) == ("int grid_x",)
    assert ctx.declare(LAUNCH_ABI[-1]) == ("void* stream",)
    assert ctx.declare(gpu_id) == ("long long tilefoundry_gpu_program_id",)
    assert ctx.declare(ProgramMetaSignature(name="tilefoundry_meta")) == (
        "tilefoundry::ProgramMetaData tilefoundry_meta",
    )


def test_the_same_tensor_declares_differently_for_two_targets() -> None:
    """The host takes the runtime's tensor, which carries its own extents.

    Both answers come out of one registry and the key says which target gave
    it; the host's reads nothing off the context it is handed.
    """
    host = codegen_registry.lookup((CpuTarget, Role.CALLEE, TensorSignature))
    assert host(_TENSOR, _ctx()) == ("tvm::ffi::Tensor x",)


def test_a_caller_asks_the_callees_target_how_the_call_is_written() -> None:
    """The argument list is the callee's answer; the names in it are the caller's."""
    ctx = _ctx()
    assert ctx.argument(_TENSOR, _TARGET) == ("x", "x_shape_1")
    with pytest.raises(RuntimeError, match=r"nothing registered for \(CpuTarget, Role.CALLER"):
        ctx.argument(_TENSOR, CpuTarget())


def test_an_unanswered_question_names_the_whole_key() -> None:
    """A miss says which target, which side, and which class went unanswered."""
    with pytest.raises(
        RuntimeError, match=r"nothing registered for \(CudaTarget, Role.CALLEE, UnitSignature\)"
    ):
        _ctx().declare(UnitSignature())


def test_how_a_function_is_called_is_stated_by_its_own_target() -> None:
    device = TirSquare.lookup("square_device")
    host = TirSquare.lookup("square_host")
    assert called_as(device).name == "tilefoundry_square_device_launch"
    assert called_as(device).trailing == LAUNCH_ABI
    assert called_as(host).name == "tilefoundry_square_host_host"
    assert called_as(host).trailing == ()
