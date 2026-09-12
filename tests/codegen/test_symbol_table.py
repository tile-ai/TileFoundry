"""The table every emitter reads before it writes a line.

One row per function a caller can name, settled once for a whole compile, so
a host entry and the shim it calls cannot describe the same call differently.
"""

from __future__ import annotations

import pytest

from tests.fixtures.placed.gpu_placed_rows import GPUS, GpuPlacedRows
from tests.fixtures.tir.square import TirSquare
from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.signature import (
    LAUNCH_ABI,
    ProgramIdSignature,
    TensorSignature,
    program_id_params,
    symbol_table,
)
from tilefoundry.ir.core.module import module_functions
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget

_PLACED = CudaTarget("nvidia.h200_sxm", device_count=GPUS)
_GPU_ID = ProgramIdSignature(name="tilefoundry_gpu_program_id", topology_level="gpu")


def test_the_ids_a_call_carries_are_the_levels_the_target_states() -> None:
    assert program_id_params(GpuPlacedRows, _PLACED) == (_GPU_ID,)


def test_a_program_that_names_no_stated_level_carries_no_ids() -> None:
    """The same target, a program that never names ``gpu``: nothing to tell it."""
    assert program_id_params(TirSquare, _PLACED) == ()


def test_a_device_function_is_reached_through_its_shim() -> None:
    table = symbol_table(GpuPlacedRows, _PLACED)
    shim = table[id(GpuPlacedRows.lookup("copy_rows_device"))]
    assert shim.name == "tilefoundry_copy_rows_device_launch"
    assert shim.leading == (_GPU_ID,)
    assert shim.trailing == LAUNCH_ABI
    assert [p.name for p in shim.params] == ["a", "out"]


def test_a_host_entry_is_reached_directly_and_told_what_it_places() -> None:
    """The id enters the program here: the process is the only thing that knows it."""
    table = symbol_table(GpuPlacedRows, _PLACED)
    entry = table[id(GpuPlacedRows.lookup("copy_rows_host"))]
    assert entry.name == "tilefoundry_copy_rows_host_host"
    assert entry.leading == (_GPU_ID,)
    assert entry.trailing == ()


def test_the_kernels_own_convention_is_not_in_the_table() -> None:
    """Only the shim is callable, so only the shim is written down."""
    table = symbol_table(GpuPlacedRows, _PLACED)
    assert not any(signature.name.endswith("_kernel") for signature in table.values())


def test_a_prototype_gets_one_row_and_its_variants_none() -> None:
    """Nothing calls a variant: the prototype is the one symbol, and it branches."""
    table = symbol_table(TirSquare, _PLACED)
    assert sorted(signature.name for signature in table.values()) == [
        "tilefoundry_square_device_launch",
        "tilefoundry_square_host_host",
    ]


def test_a_call_site_looks_its_callee_up_by_identity() -> None:
    """Keyed by ``id(fn)``: the call site holds the callee, not its module."""
    assert set(symbol_table(TirSquare, _PLACED)) == {
        id(fn) for fn in module_functions(TirSquare)
    }


def test_the_context_answers_from_the_table_it_was_given() -> None:
    device = GpuPlacedRows.lookup("copy_rows_device")
    ctx = CudaCodegenContext(symbols=symbol_table(GpuPlacedRows, _PLACED), target=_PLACED)
    assert ctx.signature_of(device).name == "tilefoundry_copy_rows_device_launch"
    with pytest.raises(KeyError, match="symbol table"):
        ctx.signature_of(TirSquare.lookup("square_host"))


def test_the_dynamic_axes_are_the_ones_the_type_leaves_open() -> None:
    signature = TensorSignature(
        name="x",
        type=TensorType(
            shape=(4, DimVar("S", 1, 8)),
            dtype=DType.f32,
            layout=None,
            storage=StorageKind.GMEM,
        ),
    )
    assert signature.dynamic_axes == (1,)
