"""The scheduling corpus's plain programs, read the way a scheduler reads them.

These are the programs an author states before choosing an instruction: a tiled
matmul chain, the same chain with both operand windows staged, its untiled
baseline, and the real matmul over a CTA grid. What each one is here to witness
is that it parses, holds together, and can be measured -- the three questions
anything downstream asks before it offers a schedule at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.metadata import MemoryMetadata, RegionMemoryMetadata
from tilefoundry.inspection import PatternPrinter, as_script
from tilefoundry.ir.core import Call, Op, Var, get_metadata
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.pattern import Tensor
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.cuda.nn.wgmma import Wgmma
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType, UnitType
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    boundary_maps,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry.visitor_registry.verify import verify_prim_function

PLAIN = (
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
)
TIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "tir").glob("*.py")))
HIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "hir").glob("*.py")))
WGMMA_DECLARATION = Path(__file__).parents[1] / "fixtures" / "schedule" / "Wgmma.described.txt"


def _prim_in(path: Path) -> PrimFunction:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if isinstance(value, PrimFunction))


def _module_in(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if type(value).__name__ == "Module")


@pytest.mark.parametrize("name", PLAIN)
def test_plain_program_is_analyzable(name: str) -> None:
    module = importlib.import_module(f"tests.fixtures.schedule.plain.{name}")
    program = next(value for value in vars(module).values() if type(value).__name__ == "Module")
    entry = next(function for function in program.functions if function.name == "gemm")
    check_program(program, entry)
    result = analyze(program, entry, analysis=("memory", "performance"))
    assert result.metadata_types


@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_scheduled_hir_program_is_well_typed(path: Path) -> None:
    program = _module_in(path)
    entry = next(function for function in program.functions if function.name == "gemm")
    check_program(program, entry)


@pytest.mark.parametrize("path", HIR, ids=lambda path: path.stem)
def test_scheduled_hir_program_has_memory_metadata(path: Path) -> None:
    program = _module_in(path)
    entry = next(function for function in program.functions if function.name == "gemm")
    if path.stem == "wgmma_tma_3stage":
        read_write = []
        for expr in collect_exprs(entry.body):
            if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp):
                schema = type(expr.target.op)._op_schema
                writes = tuple(
                    param
                    for param in schema.signature
                    if param.kind == "input" and param.effect & MemoryEffect.WRITE
                )
                if writes[0].effect & MemoryEffect.READ:
                    read_write.append(expr)
        assert len(read_write) == 1
        read_write[0].target.buffers = 3
    result = analyze(program, entry, analysis="memory")
    assert set(result.metadata_types) >= {MemoryMetadata, RegionMemoryMetadata}

    if path.stem == "wgmma_tma_3stage":
        lifetimes = get_metadata(result.function, RegionMemoryMetadata).lifetimes
        smem = sorted(item.bytes for item in lifetimes if item.memory_level == "smem")
        rmem = sorted(item.bytes for item in lifetimes if item.memory_level == "rmem")
        assert smem == sorted((512, 512 * 3, 4096, 4096 * 3))
        assert rmem == [4096, 8192, 8192, 8192, 8192]


def _copy_schedule_call(
    *, repeat=None, order=None, storage: StorageKind = StorageKind.GMEM
) -> Call:
    layout = Layout((4,), (1,))
    type_ = TensorType((4,), DType.bf16, layout, storage)
    return Call(
        target=ScheduleOp(
            op=CopyAsync(smem_layout=layout),
            repeat=repeat,
            order=order,
        ),
        args=(Var(name="src", type=type_),),
        type=type_,
    )


def test_single_issue_schedule_preserves_instruction_relations() -> None:
    schedule = _copy_schedule_call(repeat=(1,), order=(0,))
    source = schedule.args[0]
    destination_type = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.SMEM)
    instruction = Call(
        target=schedule.target.op,
        args=(source, Var(name="dst", type=destination_type)),
        type=UnitType(),
    )
    ctx = TypeInferContext()
    scheduled = boundary_maps(relations_of(schedule, ctx))
    instruction_relations = relations_of(instruction, ctx)
    single = boundary_maps(
        AccessRelations(
            inputs=(instruction_relations.inputs[0],),
            outputs=instruction_relations.outputs,
        )
    )
    assert len(scheduled) == len(single)
    assert all(left.is_equal(right) for left, right in zip(scheduled, single, strict=True))


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (_copy_schedule_call(repeat=(2,)), "conflicts with inferred repeat"),
        (_copy_schedule_call(order=(1,)), "is not a permutation"),
        (
            _copy_schedule_call(storage=StorageKind.SMEM),
            "src single-issue tile does not match",
        ),
    ),
    ids=("repeat", "order", "operand-pattern"),
)
def test_schedule_typeinfer_rejects_invalid_contract(call: Call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inference_type(call)


@register_op(dialect="T", category="schedule", name="unstated_instruction")
class _UnstatedInstruction(Op):
    value = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=Tensor,
    )


def test_schedule_typeinfer_requires_instruction_access_relations() -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    with pytest.raises(ValueError, match="states no access relations"):
        inference_type(call)


@pytest.mark.parametrize("path", TIR, ids=lambda path: path.stem)
def test_tir_program_is_verified_and_canonical(path: Path) -> None:
    function = _prim_in(path)
    verify_prim_function(function)
    assert as_script(function) == path.read_text()


def test_wgmma_declaration_is_canonical() -> None:
    assert PatternPrinter().declaration(Wgmma) + "\n" == WGMMA_DECLARATION.read_text()
