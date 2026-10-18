"""Storage-plan contracts checked through focused IR assertions."""

from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.analysis.access import AliasKind, SSAValue
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types import ComposedLayout, DType, Layout, StorageKind, Swizzle, TensorType
from tilefoundry.ir.types.utils import i64_const
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.passes.transforms.convert_hir_to_tir import (
    LoweringError,
    StoragePlan,
    TirBuilder,
    TirScope,
)
from tilefoundry.visitor_registry.call_frame import CallFrame


def storage_plan(name):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir" / f"{name}.py"
    module = load_authored_ir(str(source))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    return StoragePlan(result.function, CallFrame(result.module, result.function))


def test_staged_storage_answer_reserves_two_aligned_physical_slots():
    plan = storage_plan("smem_padded_stages")
    staged = next(
        expr for expr in collect_exprs(plan.function.body)
        if isinstance(expr, Call) and getattr(expr.target, "buffers", 1) == 2
    )
    answer = plan.answers[SSAValue(staged)]
    assert answer.partner is None
    assert answer.offsets == (0, 48)
    assert answer.refusal is None


def test_captured_update_keeps_carry_ties_and_requires_its_window_copy():
    plan = storage_plan("captured_insert_update")
    values = collect_exprs(plan.function.body)
    loop, = (value for value in values if isinstance(value, LoopRegion))
    insert, = (value for value in values if isinstance(value, Call) and isinstance(value.target, InsertSlice))
    for source, kind, target in (
        (loop.params[0], AliasKind.CARRY_INIT, loop.args[0]),
        (loop.yield_values[0], AliasKind.CARRY_YIELD, loop.params[0]),
        (insert, AliasKind.INSERT_SLICE, insert.args[0]),
    ):
        answer = plan.answers[SSAValue(source)]
        assert any(alias.kind is kind and alias.target == SSAValue(target) for alias in answer.aliases)
        assert answer.refusal is None
    update = plan.answers[SSAValue(insert.args[1])]
    assert not any(alias.kind is AliasKind.WINDOW and alias.target == SSAValue(insert) for alias in update.aliases)
    assert plan.answers[SSAValue(insert)].partner == SSAValue(loop.params[0])


@pytest.mark.parametrize("instruction", (False, True), ids=("storage", "instruction"))
@pytest.mark.parametrize(
    ("bits", "starts", "phase"), ((2, (8, 16), 16), (3, (16, 0), 0), (3, (0, 16), 16))
)
def test_swizzled_window_phase_is_relative_to_its_pointer(instruction, bits, starts, phase):
    source_type = TensorType(
        (32, 64),
        DType.bf16,
        ComposedLayout(Swizzle(bits, 4, 3), 0, Layout((32, 64), (64, 1))),
        StorageKind.SMEM,
    )
    source = Var("source", type=source_type)
    function = Function.build(name="window", params=(source,), body=source, return_type=source_type)
    plan = StoragePlan(function)
    desired = TensorType(
        (16, 16),
        DType.bf16,
        ComposedLayout(Swizzle(bits, 4, 3), 0, Layout((16, 16), (64, 1))),
        StorageKind.SMEM,
    )
    coordinates = tuple(i64_const(start) for start in starts)
    actual_starts, actual = plan.window(source, desired, coordinates, instruction=instruction)
    assert actual_starts == coordinates
    assert actual.layout.offset == phase
    assert actual.layout.inner == source_type.layout.inner


def test_loop_induction_cannot_escape_the_builder_region():
    builder = TirBuilder([TirScope()])
    induction = Var("k", type=TensorType.umat_scalar())
    bounds = (induction, i64_const(0), i64_const(4), i64_const(1))
    with pytest.raises(LoweringError, match="value 'k' escapes region mesh"):
        with builder.region(loop=bounds) as region:
            region.result = induction
