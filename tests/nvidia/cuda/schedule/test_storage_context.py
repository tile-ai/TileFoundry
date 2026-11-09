"""Storage-plan contracts checked through focused IR assertions."""

from pathlib import Path

import pytest

from tilefoundry import analysis, cli
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, dim, expr, types
from tilefoundry.ir.hir import tensor
from tilefoundry.passes import transforms


def storage_plan(name):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir" / f"{name}.py"
    module = cli.load_authored_ir(str(source))
    result = analysis.analyze(*analysis.check_program(module, module.entry_function()), analysis="memory")
    return transforms.StoragePlan(result.function, valuation.Context(result.module, result.function))


def test_staged_storage_answer_reserves_two_aligned_physical_slots():
    plan = storage_plan("smem_padded_stages")
    staged = next(
        node
        for node in core.collect_exprs(plan.function.body)
        if isinstance(node, expr.Call) and getattr(node.target, "buffers", 1) == 2
    )
    answer = plan.answers[core.SSAValue(staged)]
    assert answer.partner is None
    assert answer.offsets == (0, 48)
    assert answer.refusal is None


def test_captured_update_keeps_carry_ties_and_requires_its_window_copy():
    plan = storage_plan("captured_insert_update")
    values = core.collect_exprs(plan.function.body)
    (loop,) = (value for value in values if isinstance(value, core.LoopRegion))
    (insert,) = (
        value
        for value in values
        if isinstance(value, expr.Call) and isinstance(value.target, tensor.InsertSlice)
    )
    for source, kind, target in (
        (loop.params[0], analysis.AliasKind.CARRY_INIT, loop.args[0]),
        (loop.yield_values[0], analysis.AliasKind.CARRY_YIELD, loop.params[0]),
        (insert, analysis.AliasKind.INSERT_SLICE, insert.args[0]),
    ):
        answer = plan.answers[core.SSAValue(source)]
        assert any(
            alias.kind is kind and alias.target == core.SSAValue(target) for alias in answer.aliases
        )
        assert answer.refusal is None
    update = plan.answers[core.SSAValue(insert.args[1])]
    assert not any(
        alias.kind is analysis.AliasKind.WINDOW and alias.target == core.SSAValue(insert)
        for alias in update.aliases
    )
    assert plan.answers[core.SSAValue(insert)].partner == core.SSAValue(loop.params[0])


@pytest.mark.parametrize(
    ("bits", "starts", "phase"), ((2, (8, 16), 16), (3, (16, 0), 0), (3, (0, 16), 16))
)
def test_swizzled_window_phase_is_relative_to_its_pointer(bits, starts, phase):
    source_type = types.TensorType(
        (32, 64),
        expr.DType.bf16,
        types.ComposedLayout(types.Swizzle(bits, 4, 3), 0, types.Layout((32, 64), (64, 1))),
        expr.StorageKind.SMEM,
    )
    source = expr.Var("source", type=source_type)
    function = core.Function.from_parts(name="window", params=(source,), body=source, return_type=source_type)
    plan = transforms.StoragePlan(function)
    desired = types.TensorType(
        (16, 16),
        expr.DType.bf16,
        types.ComposedLayout(types.Swizzle(bits, 4, 3), 0, types.Layout((16, 16), (64, 1))),
        expr.StorageKind.SMEM,
    )
    coordinates = tuple(dim.i64_const(start) for start in starts)
    actual_starts, actual = plan.window(source, desired, coordinates)
    assert actual_starts == coordinates
    assert actual.layout.offset == phase
    assert actual.layout.inner == source_type.layout.inner


def test_loop_induction_cannot_escape_the_builder_region():
    builder = transforms.TirBuilder([transforms.TirScope()])
    induction = expr.Var("k", type=types.TensorType.from_umat_scalar())
    bounds = (induction, dim.i64_const(0), dim.i64_const(4), dim.i64_const(1))
    with pytest.raises(transforms.LoweringError, match="value 'k' escapes region mesh"):
        with builder.region(loop=bounds) as region:
            region.result = induction
