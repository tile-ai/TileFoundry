"""Check packed uneven exchanges, zero counts and rejected transfer contracts."""

import pytest
import torch

from tests.core.ops.ir.op_case import infer_call
from tests.core.ops.ir.test_collective import MEMBERS, MESH, evaluate
from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding


def test_uneven_exchange_preserves_source_order_and_padding():
    op = sharding.Exchange(participants=MESH, context="dispatch", counts=((1, 3), (1, 0)))
    values = (torch.tensor([10., 11., 12., 13.]), torch.tensor([20., 21., -1., -1.]))
    source = core.ShardTensor(types.TensorType.from_shape((4,)), tuple(zip(MEMBERS, values)))
    result, call = evaluate(op, source)
    torch.testing.assert_close(result.to_local(MEMBERS[0]), torch.tensor([10., 20., 0.]))
    torch.testing.assert_close(result.to_local(MEMBERS[1]), torch.tensor([11., 12., 13.]))
    assert len(access.relations_of(call, cost.Context())) == 2
    with pytest.raises(NotImplementedError, match="realization"):
        cost.Visitor().visit_Call(call, cost.Context())
    result.to_local(MEMBERS[1]).zero_()
    torch.testing.assert_close(values[0], torch.tensor([10., 11., 12., 13.]))


def test_send_only_rank_and_empty_exchange():
    type_ = types.TensorType.from_shape((4,))
    op = sharding.Exchange(participants=MESH, context="activation", counts=((0, 4), (0, 0)))
    result, _ = evaluate(op, core.ShardTensor(type_, ((MEMBERS[0], torch.arange(4.)),)))
    torch.testing.assert_close(result.to_local(MEMBERS[0]), torch.zeros(4))
    torch.testing.assert_close(result.to_local(MEMBERS[1]), torch.arange(4.))
    empty = sharding.Exchange(participants=MESH, context="empty", counts=((0, 0), (0, 0)))
    result, _ = evaluate(empty, core.ShardTensor(type_, torch.zeros(4)))
    assert all(x.numel() == 0 for _, x in result.tensor)


@pytest.mark.parametrize("counts,shape,match", (
    (((1, 2),), (4,), "square"),
    (((1, -1), (0, 0)), (4,), "nonnegative"),
    (((True, 0), (0, 0)), (4,), "nonnegative"),
    (((1.5, 0), (0, 0)), (4,), "nonnegative"),
    (((1, 4), (0, 0)), (4,), "capacity"),
    (((1, 0), (0, 0)), (2, 2), "flat"),
))
def test_invalid_exchange(counts, shape, match):
    with pytest.raises(expr.VerifyError, match=match):
        infer_call(sharding.Exchange(participants=MESH, context="dispatch", counts=counts),
                   types.TensorType.from_shape(shape))


def test_missing_and_fabricated_send_payloads_are_rejected():
    op = sharding.Exchange(participants=MESH, context="dispatch", counts=((0, 1), (2, 0)))
    type_ = types.TensorType.from_shape((2,))
    with pytest.raises(evaluator.EvalError, match="per-placement"):
        evaluate(op, core.ShardTensor(type_, torch.ones(2)))
    with pytest.raises(evaluator.EvalError, match="no piece"):
        evaluate(op, core.ShardTensor(type_, ((MEMBERS[0], torch.ones(2)),)))
    with pytest.raises(evaluator.EvalError, match="shape or dtype"):
        evaluate(op, core.ShardTensor(type_, tuple((p, torch.ones(3)) for p in MEMBERS)))


def dynamic_exchange(matrix, capacity=4, *, metadata=None):
    op = sharding.Exchange(participants=MESH, context="runtime", receive_capacity=capacity)
    source_type = types.TensorType.from_shape((4,))
    count_type = types.TensorType.from_shape((2, 2), dtype=expr.DType.i64)
    arguments = tuple(expr.Var(name=name, type=ty) for name, ty in (("x", source_type), ("counts", count_type)))
    call = expr.Call(op, arguments, type=source_type)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    fn = ir_core.Function.from_parts(name="dynamic", params=arguments, body=call, return_type=call.type)
    source = core.ShardTensor(source_type, tuple((p, torch.arange(4.) + i * 10) for i, p in enumerate(MEMBERS)))
    if metadata is None:
        metadata = core.ShardTensor(count_type, torch.tensor(matrix, dtype=torch.int64))
    return evaluator.evaluate(fn, source, metadata)


def test_runtime_counts_change_without_changing_allocation_capacity():
    first = dynamic_exchange([[1, 3], [1, 0]])
    second = dynamic_exchange([[0, 0], [2, 2]])
    torch.testing.assert_close(first.to_local(MEMBERS[0]), torch.tensor([0., 10., 0., 0.]))
    torch.testing.assert_close(first.to_local(MEMBERS[1]), torch.tensor([1., 2., 3., 0.]))
    torch.testing.assert_close(second.to_local(MEMBERS[0]), torch.tensor([10., 11., 0., 0.]))
    torch.testing.assert_close(second.to_local(MEMBERS[1]), torch.tensor([12., 13., 0., 0.]))
    assert dynamic_exchange([[0, 0], [0, 0]], capacity=0).to_local(MEMBERS[0]).numel() == 0


@pytest.mark.parametrize("matrix,capacity,match", [
    ([[1, -1], [0, 0]], 4, "nonnegative"),
    ([[3, 2], [0, 0]], 4, "input capacity"),
    ([[3, 0], [2, 0]], 4, "output capacity"),
])
def test_runtime_count_capacity_checks(matrix, capacity, match):
    with pytest.raises(evaluator.EvalError, match=match):
        dynamic_exchange(matrix, capacity)


def test_runtime_metadata_replica_consistency_and_declared_type():
    held = types.TensorType.from_shape((2, 2), dtype=expr.DType.i64)
    first = torch.tensor([[1, 1], [0, 1]], dtype=torch.int64)
    inconsistent = core.ShardTensor(held, ((MEMBERS[0], first), (MEMBERS[1], first + 1)))
    with pytest.raises(evaluator.EvalError, match="disagree"):
        dynamic_exchange(None, metadata=inconsistent)
    for wrong in (first.float(), first.flatten()):
        with pytest.raises(evaluator.EvalError, match="shape or dtype"):
            dynamic_exchange(None, metadata=core.ShardTensor(held, wrong))
