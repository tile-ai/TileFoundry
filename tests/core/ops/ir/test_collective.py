"""Check collective values, participant domains and rejected payloads."""

from __future__ import annotations

import pytest
import torch

from tests.core.ops.ir.op_case import infer_call, make_call
from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding

MESH = types.Mesh(
    (types.Topology("gpu", 8),),
    types.ComposedLayout(None, 2, types.Layout((2,), (3,))),
    ("peer",),
)
MEMBERS = types.Placement.from_mesh(MESH)
TYPE = types.TensorType.from_shape((4, 2))


def evaluate(op, value):
    """Run a typed HIR function over explicit placement data."""
    call = make_call(op, (value.type,))
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    fn = ir_core.Function.from_parts(name="communicate", params=call.args, body=call, return_type=call.type)
    return evaluator.evaluate(fn, value), call


@pytest.mark.parametrize("kind,reduction", [
    (kind, reduction)
    for kind in ("broadcast", "reduce", "all_reduce", "gather", "scatter", "all_gather", "reduce_scatter")
    for reduction in (("sum", "avg", "max") if kind in {"reduce", "all_reduce", "reduce_scatter"} else ("sum",))
])
def test_collective_rank_values(kind, reduction):
    rooted = kind in {"broadcast", "reduce", "gather", "scatter"}
    reducing = kind in {"reduce", "all_reduce", "reduce_scatter"}
    assert MEMBERS == (types.Placement({"gpu": 2}), types.Placement({"gpu": 5}))
    op = sharding.Collective(participants=MESH, context="weights", kind=kind,
                             root=1 if rooted else None, reduction=reduction if reducing else None)
    a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    b = a + 20
    pieces = ((MEMBERS[1], b),) if kind in {"broadcast", "scatter"} else tuple(zip(MEMBERS, (a, b)))
    result, call = evaluate(op, core.ShardTensor(TYPE, pieces))
    combined = {"sum": a + b, "avg": (a + b) / 2, "max": torch.maximum(a, b)}[reduction]
    whole = torch.cat((a, b)) if kind in {"gather", "all_gather"} else combined if reducing else b
    expected_members = (MEMBERS[1],) if kind in {"reduce", "gather"} else MEMBERS
    assert result.placements == expected_members
    assert types.Placement.from_mesh(call.type.layout.mesh) == expected_members
    for index, member in enumerate(expected_members):
        expected = whole.chunk(2)[index] if kind in {"scatter", "reduce_scatter"} else whole
        torch.testing.assert_close(result.to_local(member), expected)
    if kind in {"reduce", "gather"}:
        with pytest.raises(ValueError, match="no piece"):
            result.to_local(MEMBERS[0])
    result.to_local(expected_members[0]).zero_()
    torch.testing.assert_close(a, torch.arange(8, dtype=torch.float32).reshape(4, 2))
    torch.testing.assert_close(b, torch.arange(8, dtype=torch.float32).reshape(4, 2) + 20)
    relations = access.relations_of(call, cost.Context())
    assert len(relations) == 2
    with pytest.raises(NotImplementedError, match="realization"):
        cost.Visitor().visit_Call(call, cost.Context())


def test_hierarchical_reduce_then_broadcast():
    participants = types.Mesh((types.Topology("gpu", 4),), types.Layout((2, 2)), ("node", "shard"))
    pieces = tuple((p, torch.tensor([float(i + 1)])) for i, p in enumerate(types.Placement.from_mesh(participants)))
    gradient_type = types.TensorType.from_shape((1,))
    roots = []
    for node in range(2):
        op = sharding.Collective(participants=participants[node, :], context=f"shard-{node}",
                                 kind="reduce", root=0, reduction="avg")
        roots.extend(evaluate(op, core.ShardTensor(gradient_type, pieces))[0].tensor)
    op = sharding.Collective(participants=participants[:, 0], context="replicas", kind="all_reduce", reduction="avg")
    averaged, _ = evaluate(op, core.ShardTensor(gradient_type, tuple(roots)))
    for node in range(2):
        op = sharding.Collective(participants=participants[node, :], context=f"publish-{node}", kind="broadcast", root=0)
        published, _ = evaluate(op, core.ShardTensor(gradient_type, averaged.tensor))
        for _, value in published.tensor:
            torch.testing.assert_close(value, torch.tensor([2.5]))


def test_root_result_cannot_feed_an_all_rank_consumer():
    op = sharding.Collective(participants=MESH, context="reduce", kind="reduce", root=1, reduction="sum")
    root_type = infer_call(op, TYPE)
    with pytest.raises(expr.VerifyError, match="source rank"):
        infer_call(sharding.Collective(participants=MESH, context="all", kind="all_reduce", reduction="sum"), root_type)


@pytest.mark.parametrize("kind", ("reduce", "gather"))
@pytest.mark.parametrize("root", (0, 3, 7))
def test_root_selection_in_sliced_mesh_with_unit_axes(kind, root):
    mesh = types.Mesh((types.Topology("gpu", 24),), types.Layout((3, 1, 2, 4)),
                      ("pp", "data", "ep", "cp"))[1, :, :, :]
    members = types.Placement.from_mesh(mesh)
    payload = types.TensorType.from_shape((1,))
    value = core.ShardTensor(payload, tuple((p, torch.tensor([float(i)])) for i, p in enumerate(members)))
    op = sharding.Collective(participants=mesh, context="sliced-stage", kind=kind,
                             root=root, reduction="sum" if kind == "reduce" else None)
    result, call = evaluate(op, value)
    assert result.placements == (members[root],)
    assert types.Placement.from_mesh(call.type.layout.mesh) == result.placements
    torch.testing.assert_close(result.to_local(members[root]),
                               torch.tensor([28.]) if kind == "reduce" else torch.arange(8, dtype=torch.float32))


def test_gather_uses_participant_order_and_local_root():
    mesh = types.Mesh((types.Topology("gpu", 4),), types.Layout((2, 2), (2, 1)), ("a", "b"))
    members = types.Placement.from_mesh(mesh)
    assert tuple(p.ids["gpu"] for p in members) == (0, 2, 1, 3)
    source = core.ShardTensor(types.TensorType.from_shape((1,)),
                              tuple((p, torch.tensor([float(p.ids["gpu"])])) for p in members))
    op = sharding.Collective(participants=mesh, context="ordered", kind="gather", root=2)
    result, call = evaluate(op, source)
    assert result.placements == types.Placement.from_mesh(call.type.layout.mesh) == (members[2],)
    torch.testing.assert_close(result.to_local(members[2]), torch.tensor([0., 2., 1., 3.]))


@pytest.mark.parametrize("changes,match", (
    ({"root": 2}, "root"),
    ({"root": True}, "root"),
    ({"context": ""}, "context"),
    ({"kind": "unknown"}, "unknown"),
    ({"reduction": "sum"}, "reduction"),
))
def test_invalid_collective_contract(changes, match):
    args = dict(participants=MESH, context="broadcast", kind="broadcast", root=1)
    with pytest.raises(expr.VerifyError, match=match):
        infer_call(sharding.Collective(**(args | changes)), TYPE)


def test_integer_avg_and_uneven_scatter_are_rejected():
    with pytest.raises(expr.VerifyError, match="floating point"):
        infer_call(sharding.Collective(participants=MESH, context="avg", kind="all_reduce", reduction="avg"),
                   types.TensorType.from_shape((4,), dtype=expr.DType.i32))
    with pytest.raises(expr.VerifyError, match="divisible"):
        infer_call(sharding.Collective(participants=MESH, context="scatter", kind="scatter", root=0),
                   types.TensorType.from_shape((3,)))


def test_local_payloads_are_not_invented_from_global_data():
    op = sharding.Collective(participants=MESH, context="sum", kind="all_reduce", reduction="sum")
    with pytest.raises(evaluator.EvalError, match="per-placement"):
        evaluate(op, core.ShardTensor(TYPE, torch.ones(4, 2)))
    with pytest.raises(evaluator.EvalError, match="no piece"):
        evaluate(op, core.ShardTensor(TYPE, ((MEMBERS[0], torch.ones(4, 2)),)))
    with pytest.raises(evaluator.EvalError, match="shape or dtype"):
        evaluate(op, core.ShardTensor(TYPE, tuple((p, torch.ones(4)) for p in MEMBERS)))
    partial = types.TensorType.from_shape((4, 2), layout=types.ShardLayout(types.Layout((4, 2)), (types.Partial("sum"),), MESH))
    with pytest.raises(expr.VerifyError, match="Local"):
        infer_call(op, partial)


def test_zero_size_scatter_and_negative_gather_axis():
    empty = types.TensorType.from_shape((0, 2))
    result, _ = evaluate(sharding.Collective(participants=MESH, context="empty", kind="scatter", root=0),
                         core.ShardTensor(empty, ((MEMBERS[0], torch.empty(0, 2)),)))
    assert all(tuple(x.shape) == (0, 2) for _, x in result.tensor)
    source = core.ShardTensor(TYPE, tuple((p, torch.full((4, 2), float(i))) for i, p in enumerate(MEMBERS)))
    result, _ = evaluate(sharding.Collective(participants=MESH, context="columns", kind="all_gather", axis=-1), source)
    torch.testing.assert_close(result.to_local(MEMBERS[0]), torch.tensor([[0., 0., 1., 1.]]).repeat(4, 1))
