"""Partial stores retain reduction semantics and charge only the hardware transfer."""

from dataclasses import replace

import pytest

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.ir.tir.cuda.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor
from tilefoundry.ir.types import (
    Broadcast,
    DType,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    StorageKind,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, check_call
from tilefoundry.visitor_registry.visitors import CostEvaluator


def _types():
    cta = Mesh((Topology("cta", 2),), Layout((2,), (1,)))
    src = TensorType(
        (64, 64),
        DType.f32,
        ShardLayout(Layout((64, 64), (64, 1)), (Partial("sum"),), cta),
        StorageKind.SMEM,
    )
    dst = TensorType((64, 64), DType.f32, Layout((64, 64), (64, 1)), StorageKind.GMEM)
    return src, dst


def test_reduce_store_cost_dispatches_through_schedule():
    src, dst = _types()
    args = (Var("src", type=src), Var("dst", type=dst))
    call = Call(ScheduleOp(op=CopyReduceAsyncTensor()), args, type=dst)
    cost = CostEvaluator().visit(
        call, CostContext(topology_level="cta", topologies=(Topology("cta", 2),))
    )
    assert not cost.flops and not cost.service
    assert cost.traffic == (TrafficBytes(read=16_384), TrafficBytes(write=16_384), TrafficBytes())


def test_partial_to_broadcast_reshard_is_not_a_view():
    src, _ = _types()
    dst = replace(src, layout=replace(src.layout, attrs=(Broadcast(),)))
    call = Call(Reshard(layout=dst.layout, storage=dst.storage), (Var("src", type=src),), type=dst)
    assert aliased_operand(call) is None


@pytest.mark.parametrize("reverse", (False, True))
def test_tensor_map_copy_still_rejects_partial(reverse):
    src, dst = _types()
    operands = (dst, src) if reverse else (src, dst)
    mesh = make_mesh(src.layout.mesh, Mesh((Topology("thread", 32),), Layout((32,), (1,))))
    with pytest.raises(ScheduleRefusal):
        check_call(CopyAsyncTensor(), operands, mesh, frame=None)
    check_call(CopyReduceAsyncTensor(), (src, dst), mesh, frame=None)
