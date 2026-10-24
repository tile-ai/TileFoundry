"""Partial stores retain reduction semantics and charge only the hardware transfer."""

import pytest

from tests.core.ir.test_copy_reduce_async_tensor import _types
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.types import (
    Layout,
    Mesh,
    Topology,
)
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, check_call
from tilefoundry.visitor_registry.visitors import CostEvaluator
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor


def test_reduce_store_cost_dispatches_through_schedule():
    src, dst = _types()
    args = (Var("src", type=src), Var("dst", type=dst))
    call = Call(ScheduleOp(op=CopyReduceAsyncTensor()), args, type=dst)
    cost = CostEvaluator().visit(
        call, CostContext(topology_level="cta", topologies=(Topology("cta", 2),))
    )
    assert not cost.flops and not cost.service
    assert cost.traffic == (TrafficBytes(read=16_384), TrafficBytes(write=16_384), TrafficBytes())


@pytest.mark.parametrize("reverse", (False, True))
def test_tensor_map_copy_still_rejects_partial(reverse):
    src, dst = _types()
    operands = (dst, src) if reverse else (src, dst)
    mesh = make_mesh(src.layout.mesh, Mesh((Topology("thread", 32),), Layout((32,), (1,))))
    with pytest.raises(ScheduleRefusal):
        check_call(CopyAsyncTensor(), operands, mesh, frame=None)
    check_call(CopyReduceAsyncTensor(), (src, dst), mesh, frame=None)
