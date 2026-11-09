"""Partial stores retain reduction semantics and charge only the hardware transfer."""

import pytest

from tests.core.ir.test_partial_reshard import _types
from tilefoundry.interfaces import cost, schedule
from tilefoundry.ir import expr, hir, types
from tilefoundry_nvidia.cuda.ir.tir import memory


def test_reduce_store_cost_via_schedule():
    src, dst = _types()
    args = (expr.Var("src", type=src), expr.Var("dst", type=dst))
    call = expr.Call(hir.schedule.ScheduleOp(op=memory.CopyReduceAsyncTensor()), args, type=dst)
    cost_value = cost.Visitor().visit(
        call, cost.Context(topology_level="cta", topologies=(types.Topology("cta", 2),))
    )
    assert not cost_value.flops and not cost_value.service
    assert cost_value.traffic == (cost.TrafficBytes(read=16_384), cost.TrafficBytes(write=16_384), cost.TrafficBytes())


@pytest.mark.parametrize("reverse", (False, True))
def test_tensor_map_copy_rejects_partial(reverse):
    src, dst = _types()
    operands = (dst, src) if reverse else (src, dst)
    mesh = types.Mesh.from_meshes(src.layout.mesh, types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,))))
    with pytest.raises(schedule.RefusalError):
        schedule.check_call(memory.CopyAsyncTensor(), operands, mesh, frame=None)
    schedule.check_call(memory.CopyReduceAsyncTensor(), (src, dst), mesh, frame=None)
