"""Exercise completion, aliasing, groups, event generations, and repeated host protocols."""

import pytest

from tests.core.ops.ir.test_collective import MESH
from tilefoundry import analysis
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import sharding
from tilefoundry.ir.tir import communication as C


def protocol():
    buffers = [expr.Var(name=name, type=types.TensorType.from_shape((4,))) for name in ("a", "b", "c")]
    comm = expr.Var(name="comm", type=types.CommunicationHandleType("communicator", "data", MESH))
    handles = [expr.Var(name=name, type=types.CommunicationHandleType(kind, name))
               for name, kind in (("s0", "stream"), ("s1", "stream"), ("e0", "event"), ("e1", "event"))]
    op = sharding.Collective(participants=MESH, context="data", kind="all_reduce", reduction="sum")
    def enqueue(src, dst, stream):
        return core.Evaluate(C.CommEnqueue(operation=op), (src, dst, comm, stream))
    return (*buffers, *handles, comm), enqueue


def check(params, statements, **kwargs):
    fn = core.PrimFunction("host", tuple(params), core.Sequential(tuple(statements)), output_count=0)
    return analysis.check_communication(fn, **kwargs)


def effect(op, *args):
    return core.Evaluate(op(), args)


def test_enqueue_and_group_end_do_not_complete():
    params, enqueue = protocol()
    a, b, _, s0, *_ = params
    for statements in ((enqueue(a, b, s0),), (effect(C.CommGroupStart), enqueue(a, b, s0), effect(C.CommGroupEnd))):
        with pytest.raises(ValueError, match="pending"):
            check(params, statements)


def test_cross_stream_consumer_requires_the_matching_event():
    params, enqueue = protocol()
    a, b, c, s0, s1, e0, e1, _ = params
    prefix = [effect(C.EventRecord, e1, s1), enqueue(a, b, s0), effect(C.EventRecord, e0, s0)]
    suffix = [enqueue(b, c, s1), effect(C.StreamSynchronize, s0), effect(C.StreamSynchronize, s1)]
    assert check(params, prefix + suffix).hazards
    assert check(params, prefix + [effect(C.StreamWaitEvent, s1, e1)] + suffix).hazards
    assert not check(params, prefix + [effect(C.StreamWaitEvent, s1, e0)] + suffix).hazards


def test_physical_alias_reuse_waits_for_remote_read_completion():
    params, enqueue = protocol()
    a, b, c, s0, s1, e0, *_ = params
    prefix = [enqueue(a, b, s0), effect(C.EventRecord, e0, s0)]
    suffix = [enqueue(b, c, s1), effect(C.StreamSynchronize, s0), effect(C.StreamSynchronize, s1)]
    regions = {"a": analysis.BufferRegion("slot"), "c": analysis.BufferRegion("slot", 8)}
    assert any(h.kind == "overwrite_live_value" for h in check(params, prefix + suffix, regions=regions).hazards)
    assert not check(params, prefix + [effect(C.StreamWaitEvent, s1, e0)] + suffix, regions=regions).hazards


def test_nested_group_joins_streams_without_sequencing_group_members():
    params, enqueue = protocol()
    a, b, c, s0, s1, *_ = params
    grouped = [effect(C.CommGroupStart), enqueue(a, b, s0), effect(C.CommGroupStart),
               enqueue(a, c, s1), effect(C.CommGroupEnd), effect(C.CommGroupEnd)]
    assert not check(params, grouped + [effect(C.StreamSynchronize, s0)]).hazards
    grouped[3] = enqueue(b, c, s1)
    assert check(params, grouped + [effect(C.StreamSynchronize, s0)]).hazards
    with pytest.raises(ValueError, match="outermost"):
        check(params, [effect(C.CommGroupStart), effect(C.StreamSynchronize, s0)])


def test_group_preserves_fifo_for_operations_on_the_same_stream():
    params, enqueue = protocol()
    a, b, c, s0, s1, *_ = params
    d, e, f = (expr.Var(name=name, type=a.type) for name in ("d", "e", "f"))
    operations = [enqueue(a, b, s0), enqueue(c, d, s0), enqueue(e, f, s0)]
    report = check(params + (d, e, f), [
        effect(C.CommGroupStart), *operations, effect(C.CommGroupEnd),
        effect(C.StreamSynchronize, s0),
    ])
    assert analysis.overlap_candidates(report) == ()
    with pytest.raises(ValueError, match="must execute between"):
        analysis.region_boundary(report, ("comm.1", "comm.3"))
    report = check(params + (d,), [
        enqueue(a, b, s0), effect(C.CommGroupStart), enqueue(a, c, s0), enqueue(b, d, s1),
        effect(C.CommGroupEnd), effect(C.StreamSynchronize, s0), effect(C.StreamSynchronize, s1),
    ])
    assert not report.hazards


def test_exported_event_covers_only_its_generation():
    params, enqueue = protocol()
    a, b, _, s0, _, e0, *_ = params
    assert check(params, [enqueue(a, b, s0), effect(C.EventRecord, e0, s0), effect(C.ExportEvent, e0)]).exported == ("e0",)
    with pytest.raises(ValueError, match="pending"):
        check(params, [effect(C.EventRecord, e0, s0), enqueue(a, b, s0), effect(C.ExportEvent, e0)])
    with pytest.raises(ValueError, match="recorded again"):
        check(params, [effect(C.EventRecord, e0, s0), effect(C.ExportEvent, e0), enqueue(a, b, s0), effect(C.EventRecord, e0, s0)])


def test_static_loop_carries_are_checked_across_iterations():
    params, enqueue = protocol()
    a, b, _, s0, s1, *_ = params
    scalar = types.TensorType.from_scalar(expr.DType.i32)
    def constant(n):
        return expr.Constant(type=scalar, value=n)
    loop = core.For(expr.Var(name="i", type=scalar), constant(0), constant(3), constant(1),
                    core.Sequential((enqueue(a, b, s0), enqueue(b, a, s1))))
    assert check(params, [loop, effect(C.StreamSynchronize, s0), effect(C.StreamSynchronize, s1)]).hazards
    loop.body = core.Sequential((enqueue(a, b, s0), enqueue(b, a, s0)))
    assert not check(params, [loop, effect(C.StreamSynchronize, s0)]).hazards


def test_runtime_metadata_wait_and_group_boundary():
    params, enqueue = protocol()
    a, b, c, s0, s1, e0, _, communicator = params
    counts = expr.Var(name="counts", type=types.TensorType.from_shape((2, 2), dtype=expr.DType.i64))
    op = sharding.Exchange(participants=MESH, context="data", receive_capacity=4)
    dynamic = core.Evaluate(C.CommEnqueue(operation=op), (a, b, communicator, s0, counts))
    assert not check((*params, counts), [dynamic, effect(C.StreamSynchronize, s0)]).hazards
    with pytest.raises(ValueError, match="outside an open"):
        check((*params, counts), [effect(C.CommGroupStart), dynamic, effect(C.CommGroupEnd), effect(C.StreamSynchronize, s0)])
    wrong = expr.Var(name="wrong", type=types.CommunicationHandleType("communicator", "other", MESH))
    with pytest.raises(expr.VerifyError, match="identity"):
        C.check_enqueue(C.CommEnqueue(operation=op), (a, b, wrong, s0, counts))


def test_same_buffer_inplace_and_malformed_happens_before():
    params, enqueue = protocol()
    a, _, _, s0, *_ = params
    assert not check(params, [enqueue(a, a, s0), effect(C.StreamSynchronize, s0)]).hazards
    value = analysis.BufferLifetime("a", "allocation", 0, 4, analysis.BufferAccess("producer", 0, 1))
    with pytest.raises(ValueError, match="strict transitive"):
        analysis.check_buffer_lifetimes((value,), predecessors={"producer": frozenset({"producer"})})
