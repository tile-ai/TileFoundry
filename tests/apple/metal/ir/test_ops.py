"""The measured matrix lane map and the torch semantics of the Metal Ops."""

import pytest

from tilefoundry import core, utils
from tilefoundry.interfaces import evaluator, schedule
from tilefoundry.ir import expr, pattern, types
from tilefoundry_apple.metal.ir import tir


torch = utils.torch
F32, I32, U16 = expr.DType.f32, expr.DType.i32, expr.DType.u16


def bound(shard: pattern.ShardLayout, **bindings) -> types.ShardLayout:
    """The concrete ShardLayout a declared layout constant names for *bindings*."""
    held, threads = shard.layout, shard.mesh.layout
    mesh_layout = types.Layout(
        pattern.fixed_value(threads.shape, bindings), pattern.fixed_value(threads.strides, bindings)
    )
    mesh = types.Mesh((types.Topology("thread", types.size(mesh_layout)),), mesh_layout)
    return types.ShardLayout(
        types.Layout(pattern.fixed_value(held.shape, bindings), pattern.fixed_value(held.strides, bindings)),
        shard.attrs,
        mesh,
    )


def fragment(data, dtype=F32):
    groups = data.shape[0]
    held = types.TensorType((groups, 8, 8), dtype, bound(tir.layout.FRAGMENT, groups=groups), expr.StorageKind.RMEM)
    return evaluator.TensorValue(core.ShardTensor(held, data))


def memory(data, dtype=F32):
    groups = data.shape[0]
    held = types.TensorType((groups, 8, 8), dtype, types.Layout((groups, 8, 8), (64, 8, 1)), expr.StorageKind.GMEM)
    return evaluator.TensorValue(core.ShardTensor(held, data))


def lanes(data, dtype=I32, shard=tir.layout.LANES):
    groups, width = data.shape
    held = types.TensorType(tuple(data.shape), dtype, bound(shard, groups=groups, width=width), expr.StorageKind.RMEM)
    return evaluator.TensorValue(core.ShardTensor(held, data))


def scalar(value):
    held = types.TensorType((), U16, types.Layout(()), expr.StorageKind.RMEM)
    return evaluator.TensorValue(core.ShardTensor(held, torch.tensor(value, dtype=torch.uint16)))


def evaluate(op, *args):
    handler = evaluator.registry.get(type(op), None)
    return handler(evaluator.EvaluateContext(op=op, args=args, result_type=args[0].type)).data


def test_fragment_matches_measured_lane_map():
    shard = bound(tir.layout.FRAGMENT, groups=1)
    held = core.ShardTensor(
        types.TensorType((1, 8, 8), I32, shard, expr.StorageKind.RMEM),
        torch.arange(64, dtype=torch.int32).reshape(1, 8, 8),
    )
    seen = []
    for lane in range(32):
        row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4
        col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4
        local = held.to_local(types.Placement({"thread": lane})).flatten().tolist()
        assert local == [row * 8 + col, row * 8 + col + 1], lane
        seen.extend(local)
    assert sorted(seen) == list(range(64))


def test_matrix_evaluators():
    a = (torch.arange(128.0) % 5).reshape(2, 8, 8)
    b = (torch.arange(128.0) % 3).reshape(2, 8, 8)
    c = (torch.arange(128.0) % 7).reshape(2, 8, 8)
    zeros = lambda: torch.zeros(2, 8, 8)
    assert torch.equal(evaluate(tir.MatMul(dtype=F32), fragment(zeros()), fragment(a), fragment(b)), a @ b)
    assert torch.equal(
        evaluate(tir.Mma(dtype=F32), fragment(zeros()), fragment(a), fragment(b), fragment(c)), a @ b + c
    )
    accumulator = fragment(c.clone())
    assert torch.equal(
        evaluate(tir.Mma(dtype=F32), accumulator, fragment(a), fragment(b), accumulator), a @ b + c
    )
    loaded = evaluate(tir.LdMatrix(transpose=True, dtype=F32), fragment(zeros()), memory(a))
    assert torch.equal(loaded, a.mT)
    stored = evaluate(tir.StMatrix(transpose=True, dtype=F32), memory(zeros()), fragment(a))
    assert torch.equal(stored, a.mT)


def test_simd_evaluators():
    source = torch.arange(32, dtype=torch.int32)[None]
    total = lambda kind, data: evaluate(
        tir.Reduce(kind=kind, dtype=I32),
        lanes(torch.zeros(1, 1, dtype=torch.int32), shard=tir.layout.SHARED),
        lanes(data),
    )
    out = lambda: lanes(torch.zeros(1, 32, dtype=torch.int32))
    assert total(tir.ReduceKind.SUM, source).tolist() == [[496]]
    assert total(tir.ReduceKind.MAX, source).tolist() == [[31]]
    assert total(tir.ReduceKind.SUM, source[:, :4]).tolist() == [[6]]
    running = evaluate(tir.Prefix(kind=tir.PrefixKind.SUM, inclusive=False, dtype=I32), out(), lanes(source))
    assert running[0].tolist() == [t * (t - 1) // 2 for t in range(32)]
    rotated = evaluate(tir.Shuffle(kind=tir.ShuffleKind.ROTATE_DOWN, dtype=I32), out(), lanes(source), scalar(1))
    assert rotated[0].tolist() == [(t + 1) % 32 for t in range(32)]
    broadcast = evaluate(tir.Shuffle(kind=tir.ShuffleKind.BROADCAST, dtype=I32), out(), lanes(source), scalar(5))
    assert broadcast[0].tolist() == [5] * 32
    fill = source + 100
    shifted = evaluate(
        tir.ShuffleFill(kind=tir.ShuffleKind.UP, modulo=8, dtype=I32), out(), lanes(source), lanes(fill), scalar(1)
    )
    assert shifted[0].tolist() == [fill[0, t + 7].item() if t % 8 == 0 else t - 1 for t in range(32)]
    votes = evaluate(
        tir.Ballot(),
        lanes(torch.zeros(1, 1, dtype=torch.uint32), expr.DType.u32, tir.layout.SHARED),
        lanes(source % 2 == 0, expr.DType.bool),
    )
    assert int(votes) == 0x55555555
    U32 = expr.DType.u32
    wide = torch.arange(32)[None].to(torch.uint32)
    wide[0, 7] = 0xFFFFFFF0
    out32 = lambda: lanes(torch.zeros(1, 32, dtype=torch.uint32), U32)
    largest = evaluate(
        tir.Reduce(kind=tir.ReduceKind.MAX, dtype=U32),
        lanes(torch.zeros(1, 1, dtype=torch.uint32), U32, tir.layout.SHARED),
        lanes(wide, U32),
    )
    assert int(largest) == 0xFFFFFFF0
    running = evaluate(tir.Prefix(kind=tir.PrefixKind.SUM, inclusive=True, dtype=U32), out32(), lanes(wide, U32))
    assert int(running[0, 7]) == (0xFFFFFFF0 + 21) % 2**32
    rotated = evaluate(tir.Shuffle(kind=tir.ShuffleKind.ROTATE_UP, dtype=U32), out32(), lanes(wide, U32), scalar(1))
    assert int(rotated[0, 8]) == 0xFFFFFFF0


def test_lane_selection_outside_group_is_an_error():
    source = torch.arange(32, dtype=torch.int32)[None]
    out = lanes(torch.zeros(1, 32, dtype=torch.int32))
    selector = lanes(torch.full((1, 32), 40, dtype=torch.uint16), U16)
    with pytest.raises(evaluator.EvalError, match="outside the group"):
        evaluate(tir.Shuffle(kind=tir.ShuffleKind.INDEX, dtype=I32), out, lanes(source), selector)
    with pytest.raises(evaluator.EvalError, match="outside the group"):
        evaluate(
            tir.ShuffleFill(kind=tir.ShuffleKind.UP, modulo=8, dtype=I32), out, lanes(source), lanes(source), scalar(9)
        )
    quad = lanes(source[:, :4])
    wide_fill = tir.ShuffleFill(kind=tir.ShuffleKind.UP, modulo=8, dtype=I32)
    with pytest.raises(evaluator.EvalError, match="outside the group"):
        evaluate(wide_fill, quad, quad, quad, scalar(1))
    operands = (quad.type, quad.type, quad.type, scalar(1).type)
    with pytest.raises(schedule.RefusalError, match="modulo <= width"):
        schedule.check_call(wide_fill, operands, quad.type.layout.mesh, ctx=None)
