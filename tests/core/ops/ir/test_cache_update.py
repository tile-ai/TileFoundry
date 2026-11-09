"""CacheUpdate's value oracle, runtime bounds and Partial cache contract.

No corpus model calls ``cache_update``, so the oracle stays here; the GPU witness
for the same op is the decode step in ``test_insert_slice.py``. The bounds are
runtime, not static: ``cur_pos`` and ``s`` arrive as scalar tensors, so typeinfer
cannot see them and the guard has to be in the evaluator.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.core.ops.ir.op_case import Cost, ExpectedError, OpCase, op_case_test, tensor_type_of
from tilefoundry.dsl import *
from tilefoundry.interfaces import cost, evaluator, typeinfer
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import tensor


def _ref(cache, cur_pos, s, new):
    out = cache.clone()
    out[:, cur_pos : cur_pos + s] = new[:, :s]
    return out


def _i32(v):
    return torch.tensor([v], dtype=torch.int32)


def _make_cache_update_evaluate():
    torch.manual_seed(0)
    cache = torch.randn(1, 16, 4, 8)
    new = torch.randn(1, 4, 4, 8)
    return OpCase(
        "partial_cache_write",
        tensor.CacheUpdate(),
        (cache, _i32(7), _i32(2), new),
        value=_ref(cache, 7, 2, new),
    )


def _run(cur_pos, s):
    """Build + evaluate a cache_update call at the given runtime cur_pos / s."""
    torch.manual_seed(0)
    cache = torch.randn(1, 16, 4, 8)
    new = torch.randn(1, 4, 4, 8)
    inputs = (cache, _i32(cur_pos), _i32(s), new)
    params = tuple((expr.Var(type=tensor_type_of(t), name=f"x{i}") for i, t in enumerate(inputs)))
    call = expr.Call(type=params[0].type, target=tensor.CacheUpdate(), args=params)
    result_type = typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    call = replace(call, type=result_type)
    fn = core.Function.from_parts(name="cu", params=params, body=call, return_type=result_type)
    return evaluator.evaluate(fn, *inputs)


@pytest.mark.parametrize(
    "cur_pos,s,match",
    [(-1, 1, "must be >= 0"), (5, 5, "1 <= s"), (14, 4, "exceeds cache capacity")],
    ids=["neg_cur_pos", "s_over_cap", "cur_pos_plus_s_over_capacity"],
)
def test_cache_update_rejects_bad_runtime(cur_pos, s, match):
    with pytest.raises(evaluator.EvalError, match=match):
        _run(cur_pos, s)


_NEW_SHAPE = (2, 4, 4, 8)
_SCALAR_TYPE = TensorType.from_shape((1,), DType.i32)
_WINDOW_BYTES = 2 * 4 * 4 * 8 * 2
_GLOBAL_TRAFFIC = (
    cost.TrafficBytes(),
    cost.TrafficBytes(read=4),
    cost.TrafficBytes(read=4),
    cost.TrafficBytes(read=_WINDOW_BYTES),
    cost.TrafficBytes(write=_WINDOW_BYTES),
)


def _make_cache_update_cost_is_independent_of_cache_length(cache_len) -> OpCase:
    return OpCase(
        f"cache_len_{cache_len}",
        tensor.CacheUpdate(),
        (
            TensorType.from_shape((2, cache_len, 4, 8), DType.bf16),
            _SCALAR_TYPE,
            _SCALAR_TYPE,
            TensorType.from_shape(_NEW_SHAPE, DType.bf16),
        ),
        cost=Cost(traffic=_GLOBAL_TRAFFIC),
    )


CASES = (
    _make_cache_update_evaluate(),
    *(_make_cache_update_cost_is_independent_of_cache_length(_param) for _param in (16, 1024)),
    OpCase(
        "partial_cache_plain_new_rejected",
        tensor.CacheUpdate(),
        (
            make_shard_tensor_type(
                (1, 16, 4, 8),
                DType.bf16,
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(types.Partial("sum"),),
            ),
            TensorType.from_shape((1,), DType.i32),
            TensorType.from_shape((1,), DType.i32),
            TensorType.from_shape((1, 4, 4, 8), DType.bf16),
        ),
        error=ExpectedError(match="cache carries a Partial"),
    ),
    OpCase(
        "complete_cache_partial_new_rejected",
        tensor.CacheUpdate(),
        (
            TensorType.from_shape((1, 16, 4, 8), DType.bf16),
            TensorType.from_shape((1,), DType.i32),
            TensorType.from_shape((1,), DType.i32),
            make_shard_tensor_type(
                (1, 4, 4, 8),
                DType.bf16,
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(types.Partial("sum"),),
            ),
        ),
        error=ExpectedError(match="new carries Partial"),
    ),
)
test_op_case = op_case_test(CASES)
