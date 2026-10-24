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
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.types.shard_layout import Partial
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor


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
        CacheUpdate(),
        (cache, _i32(7), _i32(2), new),
        value=_ref(cache, 7, 2, new),
    )


def _run(cur_pos, s):
    """Build + evaluate a cache_update call at the given runtime cur_pos / s."""
    torch.manual_seed(0)
    cache = torch.randn(1, 16, 4, 8)
    new = torch.randn(1, 4, 4, 8)
    inputs = (cache, _i32(cur_pos), _i32(s), new)
    params = tuple((Var(type=tensor_type_of(t), name=f"x{i}") for i, t in enumerate(inputs)))
    call = Call(type=params[0].type, target=CacheUpdate(), args=params)
    result_type = TypeInferVisitor().visit(call, TypeInferContext())
    call = replace(call, type=result_type)
    fn = Function.build(name="cu", params=params, body=call, return_type=result_type)
    return evaluate(fn, *inputs)


@pytest.mark.parametrize(
    "cur_pos,s,match",
    [(-1, 1, "must be >= 0"), (5, 5, "1 <= s"), (14, 4, "exceeds cache capacity")],
    ids=["neg_cur_pos", "s_over_cap", "cur_pos_plus_s_over_capacity"],
)
def test_cache_update_evaluate_rejects_bad_runtime(cur_pos, s, match):
    with pytest.raises(EvalError, match=match):
        _run(cur_pos, s)


_NEW_SHAPE = (2, 4, 4, 8)
_SCALAR_TYPE = make_tensor_type((1,), DType.i32)
_WINDOW_BYTES = 2 * 4 * 4 * 8 * 2
_GLOBAL_TRAFFIC = (
    TrafficBytes(),
    TrafficBytes(read=4),
    TrafficBytes(read=4),
    TrafficBytes(read=_WINDOW_BYTES),
    TrafficBytes(write=_WINDOW_BYTES),
)


def _make_cache_update_cost_is_independent_of_cache_length(cache_len) -> OpCase:
    return OpCase(
        f"cache_len_{cache_len}",
        CacheUpdate(),
        (
            make_tensor_type((2, cache_len, 4, 8), DType.bf16),
            _SCALAR_TYPE,
            _SCALAR_TYPE,
            make_tensor_type(_NEW_SHAPE, DType.bf16),
        ),
        cost=Cost(traffic=_GLOBAL_TRAFFIC),
    )


CASES = (
    _make_cache_update_evaluate(),
    *(_make_cache_update_cost_is_independent_of_cache_length(_param) for _param in (16, 1024)),
    OpCase(
        "partial_cache_plain_new_rejected",
        CacheUpdate(),
        (
            make_shard_tensor_type(
                (1, 16, 4, 8),
                DType.bf16,
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Partial("sum"),),
            ),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 4, 4, 8), DType.bf16),
        ),
        error=ExpectedError(match="cache carries a Partial"),
    ),
    OpCase(
        "complete_cache_partial_new_rejected",
        CacheUpdate(),
        (
            make_tensor_type((1, 16, 4, 8), DType.bf16),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_shard_tensor_type(
                (1, 4, 4, 8),
                DType.bf16,
                mesh=Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",)),
                attrs=(Partial("sum"),),
            ),
        ),
        error=ExpectedError(match="new carries Partial"),
    ),
)
test_op_case = op_case_test(CASES)
