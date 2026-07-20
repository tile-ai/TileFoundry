"""CacheUpdate value oracle + typeinfer: same-shape functional cache write."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case, tensor_type_of
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor


def _ref(cache, cur_pos, s, new):
    out = cache.clone()
    out[:, cur_pos : cur_pos + s] = new[:, :s]
    return out


def _i32(v):
    return torch.tensor([v], dtype=torch.int32)


@pytest.mark.parametrize(
    "cur_pos,s",
    [(5, 4), (7, 2), (0, 3)],
    ids=["write_full_at_offset", "write_partial", "write_at_zero"],
)
def test_cache_update_evaluate(cur_pos, s):
    """Functional KV-cache write: the first ``s`` rows of ``new`` scatter into
    ``cache`` at ``cur_pos``; ``s`` < S_CAP leaves the rest unchanged."""
    torch.manual_seed(0)
    cache = torch.randn(1, 16, 4, 8)
    new = torch.randn(1, 4, 4, 8)
    run_eval_case(
        EvalCase(
            "",
            CacheUpdate(),
            (cache, _i32(cur_pos), _i32(s), new),
            _ref(cache, cur_pos, s, new),
        )
    )


TYPEINFER_CASES = [
    TypeInferCase(
        "output_same_shape_as_cache",
        CacheUpdate(),
        (
            make_tensor_type((1, 16, 4, 8), DType.bf16),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 4, 4, 8), DType.bf16),
        ),
        make_tensor_type((1, 16, 4, 8), DType.bf16),
    ),
    TypeInferCase(
        "s_cap_exceeds_capacity",
        CacheUpdate(),
        (
            make_tensor_type((1, 4, 4, 8), DType.bf16),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 8, 4, 8), DType.bf16),
        ),
        ExpectedError(match="exceeds cache capacity"),
    ),
    TypeInferCase(
        "cur_pos_not_i32",
        CacheUpdate(),
        (
            make_tensor_type((1, 16, 4, 8), DType.bf16),
            make_tensor_type((1,), DType.f32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 4, 4, 8), DType.bf16),
        ),
        ExpectedError(match="cur_pos must be an i32 scalar"),
    ),
    TypeInferCase(
        "kv_heads_mismatch",
        CacheUpdate(),
        (
            make_tensor_type((1, 16, 4, 8), DType.bf16),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 4, 2, 8), DType.bf16),
        ),
        ExpectedError(match="kv_heads mismatch"),
    ),
    TypeInferCase(
        "cur_pos_not_scalar",
        CacheUpdate(),
        (
            make_tensor_type((1, 16, 4, 8), DType.bf16),
            make_tensor_type((2,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 4, 4, 8), DType.bf16),
        ),
        ExpectedError(match="must be a scalar"),
    ),
    TypeInferCase(
        "partial_cache_matching_new_ok",
        CacheUpdate(),
        (
            make_shard_tensor_type(
                (1, 16, 4, 8), DType.bf16, mesh=make_mesh((4,)), attrs=(Partial("sum"),)
            ),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_shard_tensor_type(
                (1, 4, 4, 8), DType.bf16, mesh=make_mesh((4,)), attrs=(Partial("sum"),)
            ),
        ),
        make_shard_tensor_type(
            (1, 16, 4, 8), DType.bf16, mesh=make_mesh((4,)), attrs=(Partial("sum"),)
        ),
    ),
    TypeInferCase(
        "partial_cache_plain_new_rejected",
        CacheUpdate(),
        (
            make_shard_tensor_type(
                (1, 16, 4, 8), DType.bf16, mesh=make_mesh((4,)), attrs=(Partial("sum"),)
            ),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1, 4, 4, 8), DType.bf16),
        ),
        ExpectedError(match="cache carries a Partial"),
    ),
    TypeInferCase(
        "complete_cache_partial_new_rejected",
        CacheUpdate(),
        (
            make_tensor_type((1, 16, 4, 8), DType.bf16),
            make_tensor_type((1,), DType.i32),
            make_tensor_type((1,), DType.i32),
            make_shard_tensor_type(
                (1, 4, 4, 8), DType.bf16, mesh=make_mesh((4,)), attrs=(Partial("sum"),)
            ),
        ),
        ExpectedError(match="new carries Partial"),
    ),
]


@pytest.mark.parametrize("case", TYPEINFER_CASES, ids=lambda c: c.name)
def test_cache_update_typeinfer(case):
    run_typeinfer_case(case)


def _run(cur_pos, s):
    """Build + evaluate a cache_update call at the given runtime cur_pos / s."""
    torch.manual_seed(0)
    cache = torch.randn(1, 16, 4, 8)
    new = torch.randn(1, 4, 4, 8)
    inputs = (cache, _i32(cur_pos), _i32(s), new)

    params = tuple(Var(type=tensor_type_of(t), name=f"x{i}") for i, t in enumerate(inputs))
    call = Call(type=params[0].type, target=CacheUpdate(), args=params)
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="cu", params=params, body=call, return_type=result_type)
    return evaluate(fn, *inputs, device="cpu")


@pytest.mark.parametrize(
    "cur_pos,s,match",
    [
        (-1, 1, "must be >= 0"),
        (5, 0, "1 <= s"),
        (5, 5, "1 <= s"),  # s exceeds S_CAP=4
        (14, 4, "exceeds cache capacity"),  # cur_pos + s > 16
    ],
    ids=["neg_cur_pos", "s_zero", "s_over_cap", "cur_pos_plus_s_over_capacity"],
)
def test_cache_update_evaluate_rejects_bad_runtime(cur_pos, s, match):
    # ``pytest.raises`` as a context manager asserts the body raises: the test
    # FAILS if ``_run`` does not raise an ``EvalError`` matching ``match``.
    with pytest.raises(EvalError, match=match):
        _run(cur_pos, s)
