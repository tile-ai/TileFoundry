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

from tests.evaluator.eval_utils import EvalCase, run_eval_case, tensor_type_of
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry import func, module
from tilefoundry.analysis import ComputeCostMetadata, TrafficMetadata
from tilefoundry.analysis.api import analyze
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Var, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.types import (
    DType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard import Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor


def _ref(cache, cur_pos, s, new):
    out = cache.clone()
    out[:, cur_pos : cur_pos + s] = new[:, :s]
    return out


def _i32(v):
    return torch.tensor([v], dtype=torch.int32)


def test_cache_update_evaluate():
    """Functional KV-cache write.

    Functional KV-cache write: the first ``s`` rows of ``new`` scatter into
    ``cache`` at ``cur_pos``; ``s`` < S_CAP leaves the rest unchanged. The
    partial-width write is the informative one -- a full-width write cannot show
    that the untouched tail survives.
    """
    torch.manual_seed(0)
    cache = torch.randn(1, 16, 4, 8)
    new = torch.randn(1, 4, 4, 8)
    run_eval_case(
        EvalCase("", CacheUpdate(), (cache, _i32(7), _i32(2), new), _ref(cache, 7, 2, new))
    )


TYPEINFER_CASES = [
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
    result_type = TypeInferVisitor().visit(call, TypeInferContext())
    call = replace(call, type=result_type)
    fn = Function.build(name="cu", params=params, body=call, return_type=result_type)
    return evaluate(fn, *inputs)


@pytest.mark.parametrize(
    "cur_pos,s,match",
    [
        (-1, 1, "must be >= 0"),
        (5, 5, "1 <= s"),
        (14, 4, "exceeds cache capacity"),
    ],
    ids=["neg_cur_pos", "s_over_cap", "cur_pos_plus_s_over_capacity"],
)
def test_cache_update_evaluate_rejects_bad_runtime(cur_pos, s, match):

    with pytest.raises(EvalError, match=match):
        _run(cur_pos, s)


_CACHE_SHAPE = (2, 64, 4, 8)
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


@pytest.mark.parametrize("cache_len", (16, 1024))
def test_cache_update_cost_is_independent_of_cache_length(cache_len) -> None:
    run_cost_case(
        CostCase(
            f"cache_len_{cache_len}",
            CacheUpdate(),
            (
                make_tensor_type((2, cache_len, 4, 8), DType.bf16),
                _SCALAR_TYPE,
                _SCALAR_TYPE,
                make_tensor_type(_NEW_SHAPE, DType.bf16),
            ),
            traffic=_GLOBAL_TRAFFIC,
        )
    )


_CTA = Topology("cta", 2)
_CTA_MESH = make_mesh((2,), topology=_CTA)


@module(
    entry="append",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(_CTA,),
)
class _KVCacheAppend:
    @func
    def append(
        cache: Tensor[_CACHE_SHAPE, "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
        new: Tensor[_NEW_SHAPE, "bf16"],
    ):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as cta:
            local_cache = tf.reshard(cache, (2 @ cta.tile, 64, 4, 8), "gmem")
            local_new = tf.reshard(new, (2 @ cta.tile, 4, 4, 8), "gmem")
            return tf.cache_update(local_cache, cur_pos, s, local_new)


def test_cache_update_function_analyzes_program_and_cta_cost() -> None:
    """One crossing is the first legal window, in both windows of the question.

    How many rows this writes is a runtime value, so one occurrence is the first
    legal binding of it: one row. How many such crossings the program performs
    is the footprint family's question, and raising a single crossing to cover
    them would be answering it here.
    """
    entry = _KVCacheAppend.entry_function()
    update = next(
        expr
        for expr in collect_exprs(entry.body)
        if isinstance(expr, Call) and isinstance(expr.target, CacheUpdate)
    )
    selected_types = {id(arg): arg.type for arg in update.args}
    program_ctx = CostContext(
        selected_types=selected_types, selected_output_type=update.type
    )
    cta_ctx = CostContext(
        selected_types=selected_types,
        selected_output_type=update.type,
        level="cta",
        topologies=(_CTA,),
    )
    assert cta_ctx.local_type_of(update.args[3]).shape == (1, 4, 4, 8)
    assert CostEvaluator().visit_Call(update, program_ctx).traffic == _GLOBAL_TRAFFIC
    assert CostEvaluator().visit_Call(update, cta_ctx).traffic == (
        TrafficBytes(),
        TrafficBytes(read=4),
        TrafficBytes(read=4),
        TrafficBytes(read=_WINDOW_BYTES // 2),
        TrafficBytes(write=_WINDOW_BYTES // 2),
    )

    result = analyze(_KVCacheAppend, entry, analysis=("compute-cost", "memory"), level="cta")
    analysed_update = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, CacheUpdate)
    )
    record = get_metadata(analysed_update, ComputeCostMetadata)
    moved = get_metadata(analysed_update, TrafficMetadata)
    assert result.level == "cta"
    assert record is not None
    assert record.flops == record.flops_per_unit == ()
    row_bytes = _WINDOW_BYTES // 4
    assert moved.operands == (
        TrafficBytes(),
        TrafficBytes(read=4),
        TrafficBytes(read=4),
        TrafficBytes(read=row_bytes),
        TrafficBytes(write=row_bytes),
    )
