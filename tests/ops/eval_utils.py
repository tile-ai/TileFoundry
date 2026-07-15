"""Shared evaluator entry for ops tests.

Mirror of ``typeinfer_utils`` for value semantics: build a real
``Call(target=op, args=...)`` wrapped in a ``Function``, run it through
``tilefoundry.evaluator.evaluate`` on concrete inputs, and compare to a torch
reference. An op test file declares a list of ``EvalCase`` and runs each
through ``run_eval_case``; only the per-op coverage table varies.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_DTYPE_OF = {
    torch.float32: DType.f32,
    torch.float16: DType.f16,
    torch.bfloat16: DType.bf16,
    torch.int32: DType.i32,
    torch.int64: DType.i64,
    torch.bool: DType.bool,
}


def _ttype(t: torch.Tensor, storage: str = "gmem") -> TensorType:
    return TensorType(shape=tuple(t.shape), dtype=_DTYPE_OF[t.dtype], layout=None, storage=storage)


@dataclass(frozen=True)
class EvalCase:
    """One declarative value case: apply ``op`` to concrete ``inputs`` and
    expect ``expected`` (a torch tensor) within tolerance."""

    name: str
    op: object
    inputs: tuple[torch.Tensor, ...]
    expected: torch.Tensor
    atol: float = 1e-5
    rtol: float = 1e-5
    storages: tuple[str, ...] = field(default=())


def run_eval_case(case: EvalCase) -> None:
    """Run one ``EvalCase``: build the op's Function, evaluate on CPU, and
    assert the result matches ``expected``."""
    storages = case.storages or ("gmem",) * len(case.inputs)
    params = tuple(
        Var(type=_ttype(t, s), name=f"x{i}")
        for i, (t, s) in enumerate(zip(case.inputs, storages))
    )
    call = Call(type=params[0].type, target=case.op, args=params)
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    from tilefoundry.ir.hir.function import Function  # noqa: PLC0415 — avoid IR import cycle

    fn = Function.build(
        name="eval_case", params=params, body=call, return_type=result_type
    )
    out = evaluate(fn, *case.inputs, device="cpu")
    torch.testing.assert_close(
        out.float(), case.expected.float(), atol=case.atol, rtol=case.rtol
    )
