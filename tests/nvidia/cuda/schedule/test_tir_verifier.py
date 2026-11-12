"""Scheduling tir verifier checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.nvidia.cuda.schedule._support import (
    _finalized,
)
from tilefoundry import cli, verifier
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, types
from tilefoundry_nvidia import cuda


@pytest.mark.parametrize(
    "wrong,refused",
    [
        ("count", "arg count"),
        ("type", r"arg\[0\] type"),
        ("recursion", "recursion not allowed"),
    ],
)
def test_verifier_rejects_malformed_tir_calls(wrong, refused):
    param = expr.Var("x", type=types.TensorType.from_scalar(expr.DType.f32))
    callee = core.PrimFunction("callee", (param,), core.Sequential(()), target=cuda.Target("nvidia.h200_sxm"))
    args = () if wrong == "count" else (expr.Constant(0, type=types.TensorType.from_scalar(expr.DType.i32)),)
    named = "caller" if wrong == "recursion" else "callee"
    call = core.Evaluate(core.FunctionRef((), named), () if wrong == "recursion" else args)
    caller = core.PrimFunction("caller", (), core.Sequential((call,)), target=callee.target)
    module = core.Module("Calls", (callee, caller))
    with pytest.raises(expr.VerifyError, match=refused):
        verifier.verify_prim_function(caller, frame=valuation.Context(module, caller))


def test_verifier_rejects_noninteger_slot_index():
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/module_instances.py"
    lowered = _finalized(cli.load_authored_ir(str(source)))
    index = expr.Constant(0.5, type=types.TensorType.from_scalar(expr.DType.f32))
    target = core.FunctionRef(("layers", None), None)
    root = replace(lowered.lookup("gemm"), body=core.Sequential((core.Evaluate(target, (index,)),)))
    functions = tuple(
        root if function is lowered.lookup("gemm") else function for function in lowered.functions
    )
    malformed = replace(lowered, functions=functions)
    with pytest.raises(expr.VerifyError, match="index must be an integer scalar"):
        verifier.verify_module(malformed)
