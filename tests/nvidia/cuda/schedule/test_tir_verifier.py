"""Scheduling tir verifier checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.nvidia.cuda.schedule._support import (
    _finalized,
)
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import (
    Call,
    Constant,
    Var,
)
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.types import (
    DType,
    TensorType,
)
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
from tilefoundry_nvidia import CudaTarget


@pytest.mark.parametrize(
    "wrong,refused",
    [
        ("count", "arg count"),
        ("type", r"arg\[0\] type"),
        ("recursion", "recursion not allowed"),
    ],
)
def test_verifier_rejects_malformed_tir_calls(wrong, refused):
    param = Var("x", type=TensorType.scalar(DType.f32))
    callee = PrimFunction("callee", (param,), Sequential(()), target=CudaTarget("nvidia.h200_sxm"))
    args = () if wrong == "count" else (Constant(0, type=TensorType.scalar(DType.i32)),)
    caller = PrimFunction("caller", (), Sequential((Evaluate(callee, args),)), target=callee.target)
    if wrong == "recursion":
        caller.body = Sequential((Evaluate(caller, ()),))
    with pytest.raises(VerifyError, match=refused):
        verify_prim_function(caller, module_fns=(caller, callee))


def test_verifier_rejects_noninteger_slot_index():
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/module_instances.py"
    lowered = _finalized(load_authored_ir(str(source)))
    slot = lowered.children["layers"]
    target = Call(
        TupleGetItem(),
        (slot, Constant(0.5, type=TensorType.scalar(DType.f32))),
        type=slot.type.fields[0],
    )
    root = replace(lowered.entry_function(), body=Sequential((Evaluate(target, ()),)))
    functions = tuple(
        root if function is lowered.entry_function() else function for function in lowered.functions
    )
    malformed = replace(lowered, functions=functions)
    with pytest.raises(VerifyError, match="index must be an integer scalar"):
        verify_module(malformed)
