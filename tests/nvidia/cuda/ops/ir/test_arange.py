"""Arange's typed coordinate contract."""

from __future__ import annotations

from pathlib import Path

import torch

from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.specialize import residual_dims
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.isl_interop import normalize_dim

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_N = DimVar("arange_n", 1, 16)


def test_arange_symbolic_extent_resolves_from_runtime_shape():
    SymbolicArange = load_authored_ir(f"{OPS / 'symbolic_arange.py'}:SymbolicArange")
    _symbolic_arange = SymbolicArange.entry_function()
    call = _symbolic_arange.body
    assert isinstance(call, Call) and isinstance(call.target, Arange)
    assert call.type.shape == (normalize_dim(ceildiv(_N - 1, 2)),)
    assert call.type.dtype == DType.i32

    actual = evaluate(_symbolic_arange, torch.zeros(8))
    torch.testing.assert_close(actual, torch.tensor([1, 3, 5, 7], dtype=torch.int32))

    concrete = check_program(SymbolicArange, _symbolic_arange, dims={"arange_n": 8})[1]
    assert residual_dims(concrete) == ()
    assert isinstance(concrete.body.target, Arange)
    assert concrete.body.target.type.shape == (4,)
