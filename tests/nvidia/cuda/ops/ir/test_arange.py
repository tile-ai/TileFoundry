"""Arange's typed coordinate contract."""

from __future__ import annotations

from pathlib import Path

import torch

from tilefoundry import analysis, cli, services
from tilefoundry.dsl import *
from tilefoundry.interfaces import evaluator, valuation
from tilefoundry.ir import expr
from tilefoundry.ir.hir import tensor

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_N = DimVar("arange_n", 1, 16)


def test_arange_symbolic_extent_resolves():
    SymbolicArange = cli.load_authored_ir(f"{OPS / 'symbolic_arange.py'}:SymbolicArange")
    _symbolic_arange = SymbolicArange.entry_function()
    call = _symbolic_arange.body
    assert isinstance(call, expr.Call) and isinstance(call.target, tensor.Arange)
    assert call.type.shape == (services.normalize_dim(ceildiv(_N - 1, 2)),)
    assert call.type.dtype == DType.i32

    actual = evaluator.evaluate(_symbolic_arange, torch.zeros(8))
    torch.testing.assert_close(actual, torch.tensor([1, 3, 5, 7], dtype=torch.int32))

    concrete = analysis.check_program(SymbolicArange, _symbolic_arange, dims={"arange_n": 8})[1]
    assert valuation.residual_dims(concrete) == ()
    assert isinstance(concrete.body.target, tensor.Arange)
    assert concrete.body.target.type.shape == (4,)
