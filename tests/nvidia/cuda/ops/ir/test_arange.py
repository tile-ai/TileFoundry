"""Arange's typed coordinate contract."""

from __future__ import annotations

from dataclasses import replace

import torch

from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.specialize import residual_dims
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.isl_interop import normalize_dim
from tilefoundry_nvidia import CudaTarget

_N = DimVar("arange_n", 1, 16)


@func
def _symbolic_arange(x: Tensor[(_N,), "f32"]):
    return tf.arange(Tensor[(ceildiv(_N - 1, 2),), "i32"], start=1, step=2)


def test_arange_symbolic_extent_resolves_from_runtime_shape():
    call = _symbolic_arange.body
    assert isinstance(call, Call) and isinstance(call.target, Arange)
    assert call.type.shape == (normalize_dim(ceildiv(_N - 1, 2)),)
    assert call.type.dtype == DType.i32

    actual = evaluate(_symbolic_arange, torch.zeros(8))
    torch.testing.assert_close(actual, torch.tensor([1, 3, 5, 7], dtype=torch.int32))

    concrete = check_program(replace(Module("arange", (_symbolic_arange,), _symbolic_arange.name), target=CudaTarget("nvidia.h200_sxm")), _symbolic_arange, dims={"arange_n": 8})[1]
    assert residual_dims(concrete) == ()
    assert isinstance(concrete.body.target, Arange)
    assert concrete.body.target.type.shape == (4,)
