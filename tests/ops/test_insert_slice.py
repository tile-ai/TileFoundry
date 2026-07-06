"""HIR insert_slice (dynamic-update-slice) typeinfer + eval.

``insert_slice(dst, update, offsets)`` returns ``dst`` with ``update`` written
into the window starting at ``offsets`` (one i32 start per dim). This milestone
implements the 1-D case; higher rank shares the surface and is rejected at
typeinfer.
"""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types import DType
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
    ten,
)

_F = DType.f32
_I = DType.i32
_OP = InsertSlice()

CASES = [
    # 1-D window: returns dst's type unchanged.
    TypeInferCase(
        "returns_dst_type",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((1,), _I)),
        ten((8,), _F),
    ),
    # A full-width update (same extent as dst) is in bounds.
    TypeInferCase(
        "full_width_update_ok",
        _OP,
        (ten((8,), _F), ten((8,), _F), ten((1,), _I)),
        ten((8,), _F),
    ),
    # update rank must equal dst rank.
    TypeInferCase(
        "rank_mismatch_rejected",
        _OP,
        (ten((8,), _F), ten((2, 4), _F), ten((2,), _I)),
        ExpectedError("update rank .* must equal dst rank", exc=TypeError),
    ),
    # N-D not implemented yet (same surface).
    TypeInferCase(
        "nd_not_implemented",
        _OP,
        (ten((4, 8), _F), ten((1, 8), _F), ten((2,), _I)),
        ExpectedError("only the 1-D case", exc=NotImplementedError),
    ),
    # offsets length must equal dst rank.
    TypeInferCase(
        "offsets_length_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((2,), _I)),
        ExpectedError("offsets length .* must equal dst rank", exc=TypeError),
    ),
    # offsets must be i32.
    TypeInferCase(
        "offsets_dtype_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((1,), _F)),
        ExpectedError("offsets must be i32", exc=TypeError),
    ),
    # dst / update dtype must match.
    TypeInferCase(
        "dtype_mismatch_rejected",
        _OP,
        (ten((8,), _F), ten((3,), DType.bf16), ten((1,), _I)),
        ExpectedError("dst/update dtype mismatch", exc=TypeError),
    ),
    # A statically over-long update is rejected.
    TypeInferCase(
        "static_overlong_update_rejected",
        _OP,
        (ten((8,), _F), ten((10,), _F), ten((1,), _I)),
        ExpectedError("exceeds dst extent", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_insert_slice_typeinfer(case):
    run_typeinfer_case(case)


def _ref(dst, upd, start):
    out = dst.clone()
    out[start:start + upd.shape[0]] = upd
    return out


@pytest.mark.parametrize(
    "dst,upd,start",
    [
        (torch.zeros(8), torch.tensor([1.0, 2.0, 3.0]), 2),   # interior window
        (torch.arange(6.0), torch.tensor([9.0]), 0),          # single element at 0
        (torch.arange(6.0), torch.tensor([7.0]), 5),          # single element at end
        (torch.zeros(4), torch.arange(1.0, 5.0), 0),          # full overwrite
    ],
    ids=["interior", "elem_start", "elem_end", "full"],
)
def test_insert_slice_eval(dst, upd, start):
    offs = torch.tensor([start], dtype=torch.int32)
    run_eval_case(EvalCase("", _OP, (dst, upd, offs), _ref(dst, upd, start), atol=0.0))
