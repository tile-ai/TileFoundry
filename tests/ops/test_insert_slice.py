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
    # A rank-0 (scalar) offset is not the A surface — offsets is a vector.
    TypeInferCase(
        "offsets_scalar_rejected",
        _OP,
        (ten((8,), _F), ten((3,), _F), ten((), _I)),
        ExpectedError("offsets must be a rank-1 vector", exc=TypeError),
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


@pytest.mark.parametrize(
    "dst,upd,start",
    [
        (torch.zeros(8), torch.tensor([1.0, 2.0, 3.0]), -1),  # negative start
        (torch.zeros(8), torch.tensor([1.0, 2.0, 3.0]), 6),   # window runs past end
    ],
    ids=["negative_start", "past_end"],
)
def test_insert_slice_eval_out_of_bounds(dst, upd, start):
    """A runtime offset that puts the window out of dst's bounds is rejected by
    the eval guard (the static typeinfer check cannot see a runtime offset)."""
    from dataclasses import replace  # noqa: PLC0415

    from tilefoundry.evaluator import evaluate  # noqa: PLC0415
    from tilefoundry.ir.core import Call, Var  # noqa: PLC0415
    from tilefoundry.ir.hir.function import Function  # noqa: PLC0415
    from tilefoundry.ir.types import TensorType  # noqa: PLC0415
    from tilefoundry.visitor_registry.contexts import TypeInferContext  # noqa: PLC0415
    from tilefoundry.visitor_registry.visitors import TypeInferVisitor  # noqa: PLC0415

    offs = torch.tensor([start], dtype=torch.int32)
    inputs = (dst, upd, offs)
    dtypes = (_F, _F, _I)
    params = tuple(
        Var(type=TensorType(shape=tuple(t.shape), dtype=d, layout=None, storage="gmem"), name=f"x{i}")
        for i, (t, d) in enumerate(zip(inputs, dtypes))
    )
    call = Call(type=params[0].type, target=_OP, args=params)
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="eval_oob", params=params, body=call, return_type=result_type)
    with pytest.raises(ValueError, match="out of bounds"):
        evaluate(fn, *inputs, device="cpu")
