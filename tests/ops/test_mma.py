"""HIR CUDA MMA logical value, cost, fragment, and rejection contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, infer_call, run_typeinfer_case
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.tir.cuda.nn.mma import (
    SM80_16x8x16_F32BF16BF16F32_TN,
    make_atom,
)
from tilefoundry.ir.types import (
    DType,
    TensorType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard import Broadcast, Layout, Split, make_mesh
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TrafficBytes

_BF = DType.bf16
_F32 = DType.f32
_RMEM = StorageKind.RMEM
_ATOM = make_atom(SM80_16x8x16_F32BF16BF16F32_TN)
_SM80 = Mma_SM80_16x8x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=_F32)
_WGMMA = Wgmma_SM90_64x128x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=_F32)


CASES = [
    TypeInferCase(
        "sm80_logical",
        _SM80,
        (
            make_tensor_type((16, 16), _BF, storage=_RMEM),
            make_tensor_type((16, 8), _BF, storage=_RMEM),
        ),
        make_tensor_type((16, 8), _F32, storage=_RMEM),
    ),
    TypeInferCase(
        "wgmma_logical",
        _WGMMA,
        (
            make_tensor_type((64, 16), _BF, storage=_RMEM),
            make_tensor_type((16, 128), _BF, storage=_RMEM),
        ),
        make_tensor_type((64, 128), _F32, storage=_RMEM),
    ),
    TypeInferCase(
        "wrong_a_shape",
        _SM80,
        (make_tensor_type((15, 16), _BF), make_tensor_type((16, 8), _BF)),
        ExpectedError(match=r"Mma_SM80_16x8x16: a shape"),
    ),
    TypeInferCase(
        "wrong_b_shape",
        _SM80,
        (make_tensor_type((16, 16), _BF), make_tensor_type((8, 16), _BF)),
        ExpectedError(match=r"Mma_SM80_16x8x16: b shape"),
    ),
    TypeInferCase(
        "dtype_a_disagrees",
        _SM80,
        (make_tensor_type((16, 16), DType.f16), make_tensor_type((16, 8), _BF)),
        ExpectedError(match=r"Mma_SM80_16x8x16: dtype_a"),
    ),
    TypeInferCase(
        "unsupported_accumulator",
        Mma_SM80_16x8x16(dtype_a=_BF, dtype_b=_BF, dtype_acc=DType.i32),
        (make_tensor_type((16, 16), _BF), make_tensor_type((16, 8), _BF)),
        ExpectedError(match=r"Mma_SM80_16x8x16: dtype_acc combination"),
    ),
    TypeInferCase(
        "bad_a_orientation",
        Mma_SM80_16x8x16(
            dtype_a=_BF, dtype_b=_BF, dtype_acc=_F32, a_layout="K"
        ),
        (make_tensor_type((16, 16), _BF), make_tensor_type((16, 8), _BF)),
        ExpectedError(match=r"Mma_SM80_16x8x16: a_layout"),
    ),
    TypeInferCase(
        "bad_b_orientation",
        Mma_SM80_16x8x16(
            dtype_a=_BF, dtype_b=_BF, dtype_acc=_F32, b_layout="K"
        ),
        (make_tensor_type((16, 16), _BF), make_tensor_type((16, 8), _BF)),
        ExpectedError(match=r"Mma_SM80_16x8x16: b_layout"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_mma_typeinfer(case):
    run_typeinfer_case(case)


def test_plain_layout_describes_the_logical_accumulator_result():
    a = make_tensor_type(
        (16, 16), _BF, storage=_RMEM, layout=Layout((16, 16), (16, 1))
    )
    b = make_tensor_type(
        (16, 8), _BF, storage=_RMEM, layout=Layout((16, 8), (8, 1))
    )

    result = infer_call(_SM80, a, b)

    assert result.layout == Layout(shape=(16, 8), strides=(8, 1))


def test_fragment_mesh_coordinate_names_do_not_affect_compatibility():
    a_mesh = replace(_ATOM.required_scope, names=("a_m", "a_k"))
    b_mesh = replace(_ATOM.required_scope, names=("b_k", "b_n"))
    a_layout = replace(_ATOM.A, mesh=a_mesh)
    b_layout = replace(_ATOM.B, mesh=b_mesh)
    expected_c = replace(_ATOM.C, mesh=a_mesh)
    a = TensorType((16, 16), _BF, a_layout, _RMEM)
    b = TensorType((16, 8), _BF, b_layout, _RMEM)

    result = infer_call(_SM80, a, b)

    assert result.layout == expected_c
    assert result.layout is not _ATOM.C
    assert result.storage is _RMEM


@pytest.mark.parametrize(
    ("name", "op", "a", "b", "match"),
    [
        pytest.param(
            "mismatched_b_fragment",
            _SM80,
            TensorType((16, 16), _BF, _ATOM.A, _RMEM),
            TensorType(
                (16, 8),
                _BF,
                replace(
                    _ATOM.B,
                    layout=Layout(_ATOM.B.layout.shape, (2, 8, 16, 64)),
                ),
                _RMEM,
            ),
            r"input 1.*known SM80 B fragment layout.*Reshard.*materialize-to-RMEM",
            id="mismatched_b_fragment",
        ),
        pytest.param(
            "different_physical_mesh_layout",
            _SM80,
            TensorType((16, 16), _BF, _ATOM.A, _RMEM),
            TensorType(
                (16, 8),
                _BF,
                replace(
                    _ATOM.B,
                    mesh=replace(
                        _ATOM.B.mesh,
                        layout=Layout(shape=(8, 4), strides=(1, 8)),
                    ),
                ),
                _RMEM,
            ),
            r"input 1.*known SM80 B fragment layout.*Reshard.*materialize-to-RMEM",
            id="different_physical_mesh_layout",
        ),
        pytest.param(
            "non_rmem_fragment",
            _SM80,
            TensorType((16, 16), _BF, _ATOM.A, StorageKind.GMEM),
            TensorType((16, 8), _BF, _ATOM.B, StorageKind.GMEM),
            r"input 0.*not RMEM.*Reshard.*materialize-to-RMEM",
            id="non_rmem_fragment",
        ),
        pytest.param(
            "wgmma_shard_claim",
            _WGMMA,
            make_shard_tensor_type(
                (64, 16), _BF, storage=_RMEM, mesh=make_mesh((4,)), attrs=(Split(0),)
            ),
            make_tensor_type((16, 128), _BF, storage=_RMEM),
            r"input 0.*unrepresentable WGMMA ShardLayout.*Reshard",
            id="wgmma_shard_claim",
        ),
    ],
)
def test_unrepresentable_fragment_claims_fail(name, op, a, b, match):
    run_typeinfer_case(TypeInferCase(name, op, (a, b), ExpectedError(match=match)))


def test_fully_broadcast_inputs_do_not_pin_an_output_mesh():
    a = make_shard_tensor_type(
        (16, 16),
        _BF,
        storage=_RMEM,
        mesh=make_mesh((2,)),
        attrs=(Broadcast(),),
    )
    b = make_shard_tensor_type(
        (16, 8),
        _BF,
        storage=_RMEM,
        mesh=make_mesh((4,)),
        attrs=(Broadcast(),),
    )

    result = infer_call(_SM80, a, b)

    assert result.layout is None


_SM80_A = torch.arange(16 * 16, dtype=torch.bfloat16).reshape(16, 16) / 32
_SM80_B = torch.arange(16 * 8, dtype=torch.bfloat16).reshape(16, 8) / 16
_WGMMA_A = torch.arange(64 * 16, dtype=torch.bfloat16).reshape(64, 16) / 64
_WGMMA_B = torch.arange(16 * 128, dtype=torch.bfloat16).reshape(16, 128) / 128


@pytest.mark.parametrize(
    "case",
    [
        EvalCase(
            "sm80_a_matmul_b",
            _SM80,
            (_SM80_A, _SM80_B),
            _SM80_A.float() @ _SM80_B.float(),
        ),
        EvalCase(
            "wgmma_a_matmul_b",
            _WGMMA,
            (_WGMMA_A, _WGMMA_B),
            _WGMMA_A.float() @ _WGMMA_B.float(),
        ),
        EvalCase(
            "orientation_is_encoding_not_logical_transpose",
            Mma_SM80_16x8x16(
                dtype_a=_BF,
                dtype_b=_BF,
                dtype_acc=_F32,
                a_layout="N",
                b_layout="T",
            ),
            (_SM80_A, _SM80_B),
            _SM80_A.float() @ _SM80_B.float(),
        ),
    ],
    ids=lambda case: case.name,
)
def test_mma_evaluator_is_the_two_input_product(case):
    run_eval_case(case)


@pytest.mark.parametrize(
    "case",
    [
        CostCase(
            "sm80_logical",
            _SM80,
            (make_tensor_type((16, 16), _BF), make_tensor_type((16, 8), _BF)),
            flops={_BF: 2 * 16 * 8 * 16},
            traffic=(
                TrafficBytes(read=512),
                TrafficBytes(read=256),
                TrafficBytes(write=512),
            ),
        ),
        CostCase(
            "wgmma_logical",
            _WGMMA,
            (
                make_tensor_type((64, 16), _BF),
                make_tensor_type((16, 128), _BF),
            ),
            flops={_BF: 2 * 64 * 128 * 16},
            traffic=(
                TrafficBytes(read=2_048),
                TrafficBytes(read=4_096),
                TrafficBytes(write=32_768),
            ),
        ),
        CostCase(
            "sm80_thread_fragment",
            _SM80,
            (
                TensorType((16, 16), _BF, _ATOM.A, _RMEM),
                TensorType((16, 8), _BF, _ATOM.B, _RMEM),
            ),
            flops={_BF: 2 * 16 * 8 * 16},
            traffic=(
                TrafficBytes(read=16),
                TrafficBytes(read=8),
                TrafficBytes(write=16),
            ),
            level="thread",
            topologies=_ATOM.required_scope.topologies,
        ),
    ],
    ids=lambda case: case.name,
)
def test_mma_cost(case):
    run_cost_case(case)
