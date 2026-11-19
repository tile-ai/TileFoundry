"""A Varying operand reaches the result of ops that take their type from one operand.

On a target whose gmem is owned per gpu, Varying and Broadcast differ, so
`Broadcast` combined with `Varying` is `Varying` and `Partial` with `Varying` is
refused ([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)). Without
such a target the two read alike, which is why these cases live here and not in
the core op tables.
"""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import math, nn, tensor
from tilefoundry_nvidia import cuda

_H200 = cuda.Target("nvidia.h200_sxm", device_count=2)
_M = types.Mesh((types.Topology("gpu", 2),), types.Layout((2,), (1,)), ("g",))
_CTA = types.Mesh((types.Topology("cta", 4),), types.Layout((4,), (1,)), ("c",))
_B, _V, _P = types.Broadcast(), types.Varying(), types.Partial("sum")


def held(shape, attr, dtype=expr.DType.f32):
    return utils.make_shard_tensor_type(shape, dtype, mesh=_M, attrs=(attr,))


def plain(shape, dtype=expr.DType.f32):
    return types.TensorType(shape=shape, dtype=dtype, layout=types.Layout(shape), storage="gmem")


_INDEX = held((3,), _V, expr.DType.i64)
_HALF = expr.Constant(type=types.TensorType.from_umat_scalar(expr.DType.f32), value=0.5)
_ONE = expr.Constant(type=types.TensorType.from_umat_scalar(expr.DType.f32), value=1.0)
_THREE = expr.Constant(type=types.TensorType.from_umat_scalar(expr.DType.i64), value=3)
_MUL, _ADD = math.Binary(kind=expr.BinaryKind.MUL), math.Binary(kind=expr.BinaryKind.ADD)
_ROPE_TABLE = held((16, 8), _B, expr.DType.bf16)
CASES = (
    OpCase(
        "binary_literal_keeps_broadcast",
        _MUL,
        (held((4,), _B), _HALF),
        type=held((4,), _B),
        target=_H200,
    ),
    OpCase(
        "insert_slice_literal_offset_keeps_broadcast",
        tensor.InsertSlice(),
        (held((8,), _B), held((2,), _B), expr.Tuple(type=types.TupleType(fields=(_THREE.type,)), elements=(_THREE,))),
        type=held((8,), _B),
        target=_H200,
    ),
    OpCase(
        "binary_literal_scale_keeps_partial_sum",
        _MUL,
        (held((4,), _P), _HALF),
        type=held((4,), _P),
        target=_H200,
    ),
    OpCase(
        "binary_literal_shift_refuses_partial_sum",
        _ADD,
        (held((4,), _P), _ONE),
        error=ExpectedError(match=r"ADD.*Partial\(reduction='sum'\) and Broadcast\(\)"),
        target=_H200,
    ),
    OpCase(
        "binary_plain_scalar_variable_stays_varying",
        _MUL,
        (held((4,), _B), types.TensorType.from_umat_scalar(expr.DType.f32)),
        type=plain((4,)),
        target=_H200,
    ),
    OpCase(
        "index_select_varying_index_makes_a_broadcast_table_plain",
        tensor.IndexSelect(dim=0),
        (held((8, 4), _B), _INDEX),
        type=plain((3, 4)),
        target=_H200,
    ),
    OpCase(
        "index_select_plain_index_makes_a_broadcast_table_plain",
        tensor.IndexSelect(dim=0),
        (held((8, 4), _B), plain((3,), expr.DType.i64)),
        type=plain((3, 4)),
        target=_H200,
    ),
    OpCase(
        "index_select_partial_result_refuses_a_varying_index",
        tensor.IndexSelect(dim=0),
        (held((8, 4), types.Split(0)), _INDEX),
        error=ExpectedError(match="Partial result cannot combine with a Varying operand"),
        target=_H200,
    ),
    OpCase(
        "insert_slice_plain_offset_keeps_a_shared_copy_broadcast",
        tensor.InsertSlice(),
        (
            utils.make_shard_tensor_type((8,), mesh=_CTA, attrs=(_B,)),
            utils.make_shard_tensor_type((2,), mesh=_CTA, attrs=(_B,)),
            types.TensorType.from_umat_scalar(expr.DType.i32),
        ),
        type=utils.make_shard_tensor_type((8,), mesh=_CTA, attrs=(_B,)),
        target=_H200,
    ),
    OpCase(
        "index_add_varying_src_makes_dst_plain",
        tensor.IndexAdd(dim=0),
        (held((8, 4), _B), held((3,), _B, expr.DType.i64), held((3, 4), _V)),
        type=plain((8, 4)),
        target=_H200,
    ),
    OpCase(
        "index_copy_varying_index_makes_dst_plain",
        tensor.IndexCopy(dim=0),
        (held((8, 4), _B), _INDEX, held((3, 4), _B)),
        type=plain((8, 4)),
        target=_H200,
    ),
    OpCase(
        "insert_slice_varying_update_makes_dst_plain",
        tensor.InsertSlice(),
        (held((8,), _B), held((2,), _V), held((), _B, expr.DType.i32)),
        type=plain((8,)),
        target=_H200,
    ),
    OpCase(
        "rms_norm_varying_weight_makes_x_plain",
        nn.RMSNorm(eps=1e-06),
        (held((4, 8), _B), held((8,), _V)),
        type=plain((4, 8)),
        target=_H200,
    ),
    OpCase(
        "layer_norm_varying_weight_makes_x_plain",
        nn.LayerNorm(axis=-1, eps=1e-05),
        (held((4, 8), _B), held((8,), _V), held((8,), _B)),
        type=plain((4, 8)),
        target=_H200,
    ),
    OpCase(
        "rope_varying_table_makes_q_and_k_plain",
        nn.RoPE(),
        (
            held((1, 2, 8), _B, expr.DType.bf16),
            held((1, 2, 8), _B, expr.DType.bf16),
            held((16, 8), _V, expr.DType.bf16),
            _ROPE_TABLE,
            held((1,), _B, expr.DType.i32),
        ),
        type=types.TupleType(fields=(plain((1, 2, 8), expr.DType.bf16),) * 2),
        target=_H200,
    ),
    OpCase(
        "rope_partial_q_keeps_partial_beside_broadcast_tables",
        nn.RoPE(),
        (
            held((1, 2, 8), _P, expr.DType.bf16),
            held((1, 2, 8), _B, expr.DType.bf16),
            _ROPE_TABLE,
            _ROPE_TABLE,
            held((1,), _B, expr.DType.i32),
        ),
        type=types.TupleType(
            fields=(held((1, 2, 8), _P, expr.DType.bf16), held((1, 2, 8), _B, expr.DType.bf16))
        ),
        target=_H200,
    ),
)
test_op_case = op_case_test(CASES)
