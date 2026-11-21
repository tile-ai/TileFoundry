"""A Varying operand reaches the result of every op; a plain value is Broadcast.

On a target whose gmem is owned per gpu, Varying and Broadcast differ, so
`Broadcast` combined with `Varying` is `Varying` and `Partial` with `Varying` is
refused ([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)). Without
such a target the two read alike, which is why these cases live here and not in
the core op tables.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry import cli
from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import core, expr, types, utils
from tilefoundry.ir.hir import math, nn, tensor
from tilefoundry_nvidia import cuda

_FIXTURES = Path(__file__).parents[2] / "fixtures"
_H200 = cuda.Target("nvidia.h200_sxm", device_count=2)
_M = types.Mesh((types.Topology("gpu", 2),), types.Layout((2,), (1,)), ("g",))
_CTA = types.Mesh((types.Topology("cta", 4),), types.Layout((4,), (1,)), ("c",))
_THREADS = types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)), ("t",))
_WARPS_BY_LANES = types.Mesh((types.Topology("thread", 32),), types.Layout((4, 8), (8, 1)), ("w", "l"))
_FOUR_THREADS = types.Mesh((types.Topology("thread", 4),), types.Layout((4,), (1,)), ("t",))
_TWO_CTAS = types.Mesh((types.Topology("cta", 2),), types.Layout((2,), (1,)), ("n",))
_ROW_MAJOR = (((10, 8), (4, 16)), ((512, 64), (16, 1)))
_K_MAJOR = (((4, 16), (10, 8)), ((16, 1), (512, 64)))
_RMEM = expr.StorageKind.RMEM
_B, _V, _P = types.Broadcast(), types.Varying(), types.Partial("sum")


def held(shape, attr, dtype=expr.DType.f32):
    return utils.make_shard_tensor_type(shape, dtype, mesh=_M, attrs=(attr,))


def plain(shape, dtype=expr.DType.f32, storage="gmem"):
    return types.TensorType(shape=shape, dtype=dtype, layout=types.Layout(shape), storage=storage)


def _ring_per_cta(outer):
    """A swizzled smem ring that each of two CTAs holds its own copy of."""
    layout = types.ComposedLayout(inner=types.Swizzle(3, 4, 3), offset=0, outer=types.Layout(*outer))
    return types.ShardLayout(layout, (_V,), _TWO_CTAS)


_INDEX = held((3,), _V, expr.DType.i64)
_HALF = expr.Constant(type=types.TensorType.from_umat_scalar(expr.DType.f32), value=0.5)
_ONE = expr.Constant(type=types.TensorType.from_umat_scalar(expr.DType.f32), value=1.0)
_THREE = expr.Constant(type=types.TensorType.from_umat_scalar(expr.DType.i64), value=3)
_MUL, _ADD = math.Binary(kind=expr.BinaryKind.MUL), math.Binary(kind=expr.BinaryKind.ADD)
_ROPE_TABLE = plain((16, 8), expr.DType.bf16)
_START = expr.Var(name="start", type=held((), _V, expr.DType.i64))
CASES = (
    OpCase(
        "binary_literal_arithmetic_is_broadcast",
        _MUL,
        (plain((4,)), expr.Call(_MUL, (_HALF, _ONE), type=_HALF.type)),
        type=plain((4,)),
        target=_H200,
    ),
    OpCase(
        "binary_plain_beside_split_is_split",
        _MUL,
        (held((4,), types.Split(0)), plain((4,))),
        type=held((4,), types.Split(0)),
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
        "binary_varying_scalar_makes_a_plain_operand_varying",
        _MUL,
        (plain((4,)), held((), _V)),
        type=held((4,), _V),
        target=_H200,
    ),
    OpCase(
        "binary_plain_scalar_on_a_slice_keeps_its_mesh",
        _MUL,
        (utils.make_shard_tensor_type((4,), mesh=_M[0], attrs=(_B,)), types.TensorType.from_umat_scalar(expr.DType.f32)),
        type=utils.make_shard_tensor_type((4,), mesh=_M[0], attrs=(_B,)),
        target=_H200,
    ),
    OpCase(
        "slice_varying_start_makes_the_window_varying",
        tensor.Slice(sizes=(2,), strides=(1,)),
        (plain((8,)), expr.Tuple(type=types.TupleType(fields=(_START.type,)), elements=(_START,))),
        type=held((2,), _V),
        target=_H200,
    ),
    OpCase(
        "repeat_interleave_varying_on_a_slice_keeps_its_mesh",
        tensor.RepeatInterleave(repeats=2, axis=0),
        (utils.make_shard_tensor_type((4,), mesh=_M[0], attrs=(_V,)),),
        type=utils.make_shard_tensor_type((8,), mesh=_M[0], attrs=(_V,)),
        target=_H200,
    ),
    OpCase(
        "repeat_interleave_varying_on_every_unit_keeps_its_mesh",
        tensor.RepeatInterleave(repeats=2, axis=0),
        (held((4,), _V),),
        type=held((8,), _V),
        target=_H200,
    ),
    OpCase(
        "index_select_varying_index_makes_a_plain_table_varying",
        tensor.IndexSelect(dim=0),
        (plain((8, 4)), _INDEX),
        type=held((3, 4), _V),
        target=_H200,
    ),
    OpCase(
        "index_select_plain_index_keeps_a_plain_table_plain",
        tensor.IndexSelect(dim=0),
        (plain((8, 4)), plain((3,), expr.DType.i64)),
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
        "insert_slice_varying_offset_keeps_a_shared_copy_broadcast",
        tensor.InsertSlice(),
        (
            plain((8,)),
            plain((2,)),
            utils.make_shard_tensor_type((), expr.DType.i32, mesh=_CTA, attrs=(_V,)),
        ),
        type=plain((8,)),
        target=_H200,
    ),
    OpCase(
        "index_add_varying_src_makes_dst_varying",
        tensor.IndexAdd(dim=0),
        (plain((8, 4)), plain((3,), expr.DType.i64), held((3, 4), _V)),
        type=held((8, 4), _V),
        target=_H200,
    ),
    OpCase(
        "index_copy_varying_index_makes_dst_varying",
        tensor.IndexCopy(dim=0),
        (plain((8, 4)), _INDEX, plain((3, 4))),
        type=held((8, 4), _V),
        target=_H200,
    ),
    OpCase(
        "insert_slice_varying_update_makes_dst_varying",
        tensor.InsertSlice(),
        (plain((8,)), held((2,), _V), plain((), expr.DType.i32)),
        type=held((8,), _V),
        target=_H200,
    ),
    OpCase(
        "binary_reads_varying_on_a_topology_the_first_operand_omits",
        _ADD,
        (
            utils.make_shard_tensor_type((4,), mesh=_CTA, attrs=(_V,), storage=_RMEM),
            utils.make_shard_tensor_type((4,), mesh=_THREADS, attrs=(_V,), storage=_RMEM),
        ),
        type=utils.make_shard_tensor_type((4,), mesh=utils.from_meshes(_CTA, _THREADS), attrs=(_V, _V), storage=_RMEM),
        target=_H200,
    ),
    OpCase(
        "insert_slice_update_varying_on_a_topology_the_dst_omits",
        tensor.InsertSlice(),
        (
            utils.make_shard_tensor_type((8,), mesh=_THREADS, attrs=(_V,), storage=_RMEM),
            utils.make_shard_tensor_type((2,), mesh=_CTA, attrs=(_V,), storage=_RMEM),
            plain((), expr.DType.i32),
        ),
        type=utils.make_shard_tensor_type((8,), mesh=utils.from_meshes(_CTA, _THREADS), attrs=(_V, _V), storage=_RMEM),
        target=_H200,
    ),
    OpCase(
        "rms_norm_varying_weight_makes_x_varying",
        nn.RMSNorm(eps=1e-06),
        (plain((4, 8)), held((8,), _V)),
        type=held((4, 8), _V),
        target=_H200,
    ),
    OpCase(
        "layer_norm_varying_weight_makes_x_varying",
        nn.LayerNorm(axis=-1, eps=1e-05),
        (plain((4, 8)), held((8,), _V), plain((8,))),
        type=held((4, 8), _V),
        target=_H200,
    ),
    OpCase(
        "rope_varying_table_makes_q_and_k_varying",
        nn.RoPE(),
        (
            plain((1, 2, 8), expr.DType.bf16),
            plain((1, 2, 8), expr.DType.bf16),
            held((16, 8), _V, expr.DType.bf16),
            _ROPE_TABLE,
            plain((1,), expr.DType.i32),
        ),
        type=types.TupleType(fields=(held((1, 2, 8), _V, expr.DType.bf16),) * 2),
        target=_H200,
    ),
    OpCase(
        "rope_partial_q_keeps_partial_beside_broadcast_tables",
        nn.RoPE(),
        (
            held((1, 2, 8), _P, expr.DType.bf16),
            plain((1, 2, 8), expr.DType.bf16),
            _ROPE_TABLE,
            _ROPE_TABLE,
            plain((1,), expr.DType.i32),
        ),
        type=types.TupleType(fields=(held((1, 2, 8), _P, expr.DType.bf16), plain((1, 2, 8), expr.DType.bf16))),
        target=_H200,
    ),
    OpCase(
        "index_select_varying_on_another_axis_structure_of_one_topology_is_varying_everywhere",
        tensor.IndexSelect(dim=0),
        (
            types.TensorType((16, 4), expr.DType.f32, types.ShardLayout(types.Layout((16, 4)), (_V, _B), _WARPS_BY_LANES), "rmem"),
            utils.make_shard_tensor_type((2,), expr.DType.i64, mesh=_THREADS, attrs=(_V,), storage=_RMEM),
        ),
        type=types.TensorType((2, 4), expr.DType.f32, types.ShardLayout(types.Layout((2, 4), (1, 2)), (_V, _V), _WARPS_BY_LANES), "rmem"),
        target=cuda.Target("nvidia.h200_sxm"),
    ),
    OpCase(
        "index_select_from_a_shared_tile_by_a_varying_row_is_varying",
        tensor.IndexSelect(dim=0),
        (plain((4, 3), storage="smem"), utils.make_shard_tensor_type((1,), expr.DType.i64, mesh=_FOUR_THREADS, attrs=(_V,), storage=_RMEM)),
        type=utils.make_shard_tensor_type((1, 3), mesh=_FOUR_THREADS, attrs=(_V,), storage="smem"),
        target=cuda.Target("nvidia.h200_sxm"),
    ),
    OpCase(
        "insert_slice_varying_update_per_cta_keeps_a_shared_copy_broadcast",
        tensor.InsertSlice(),
        (plain((8,)), utils.make_shard_tensor_type((2,), mesh=_CTA, attrs=(_V,)), plain((), expr.DType.i32)),
        type=plain((8,)),
        target=cuda.Target("nvidia.h200_sxm"),
    ),
    OpCase(
        "bitcast_of_a_varying_smem_ring_states_and_keeps_varying",
        tensor.Bitcast(layout=_ring_per_cta(_K_MAJOR)),
        (types.TensorType((80, 64), expr.DType.bf16, _ring_per_cta(_ROW_MAJOR), "smem"),),
        type=types.TensorType((64, 80), expr.DType.bf16, _ring_per_cta(_K_MAJOR), "smem"),
        target=cuda.Target("nvidia.h200_sxm"),
    ),
)
test_op_case = op_case_test(CASES)


@pytest.mark.parametrize(
    ("fixture", "module", "varying"),
    (
        ("loop_index_reads.py", "LiteralBounds", None),
        ("loop_index_reads.py", "PlainCount", None),
        ("loop_index_reads.py", "DimVarExtent", None),
        ("loop_index_reads.py", "OuterIndexStart", None),
        ("loop_index_reads.py", "CoordStart", ("gpu",)),
        ("loop_index_reads.py", "DirectCoord", ("gpu",)),
        ("loop_index_reads.py", "HiddenCoord", ("gpu",)),
        ("varying_loops_and_fields.py", "TripCount", ("gpu",)),
        ("varying_loops_and_fields.py", "OutsideBound", ("cta",)),
        ("varying_loops_and_fields.py", "TupleField", None),
    ),
)
def test_a_result_is_varying_only_where_a_loop_bound_or_its_field_is(fixture, module, varying) -> None:
    """A row picked at a loop index, a loop result, a tuple field: Varying on these, else plain."""
    returned = cli.load_authored_ir(f"{_FIXTURES / 'ops' / fixture}:{module}").entry_function().return_type
    shard = types.shard_layout_of(returned.layout)
    assert (None if shard is None else utils.topology_names(shard.mesh)) == varying, returned.layout
    assert shard is None or shard.attrs == (_V,)


def test_broadcast_index_binds_a_plain_parameter() -> None:
    """A Broadcast argument binds a plain parameter."""
    loaded = cli.load_authored_ir(f"{_FIXTURES / 'ops' / 'loop_index_reads.py'}:IndexArgument")
    (call,) = (
        node
        for node in core.collect_exprs(loaded.entry_function().body)
        if isinstance(node, expr.Call) and isinstance(node.target, core.FunctionRef)
    )
    assert types.shard_layout_of(call.args[1].type.layout) is None


@pytest.mark.parametrize(
    ("fixture", "module", "line"),
    (
        ("diagnostics/varying_into_broadcast.py", "CoordArgument", 28),
        ("diagnostics/varying_into_broadcast.py", "LoopCarry", 39),
        ("diagnostics/varying_into_broadcast.py", "ScheduleTarget", 52),
        ("ops/varying_loops_and_fields.py", "OtherSlice", 73),
        ("diagnostics/varying_bitcast_plain_end.py", "BitcastPlainEnd", 14),
    ),
)
def test_a_varying_value_where_the_type_says_broadcast_is_refused(fixture, module, line) -> None:
    """A Varying argument, carry, schedule or bitcast source into Broadcast; an unreached slice.

    The refusal carries the line of the refused value; a carry's is its loop's ``for``.
    """
    with pytest.raises(expr.VerifyError) as refused:
        cli.load_authored_ir(f"{_FIXTURES / fixture}:{module}")
    assert refused.value.span.line == line


@pytest.mark.parametrize(("outside", "refused"), ((_B, False), (types.Split(0), True)), ids=("broadcast", "split"))
def test_an_execution_mesh_may_omit_an_outside_topology_only_without_split_or_partial(outside, refused) -> None:
    """A smem value Varying per thread is read in a thread-only scope unless its CTA axis splits it.

    See [shard §6.1](docs/spec/shard.md#61-shard-attr-semantics).
    """
    both = types.Mesh(
        (types.Topology("cta", 4), types.Topology("thread", 32)), types.Layout(((4,), (32,)), ((32,), (1,))), ("c", "t")
    )
    value = types.TensorType((4,), expr.DType.f32, types.ShardLayout(types.Layout((4,)), (outside, _V), both), "smem")
    with pytest.raises(ValueError) if refused else contextlib.nullcontext():
        typeinfer.check_input_reachability(tensor.Reshape(new_shape=(4,)), (value,), _THREADS)


def test_per_cta_tiles_lower_through_thread_slices(tmp_path) -> None:
    """A tile Varying per CTA is one plain tile to the threads: WGMMA reads it, a Copy writes it."""
    fixture = _FIXTURES / "ops" / "varying_tile_from_fragments.py"
    assert cli.main(["schedule", "finalize", str(fixture), str(tmp_path / "tir.py")]) == 0
