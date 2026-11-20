"""A Varying operand reaches the result of every op; a plain value is Broadcast.

On a target whose gmem is owned per gpu, Varying and Broadcast differ, so
`Broadcast` combined with `Varying` is `Varying` and `Partial` with `Varying` is
refused ([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)). Without
such a target the two read alike, which is why these cases live here and not in
the core op tables.
"""

from __future__ import annotations

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
)
test_op_case = op_case_test(CASES)


@pytest.mark.parametrize(
    ("module", "varying"),
    (
        ("LiteralBounds", False),
        ("PlainCount", False),
        ("DimVarExtent", False),
        ("OuterIndexStart", False),
        ("CoordStart", True),
        ("DirectCoord", True),
        ("HiddenCoord", True),
    ),
)
def test_loop_index_reads_varying_only_from_a_varying_bound(module, varying) -> None:
    """A row picked at the index is Varying exactly when the index differs per GPU; else plain."""
    loaded = cli.load_authored_ir(f"{_FIXTURES / 'ops' / 'loop_index_reads.py'}:{module}")
    (picked,) = (
        node
        for node in core.collect_exprs(loaded.entry_function().body)
        if isinstance(node, expr.Call) and isinstance(node.target, tensor.IndexSelect)
    )
    shard = types.shard_layout_of(picked.type.layout)
    assert (shard.attrs == (_V,) if varying else shard is None), picked.type.layout


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
    ("module", "message"),
    (
        ("CoordArgument", r"argument 1 is Varying on mesh axis g; parameter i is Broadcast\. Write the parameter as '@ \{g @ V\(\)\}'"),
        ("LoopCarry", r"enters B and leaves V; write the target as '@ \{g @ V\(\)\}'"),
        ("ScheduleTarget", r"'src' is Varying on mesh axis lane; the target is Broadcast there: write the target as '@ \{lane @ V\(\)\}'"),
    ),
)
def test_varying_into_broadcast_names_the_axis_to_write(module, message) -> None:
    """A Varying value where the type says Broadcast is refused; the remedy names the mesh axis."""
    with pytest.raises(expr.VerifyError, match=message):
        cli.load_authored_ir(f"{_FIXTURES / 'diagnostics' / 'varying_into_broadcast.py'}:{module}")


def _fixture(fixture: str, module: str):
    return cli.load_authored_ir(f"{_FIXTURES / 'ops' / fixture}:{module}")


def _calls(loaded, op):
    return [node for node in core.collect_exprs(loaded.entry_function().body) if isinstance(node, expr.Call) and isinstance(node.target, op)]


def test_window_at_a_varying_offset_needs_a_varying_carry() -> None:
    """A gmem window at a per-CTA offset into a plain smem carry is refused; written V, it loads."""
    with pytest.raises(expr.VerifyError, match=r"'src' is Varying on mesh axis c; the target is Broadcast there"):
        _fixture("varying_window_carry.py", "WindowCarryBroadcast")
    _fixture("varying_window_carry.py", "WindowCarryVarying")


def test_bitcast_relays_a_varying_smem_ring_and_keeps_its_attrs() -> None:
    """TMA's swizzle pattern reads through a stated V; a bitcast without Split/Partial keeps V."""
    loaded = _fixture("varying_ring_bitcast.py", "RingBitcast")
    for call in _calls(loaded, tensor.Bitcast):
        shard = types.shard_layout_of(call.type.layout)
        assert shard is not None and shard.attrs == (_V,)
        assert utils.topology_names(shard.mesh) == ("cta",)
        assert call.type.storage is expr.StorageKind.SMEM


def test_smem_tile_written_in_one_thread_slice_is_read_by_all_threads() -> None:
    """A body layout drops the thread topology: smem is one copy per CTA, read by every thread."""
    loaded = _fixture("varying_tile_other_slice.py", "TileOtherSlice")
    tiles = [
        type_
        for node in core.collect_exprs(loaded.entry_function().body)
        for type_ in utils.tensor_types(getattr(node, "type", None) or types.TupleType(fields=()))
        if type_.storage is expr.StorageKind.SMEM
    ]
    assert tiles and all(utils.topology_names(types.shard_layout_of(t.layout).mesh) == ("cta",) for t in tiles)


@pytest.mark.parametrize(("attr", "refused"), ((_B, False), (types.Split(0), True)))
def test_reachability_omits_an_outside_topology_without_split_or_partial(attr, refused) -> None:
    """An execution mesh may omit outside topologies that contain neither Split nor Partial.

    See [shard §6.1](docs/spec/shard.md#61-shard-attr-semantics).
    """
    both = types.Mesh(
        (types.Topology("cta", 4), types.Topology("thread", 32)), types.Layout(((4,), (32,)), ((32,), (1,))), ("c", "t")
    )
    threads = types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)), ("t",))
    value = types.TensorType((4,), expr.DType.f32, types.ShardLayout(types.Layout((4,)), (attr, _V), both), "smem")
    if refused:
        with pytest.raises(ValueError, match="laid out more finely"):
            typeinfer.check_input_reachability(tensor.Reshape(new_shape=(4,)), (value,), threads)
    else:
        typeinfer.check_input_reachability(tensor.Reshape(new_shape=(4,)), (value,), threads)
