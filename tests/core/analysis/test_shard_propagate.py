"""Generic shard propagation engine over a matmul-shaped relation.

Domain dims: 0=M, 1=N, 2=K. lhs[M,K], rhs[K,N], out[M,N] (K reduced).

What is kept here is one case per rule the engine has to decide, plus every case
where the answer is "refuse": a Split the engine cannot carry into the output has
to fail rather than silently degrade to Broadcast, because a value quietly
declared replicated when each point holds a shard of it is read whole by everyone
downstream.
"""

from __future__ import annotations

import isl
import pytest

from tilefoundry import utils
from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import types
from tilefoundry.ir import utils as ir_utils

_GPU = types.Mesh((types.Topology("gpu", 8),), types.Layout((8,), (1,)), names=("g",))


_GPU2 = types.Mesh((types.Topology("gpu", 4),), types.Layout((2, 2), (2, 1)), names=("a", "b"))


def _access(source: str) -> access.Relation:
    relation = isl.map(source)
    return access.Relation(relation, affine=utils.as_multi_aff(relation))


def _matmul_relation() -> tuple[access.Relation, ...]:
    return (
        _access("{ [m, n, k] -> [m, k] }"),
        _access("{ [m, n, k] -> [k, n] }"),
        _access("{ [m, n, k] -> [m, n] }"),
    )


def _elementwise_relation() -> tuple[access.Relation, ...]:
    ident = _access("{ [m, n] -> [m, n] }")
    return (ident, ident, ident)


def _strides(shape) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _shard(shape, *attrs) -> types.ShardLayout:
    return types.ShardLayout(layout=types.Layout(shape=shape, strides=_strides(shape)), attrs=attrs, mesh=_GPU)


def _shard2(shape, *attrs) -> types.ShardLayout:
    return types.ShardLayout(layout=types.Layout(shape=shape, strides=_strides(shape)), attrs=attrs, mesh=_GPU2)


def test_a_split_of_a_surviving_axis_reaches_the_output():
    """Either operand may be the sharded one, and the axis it splits decides the output axis.

    Either operand may be the sharded one, and the axis it splits decides the
    output axis: rhs[K,N] split on N lands on out axis 1, lhs[M,K] split on M on
    out axis 0. With no sharded input at all there is nothing to propagate, which
    is ``None`` rather than an unsharded layout.
    """
    rhs_t = types.TensorType.from_shape((4, 8), layout=_shard((4, 8), types.Split(1)))
    on_n = typeinfer.derive_output_shard_layout(
        (types.TensorType.from_shape((16, 4)), rhs_t), _matmul_relation(), (16, 8)
    )
    assert on_n.attrs == (types.Split(1),)

    lhs_t = types.TensorType.from_shape((16, 4), layout=_shard((16, 4), types.Split(0)))
    on_m = typeinfer.derive_output_shard_layout(
        (lhs_t, types.TensorType.from_shape((4, 8))), _matmul_relation(), (16, 8)
    )
    assert on_m.attrs == (types.Split(0),)

    assert (
        typeinfer.derive_output_shard_layout(
            (types.TensorType.from_shape((16, 4)), types.TensorType.from_shape((4, 8))),
            _matmul_relation(),
            (16, 8),
        )
        is None
    )


CONTRACTION_SPLITS = [
    pytest.param(
        _shard((4, 8), types.Split(0)),
        frozenset({2}),
        (types.Partial("sum"),),
        id="partial_reduction",
    ),
    pytest.param(None, frozenset(), (types.Broadcast(),), id="complete_reduction"),
]


@pytest.mark.parametrize(("rhs_layout", "partial_dims", "attrs"), CONTRACTION_SPLITS)
def test_a_split_contraction_dim_becomes_a_value_state(rhs_layout, partial_dims, attrs):
    lhs_t = types.TensorType.from_shape((16, 4), layout=_shard((16, 4), types.Split(1)))
    rhs_t = types.TensorType.from_shape((4, 8), layout=rhs_layout) if rhs_layout else types.TensorType.from_shape((4, 8))
    out = typeinfer.derive_output_shard_layout(
        (lhs_t, rhs_t), _matmul_relation(), (16, 8), partial_reduction_dims=partial_dims
    )

    assert out.attrs == attrs


REFUSED = [
    pytest.param(
        (
            types.TensorType.from_shape((16, 4), layout=_shard((16, 4), types.Split(0))),
            types.TensorType.from_shape((4, 8), layout=_shard((4, 8), types.Split(1))),
        ),
        _matmul_relation(),
        (16, 8),
        "incompatible output shard",
        id="incompatible_split",
    ),
    pytest.param(
        (types.TensorType.from_shape((12,), layout=_shard((12,), types.Split(0))),),
        (
            _access("{ [m, n] -> [m + n] }"),
            _access("{ [m, n] -> [m, n] }"),
        ),
        (4, 8),
        "non-projection access",
        id="split_on_non_projection_access",
    ),
    pytest.param(
        (types.TensorType.from_shape((4, 8), layout=_shard((4, 8), types.Split(0))),),
        (
            _access("{ [m, n] -> [m, n] }"),
            _access("{ [m, n] -> [m + n] }"),
        ),
        (12,),
        "non-projection output access",
        id="split_surviving_via_complex_output",
    ),
]


@pytest.mark.parametrize(("operands", "relation", "shape", "refusal"), REFUSED)
def test_what_cannot_be_derived_fails_closed(operands, relation, shape, refusal):
    with pytest.raises(ValueError, match=refusal):
        typeinfer.derive_output_shard_layout(operands, relation, shape)


def test_two_mesh_axes_on_same_output_axis_factorize():

    lhs_t = types.TensorType.from_shape((4, 8), layout=_shard2((4, 8), types.Split(0), types.Broadcast()))
    rhs_t = types.TensorType.from_shape((4, 8), layout=_shard2((4, 8), types.Broadcast(), types.Split(0)))
    out = typeinfer.derive_output_shard_layout((lhs_t, rhs_t), _elementwise_relation(), (4, 8))

    assert out.layout.shape == (2, 2, 8)
    assert out.attrs == (types.Split(0), types.Split(1))
    assert out.mesh is _GPU2


def test_a_synthesised_layout_agrees_with_a_from_scratch_one():
    """Test a synthesised layout agrees with a from scratch one.

    From-scratch and propagated sharding both use ``default_shard_layout``
    ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)), so equal logical
    distributions must compare equal. Neither operand carries both splits here,
    so synthesis factors each size-8 axis into mesh extent 2 and residual 4.
    """
    lhs_t = types.TensorType.from_shape((8, 8), layout=_shard2((8, 8), types.Split(0), types.Broadcast()))
    rhs_t = types.TensorType.from_shape((8, 8), layout=_shard2((8, 8), types.Broadcast(), types.Split(1)))
    ident = _access("{ [m, n] -> [m, n] }")
    rel = (ident, ident, ident)

    out = typeinfer.derive_output_shard_layout((lhs_t, rhs_t), rel, (8, 8))

    expected = ir_utils.make_shard_tensor_type((8, 8), mesh=_GPU2, attrs=(types.Split(0), types.Split(1)))
    assert out == expected.layout


def test_an_input_partial_propagates_on_its_own_mesh_axis():
    """A Partial is a value state with no layout axis.

    A Partial is a value state with no layout axis, so what identifies it is
    the mesh axis it sits on. It propagates through an elementwise identity rather
    than being dropped, and two different reductions on two axes stay two
    reductions -- collapsing them would make a sum-partial and a max-partial
    indistinguishable, which is a wrong result rather than a wrong layout.
    """
    ident = _access("{ [m, n] -> [m, n] }")
    rel = (ident, ident)
    x_t = types.TensorType.from_shape((4, 8), layout=_shard2((4, 8), types.Partial("sum"), types.Broadcast()))

    out = typeinfer.derive_output_shard_layout((x_t,), rel, (4, 8))
    assert out.attrs == (types.Partial("sum"), types.Broadcast())

    on_a = _shard2((4, 8), types.Partial("sum"), types.Broadcast())
    on_b = _shard2((4, 8), types.Broadcast(), types.Partial("sum"))
    assert types.partial_reductions_by_axis(on_a) == ("sum", None)
    assert types.partial_reductions_by_axis(on_b) == (None, "sum")
    assert types.partial_reductions_by_axis(_shard2((4, 8), types.Partial("sum"), types.Partial("max"))) == (
        "sum",
        "max",
    )
    assert types.partial_reductions_by_axis(types.TensorType.from_shape((4, 8))) == ()


def test_an_input_whole_at_one_level_keeps_that_level_aligned():
    """Rows split over CTAs and threads, times a gamma every CTA holds whole.

    The gamma states the CTA level as all Broadcast. It must keep that level, so
    its thread split sits on the same mesh axis as the rows' thread split.
    """
    mesh = types.Mesh((types.Topology("cta", 2), types.Topology("thread", 4)), layout=(2, 4), names=("tile", "y"))
    rows = types.TensorType.from_shape(
        (2, 16),
        layout=types.ShardLayout(types.Layout((2, 16), (16, 1)), (types.Split(0), types.Split(1)), mesh),
    )
    gamma = types.TensorType.from_shape(
        (1, 16),
        layout=types.ShardLayout(types.Layout((1, 16), (16, 1)), (types.Broadcast(), types.Split(1)), mesh),
    )
    relation = (
        _access("{ [m, n] -> [m, n] }"),
        _access("{ [m, n] -> [0, n] }"),
        _access("{ [m, n] -> [m, n] }"),
    )

    out = typeinfer.derive_output_shard_layout((rows, gamma), relation, (2, 16))

    assert out.attrs == (types.Split(0), types.Split(1))
    assert out.mesh == mesh
