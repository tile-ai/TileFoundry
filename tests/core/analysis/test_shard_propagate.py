"""Target-neutral tests and helpers."""

from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.types import (
    Layout,
    Mesh,
    ShardLayout,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Broadcast, Partial, Split
from tilefoundry.utils.isl_utils import as_multi_aff
from tilefoundry.visitor_registry.access_relation import AccessRelation
from tilefoundry.visitor_registry.shard_propagate import (
    derive_output_shard_layout,
    partial_reductions_by_axis,
)

_GPU = Mesh((Topology("gpu", 8),), Layout((8,), (1,)), names=("g",))


_GPU2 = Mesh((Topology("gpu", 4),), Layout((2, 2), (2, 1)), names=("a", "b"))


def _access(source: str) -> AccessRelation:
    relation = isl.map(source)
    return AccessRelation(relation, affine=as_multi_aff(relation))


def _matmul_relation() -> tuple[AccessRelation, ...]:
    return (
        _access("{ [m, n, k] -> [m, k] }"),
        _access("{ [m, n, k] -> [k, n] }"),
        _access("{ [m, n, k] -> [m, n] }"),
    )


def _elementwise_relation() -> tuple[AccessRelation, ...]:
    ident = _access("{ [m, n] -> [m, n] }")
    return (ident, ident, ident)


def _strides(shape) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _shard(shape, *attrs) -> ShardLayout:
    return ShardLayout(layout=Layout(shape=shape, strides=_strides(shape)), attrs=attrs, mesh=_GPU)


def _shard2(shape, *attrs) -> ShardLayout:
    return ShardLayout(layout=Layout(shape=shape, strides=_strides(shape)), attrs=attrs, mesh=_GPU2)


def test_a_split_of_a_surviving_axis_reaches_the_output():
    """Either operand may be the sharded one, and the axis it splits decides the output axis.

    Either operand may be the sharded one, and the axis it splits decides the
    output axis: rhs[K,N] split on N lands on out axis 1, lhs[M,K] split on M on
    out axis 0. With no sharded input at all there is nothing to propagate, which
    is ``None`` rather than an unsharded layout.
    """
    rhs_t = make_tensor_type((4, 8), layout=_shard((4, 8), Split(1)))
    on_n = derive_output_shard_layout(
        (make_tensor_type((16, 4)), rhs_t), _matmul_relation(), (16, 8)
    )
    assert on_n.attrs == (Split(1),)

    lhs_t = make_tensor_type((16, 4), layout=_shard((16, 4), Split(0)))
    on_m = derive_output_shard_layout(
        (lhs_t, make_tensor_type((4, 8))), _matmul_relation(), (16, 8)
    )
    assert on_m.attrs == (Split(0),)

    assert (
        derive_output_shard_layout(
            (make_tensor_type((16, 4)), make_tensor_type((4, 8))),
            _matmul_relation(),
            (16, 8),
        )
        is None
    )


CONTRACTION_SPLITS = [
    pytest.param(
        _shard((4, 8), Split(0)),
        frozenset({2}),
        (Partial("sum"),),
        id="partial_reduction",
    ),
    pytest.param(None, frozenset(), (Broadcast(),), id="complete_reduction"),
]


@pytest.mark.parametrize(("rhs_layout", "partial_dims", "attrs"), CONTRACTION_SPLITS)
def test_a_split_contraction_dim_becomes_a_value_state(rhs_layout, partial_dims, attrs):
    lhs_t = make_tensor_type((16, 4), layout=_shard((16, 4), Split(1)))
    rhs_t = make_tensor_type((4, 8), layout=rhs_layout) if rhs_layout else make_tensor_type((4, 8))
    out = derive_output_shard_layout(
        (lhs_t, rhs_t), _matmul_relation(), (16, 8), partial_reduction_dims=partial_dims
    )

    assert out.attrs == attrs


REFUSED = [
    pytest.param(
        (
            make_tensor_type((16, 4), layout=_shard((16, 4), Split(0))),
            make_tensor_type((4, 8), layout=_shard((4, 8), Split(1))),
        ),
        _matmul_relation(),
        (16, 8),
        "incompatible output shard",
        id="incompatible_split",
    ),
    pytest.param(
        (make_tensor_type((12,), layout=_shard((12,), Split(0))),),
        (
            _access("{ [m, n] -> [m + n] }"),
            _access("{ [m, n] -> [m, n] }"),
        ),
        (4, 8),
        "non-projection access",
        id="split_on_non_projection_access",
    ),
    pytest.param(
        (make_tensor_type((4, 8), layout=_shard((4, 8), Split(0))),),
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
        derive_output_shard_layout(operands, relation, shape)


def test_two_mesh_axes_on_same_output_axis_factorize():

    lhs_t = make_tensor_type((4, 8), layout=_shard2((4, 8), Split(0), Broadcast()))
    rhs_t = make_tensor_type((4, 8), layout=_shard2((4, 8), Broadcast(), Split(0)))
    out = derive_output_shard_layout((lhs_t, rhs_t), _elementwise_relation(), (4, 8))

    assert out.layout.shape == (2, 2, 8)
    assert out.attrs == (Split(0), Split(1))
    assert out.mesh is _GPU2


def test_a_synthesised_layout_agrees_with_a_from_scratch_one():
    """Test a synthesised layout agrees with a from scratch one.

    From-scratch and propagated sharding both use ``default_shard_layout``
    ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)), so equal logical
    distributions must compare equal. Neither operand carries both splits here,
    so synthesis factors each size-8 axis into mesh extent 2 and residual 4.
    """
    lhs_t = make_tensor_type((8, 8), layout=_shard2((8, 8), Split(0), Broadcast()))
    rhs_t = make_tensor_type((8, 8), layout=_shard2((8, 8), Broadcast(), Split(1)))
    ident = _access("{ [m, n] -> [m, n] }")
    rel = (ident, ident, ident)

    out = derive_output_shard_layout((lhs_t, rhs_t), rel, (8, 8))

    expected = make_shard_tensor_type((8, 8), mesh=_GPU2, attrs=(Split(0), Split(1)))
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
    x_t = make_tensor_type((4, 8), layout=_shard2((4, 8), Partial("sum"), Broadcast()))

    out = derive_output_shard_layout((x_t,), rel, (4, 8))
    assert out.attrs == (Partial("sum"), Broadcast())

    on_a = _shard2((4, 8), Partial("sum"), Broadcast())
    on_b = _shard2((4, 8), Broadcast(), Partial("sum"))
    assert partial_reductions_by_axis(on_a) == ("sum", None)
    assert partial_reductions_by_axis(on_b) == (None, "sum")
    assert partial_reductions_by_axis(_shard2((4, 8), Partial("sum"), Partial("max"))) == (
        "sum",
        "max",
    )
    assert partial_reductions_by_axis(make_tensor_type((4, 8))) == ()


def test_an_input_whole_at_one_level_keeps_that_level_aligned():
    """Rows split over CTAs and threads, times a gamma every CTA holds whole.

    The gamma states the CTA level as all Broadcast. It must keep that level, so
    its thread split sits on the same mesh axis as the rows' thread split.
    """
    mesh = Mesh((Topology("cta", 2), Topology("thread", 4)), layout=(2, 4), names=("tile", "y"))
    rows = make_tensor_type(
        (2, 16),
        layout=ShardLayout(Layout((2, 16), (16, 1)), (Split(0), Split(1)), mesh),
    )
    gamma = make_tensor_type(
        (1, 16),
        layout=ShardLayout(Layout((1, 16), (16, 1)), (Broadcast(), Split(1)), mesh),
    )
    relation = (
        _access("{ [m, n] -> [m, n] }"),
        _access("{ [m, n] -> [0, n] }"),
        _access("{ [m, n] -> [m, n] }"),
    )

    out = derive_output_shard_layout((rows, gamma), relation, (2, 16))

    assert out.attrs == (Split(0), Split(1))
    assert out.mesh == mesh
