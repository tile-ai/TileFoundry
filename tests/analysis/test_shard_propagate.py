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

from tilefoundry.ir.types import make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.visitor_registry.access_relation import AccessRelationResult
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import (
    derive_output_shard_layout,
    partial_reductions_by_axis,
)

_GPU = Mesh((Topology("gpu", 8),), Layout((8,), (1,)), names=("g",))
_GPU2 = Mesh((Topology("gpu", 4),), Layout((2, 2), (2, 1)), names=("a", "b"))


def _matmul_relation() -> AccessRelationResult:
    return AccessRelationResult(
        domain=build_domain((16, 8, 4)),  # M, N, K
        maps=(
            isl.map("{ [m, n, k] -> [m, k] }"),  # lhs
            isl.map("{ [m, n, k] -> [k, n] }"),  # rhs
            isl.map("{ [m, n, k] -> [m, n] }"),  # out
        ),
    )


def _elementwise_relation() -> AccessRelationResult:
    ident = isl.map("{ [m, n] -> [m, n] }")
    return AccessRelationResult(domain=build_domain((4, 8)), maps=(ident, ident, ident))


def _strides(shape) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _shard(shape, *attrs) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=_strides(shape)), attrs=attrs, mesh=_GPU
    )


def _shard2(shape, *attrs) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=_strides(shape)), attrs=attrs, mesh=_GPU2
    )


def test_a_split_of_a_surviving_axis_reaches_the_output():
    """Either operand may be the sharded one, and the axis it splits decides the
    output axis: rhs[K,N] split on N lands on out axis 1, lhs[M,K] split on M on
    out axis 0. With no sharded input at all there is nothing to propagate, which
    is ``None`` rather than an unsharded layout."""
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


#: What a split contraction dim leaves on the output, by whether the reduction over
#: it is partial. Partial: the Split becomes a mesh-axis value state with no layout
#: axis. Complete: every point ends up holding the whole result.
CONTRACTION_SPLITS = [
    pytest.param(
        # Both lhs[M,K] (K = layout axis 1) and rhs[K,N] (K = layout axis 0) split on K.
        _shard((4, 8), Split(0)), frozenset({2}), (Partial("sum"),), id="partial_reduction",
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


#: What propagation refuses rather than answering. The first is a conflict: lhs
#: splits M and rhs splits N on the SAME mesh axis, so no single output sharding
#: satisfies both. The other two are accesses that are not projections -- a rank-1
#: input read at ``m + n`` of a 2-D domain, then a surviving input Split whose
#: output is written at ``m + n``. Each must fail closed; deriving Broadcast instead
#: would silently drop the shard.
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
        AccessRelationResult(
            domain=build_domain((4, 8)),
            maps=(isl.map("{ [m, n] -> [m + n] }"), isl.map("{ [m, n] -> [m, n] }")),
        ),
        (4, 8),
        "non-projection access",
        id="split_on_non_projection_access",
    ),
    pytest.param(
        (make_tensor_type((4, 8), layout=_shard((4, 8), Split(0))),),
        AccessRelationResult(
            domain=build_domain((4, 8)),
            maps=(isl.map("{ [m, n] -> [m, n] }"), isl.map("{ [m, n] -> [m + n] }")),
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
    # lhs splits tensor axis 0 on mesh axis a, rhs splits tensor axis 0 on mesh
    # axis b -> the output factorizes axis 0 into two layout sub-positions (one
    # per mesh extent), each bound by its own mesh axis.
    lhs_t = make_tensor_type((4, 8), layout=_shard2((4, 8), Split(0), Broadcast()))
    rhs_t = make_tensor_type((4, 8), layout=_shard2((4, 8), Broadcast(), Split(0)))
    out = derive_output_shard_layout((lhs_t, rhs_t), _elementwise_relation(), (4, 8))
    # axis 0 (size 4) = mesh-a(2) x mesh-b(2); axis 1 (size 8) stays whole.
    assert out.layout.shape == (2, 2, 8)
    assert out.attrs == (Split(0), Split(1))
    assert out.mesh is _GPU2


def test_a_synthesised_layout_agrees_with_a_from_scratch_one():
    """``make_shard_tensor_type`` (a from-scratch sharding) and
    ``derive_output_shard_layout`` (a propagated one) both build a [shard §7.1.1](docs/spec/shard.md#711-layoutshape) layout through the shared ``canonical_shard_layout``, so for the same
    logical sharding they must compare equal -- otherwise a propagated value and
    an authored one describing the same distribution would need a reshard between
    them.

    Here neither operand alone realises the other's Split, so the carry branch
    fails for both and the output is synthesised: each mesh extent (2) is smaller
    than its axis (8), so each axis factors into an extent position and a residual
    (8 / 2 = 4).
    """
    lhs_t = make_tensor_type((8, 8), layout=_shard2((8, 8), Split(0), Broadcast()))
    rhs_t = make_tensor_type((8, 8), layout=_shard2((8, 8), Broadcast(), Split(1)))
    ident = isl.map("{ [m, n] -> [m, n] }")
    rel = AccessRelationResult(domain=build_domain((8, 8)), maps=(ident, ident, ident))

    out = derive_output_shard_layout((lhs_t, rhs_t), rel, (8, 8))

    expected = make_shard_tensor_type((8, 8), mesh=_GPU2, attrs=(Split(0), Split(1)))
    assert out == expected.layout


def test_an_input_partial_propagates_on_its_own_mesh_axis():
    """A Partial is a value state with no layout axis, so what identifies it is
    the mesh axis it sits on. It propagates through an elementwise identity rather
    than being dropped, and two different reductions on two axes stay two
    reductions -- collapsing them would make a sum-partial and a max-partial
    indistinguishable, which is a wrong result rather than a wrong layout.
    """
    ident = isl.map("{ [m, n] -> [m, n] }")
    rel = AccessRelationResult(domain=build_domain((4, 8)), maps=(ident, ident))
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
