"""Generic shard propagation engine over a matmul-shaped relation.

Domain dims: 0=M, 1=N, 2=K. lhs[M,K], rhs[K,N], out[M,N] (K reduced).
"""
from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.types import make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.visitor_registry.access_relation import AccessRelationResult
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

_GPU = Mesh(Topology("gpu", 8), (8,), names=("g",))


def _matmul_relation() -> AccessRelationResult:
    return AccessRelationResult(
        domain=build_domain((16, 8, 4)),  # M, N, K
        maps=(
            isl.map("{ [m, n, k] -> [m, k] }"),  # lhs
            isl.map("{ [m, n, k] -> [k, n] }"),  # rhs
            isl.map("{ [m, n, k] -> [m, n] }"),  # out
        ),
    )


def _shard(shape, *attrs) -> ShardLayout:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return ShardLayout(layout=Layout(shape=shape, strides=tuple(strides)), attrs=attrs, mesh=_GPU)


def test_rhs_n_split_to_output_split():
    # rhs[K,N] split on N (layout axis 1) -> output Split on N (out axis 1).
    rhs_t = make_tensor_type((4, 8), layout=_shard((4, 8), Split(1)))
    out = derive_output_shard_layout(
        (make_tensor_type((16, 4)), rhs_t), _matmul_relation(), (16, 8)
    )
    assert out.attrs == (Split(1),)


def test_lhs_m_split_to_output_split():
    lhs_t = make_tensor_type((16, 4), layout=_shard((16, 4), Split(0)))
    out = derive_output_shard_layout(
        (lhs_t, make_tensor_type((4, 8))), _matmul_relation(), (16, 8)
    )
    assert out.attrs == (Split(0),)


def test_k_split_to_partial():
    # Both lhs[M,K] (K = layout axis 1) and rhs[K,N] (K = layout axis 0) split on K
    # -> the Split of the contraction dim becomes a mesh-axis Partial value
    # state on that mesh axis (no layout axis).
    lhs_t = make_tensor_type((16, 4), layout=_shard((16, 4), Split(1)))
    rhs_t = make_tensor_type((4, 8), layout=_shard((4, 8), Split(0)))
    out = derive_output_shard_layout(
        (lhs_t, rhs_t), _matmul_relation(), (16, 8), partial_reduction_dims=frozenset({2})
    )
    assert out.attrs == (Partial("sum"),)


def test_k_split_complete_to_broadcast():
    # K split but reduction effect is complete (K not in partial set) -> Broadcast.
    lhs_t = make_tensor_type((16, 4), layout=_shard((16, 4), Split(1)))
    out = derive_output_shard_layout(
        (lhs_t, make_tensor_type((4, 8))), _matmul_relation(), (16, 8), partial_reduction_dims=frozenset()
    )
    assert out.attrs == (Broadcast(),)


def test_no_sharded_input_returns_none():
    out = derive_output_shard_layout((make_tensor_type((16, 4)), make_tensor_type((4, 8))), _matmul_relation(), (16, 8))
    assert out is None


def test_incompatible_split_errors():
    # lhs splits M, rhs splits N on the SAME mesh axis -> conflict.
    lhs_t = make_tensor_type((16, 4), layout=_shard((16, 4), Split(0)))
    rhs_t = make_tensor_type((4, 8), layout=_shard((4, 8), Split(1)))
    with pytest.raises(ValueError, match="incompatible output shard"):
        derive_output_shard_layout((lhs_t, rhs_t), _matmul_relation(), (16, 8))


_GPU2 = Mesh(Topology("gpu", 4), (2, 2), names=("a", "b"))


def _shard2(shape, *attrs) -> ShardLayout:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return ShardLayout(layout=Layout(shape=shape, strides=tuple(strides)), attrs=attrs, mesh=_GPU2)


def _elementwise_relation() -> AccessRelationResult:
    ident = isl.map("{ [m, n] -> [m, n] }")
    return AccessRelationResult(domain=build_domain((4, 8)), maps=(ident, ident, ident))


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


def test_carry_candidates_disagree_falls_through_to_synthesis():
    # Two full-shape inputs realise the same logical sharding (axis 0 split on
    # both mesh axes) with different layout factorizations: lhs (2,2,2), rhs
    # (2,4). The carry branch must not arbitrarily pick the first operand; it
    # falls through to the canonical synthesis, which is order-independent.
    lhs = make_tensor_type((8,), layout=_shard2((2, 2, 2), Split(0), Split(1)))
    rhs = make_tensor_type((8,), layout=_shard2((2, 4), Split(0), Split(1)))
    ident = isl.map("{ [m] -> [m] }")
    rel = AccessRelationResult(domain=build_domain((8,)), maps=(ident, ident, ident))
    out_lr = derive_output_shard_layout((lhs, rhs), rel, (8,))
    out_rl = derive_output_shard_layout((rhs, lhs), rel, (8,))
    # canonical: axis 0 (8) = mesh-a(2) x mesh-b(2) x remainder(2).
    assert out_lr.layout.shape == (2, 2, 2)
    assert out_lr.attrs == (Split(0), Split(1))
    # order-independent.
    assert out_rl.layout.shape == out_lr.layout.shape
    assert out_rl.attrs == out_lr.attrs


def test_split_on_non_projection_access_errors():
    # Input is rank-1 and accesses (m + n) of a 2-D domain — not a projection;
    # a Split on it must fail closed rather than silently drop as broadcast.
    rel = AccessRelationResult(
        domain=build_domain((4, 8)),
        maps=(isl.map("{ [m, n] -> [m + n] }"), isl.map("{ [m, n] -> [m, n] }")),
    )
    x_t = make_tensor_type((12,), layout=_shard((12,), Split(0)))
    with pytest.raises(ValueError, match="non-projection access"):
        derive_output_shard_layout((x_t,), rel, (4, 8))


def test_split_surviving_via_complex_output_errors():
    # Input Split(m) survives in the output, but the output accesses (m + n),
    # so the output layout axis is underivable — fail closed, not Broadcast.
    rel = AccessRelationResult(
        domain=build_domain((4, 8)),
        maps=(isl.map("{ [m, n] -> [m, n] }"), isl.map("{ [m, n] -> [m + n] }")),
    )
    x_t = make_tensor_type((4, 8), layout=_shard((4, 8), Split(0)))
    with pytest.raises(ValueError, match="non-projection output access"):
        derive_output_shard_layout((x_t,), rel, (12,))


def test_input_partial_propagates_to_output_partial():
    # Elementwise identity: input Partial("sum") propagates, not dropped.
    ident = isl.map("{ [m, n] -> [m, n] }")
    rel = AccessRelationResult(domain=build_domain((4, 8)), maps=(ident, ident))
    # 2-axis mesh: a Partial value state on mesh axis 0 carries to the output on
    # the same mesh axis (it has no layout axis).
    x_t = make_tensor_type((4, 8), layout=_shard2((4, 8), Partial("sum"), Broadcast()))
    out = derive_output_shard_layout((x_t,), rel, (4, 8))
    assert out.attrs == (Partial("sum"), Broadcast())


def test_replicated_input_on_other_mesh_is_ignored():
    # lhs is all-Broadcast (replicated) on a different mesh; it pins no mesh and
    # contributes no sharding. The output takes the genuinely-sharded rhs's
    # mesh and shard — no cross-mesh error.
    lhs_t = make_tensor_type((16, 4), layout=_shard2((16, 4), Broadcast(), Broadcast()))
    rhs_t = make_tensor_type((4, 8), layout=_shard((4, 8), Split(1)))
    out = derive_output_shard_layout(
        (lhs_t, rhs_t), _matmul_relation(), (16, 8)
    )
    assert out.mesh is _GPU
    assert out.attrs == (Split(1),)
