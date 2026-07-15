"""`mesh_scope_matches_required_scope` — structural thread-participation match.

The predicate underlies the fragment use-point check and ``T.mma``
verify. It matches on thread participation (program level + static lane count +
exact required thread-value layout shape+strides), independent of binding-var
names, axis names, and mesh object identity.
"""
from __future__ import annotations

from tilefoundry.ir.tir.cuda.nn.mma import SM80_16x8x16_F32BF16BF16F32_TN, make_atom
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.ir.types.shard.scope_match import mesh_scope_matches_required_scope

# The required thread scope is read off the realized atom, not a separate export.
_REQ = make_atom(SM80_16x8x16_F32BF16BF16F32_TN).required_scope  # thread(32), (4,8)/(1,4)


def _thread(shape, strides, *, size=32, names=()):
    return Mesh(
        topology=Topology("thread", size),
        layout=Layout(shape=shape, strides=strides),
        names=names,
    )


def test_exact_4x8_warp_matches() -> None:
    assert mesh_scope_matches_required_scope(_thread((4, 8), (1, 4)), _REQ)


def test_flat_32_lane_scope_rejected() -> None:
    # A flat (32,) scope cannot host the fragment's 2-axis (4,8) Split layout.
    assert not mesh_scope_matches_required_scope(_thread((32,), (1,)), _REQ)


def test_axis_names_irrelevant() -> None:
    # Axis names are not part of the thread-value decomposition.
    assert mesh_scope_matches_required_scope(
        _thread((4, 8), (1, 4), names=("warp", "lane")), _REQ
    )


def test_wrong_strides_rejected() -> None:
    # Right shape, wrong lane order (row-major (8,1) vs required (1,4)).
    assert not mesh_scope_matches_required_scope(_thread((4, 8), (8, 1)), _REQ)


def test_cta_scope_rejected() -> None:
    cta = Mesh(topology=Topology("cta", 32), layout=Layout(shape=(4, 8), strides=(1, 4)))
    assert not mesh_scope_matches_required_scope(cta, _REQ)


def test_wrong_thread_count_rejected() -> None:
    assert not mesh_scope_matches_required_scope(_thread((8, 8), (1, 8), size=64), _REQ)


def test_inconsistent_mesh_rejected() -> None:
    # thread(64) topology but a 32-element layout — malformed.
    assert not mesh_scope_matches_required_scope(_thread((4, 8), (1, 4), size=64), _REQ)


def test_noncontiguous_layout_rejected() -> None:
    # Right shape but gapped strides (1,8) != required (1,4) — exact mismatch.
    assert not mesh_scope_matches_required_scope(_thread((4, 8), (1, 8)), _REQ)
