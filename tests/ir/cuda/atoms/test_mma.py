"""Warp MMA register fragments compared with CuTe's MMA traits."""

import pytest

from tilefoundry.dsl import T

from ._cute_reference import (
    cute_reference,
    reference_threads,
    reference_tv,
    to_thread_layout,
    to_tv_layout,
)
from ._operand_layout import (
    _assert_reshard_typeinfer_ok,
    _bound_fragment,
    _check_split_extents_match_mesh,
)


@pytest.mark.parametrize(
    ("declaration", "operand_dtype"),
    [(T.cuda.sm80.Mma, "bf16"), (T.cuda.sm89.Mma, "fp8e4m3")],
    ids=["sm80", "sm89"],
)
def test_mma_atom(declaration, operand_dtype):
    """Each operand holds the same elements on the same threads as CuTe."""
    reference = cute_reference()[f"mma-{operand_dtype}"]
    failures = []
    for role in ("A", "B", "C"):
        atom = declaration()
        assert to_thread_layout(atom) == reference_threads(reference["ThrID"])
        shard = _bound_fragment(atom, role)
        _check_split_extents_match_mesh(shard)
        shape = atom.operand_shapes()[("C", "A", "B").index(role)]
        _assert_reshard_typeinfer_ok(shape, "f32" if role == "C" else operand_dtype, shard)
        ours = to_tv_layout(atom, role)
        theirs = reference_tv(reference[role])
        if ours != theirs:
            failures.append(f"mma {role}: {ours} != {theirs}")
    assert not failures, "\n".join(failures)
