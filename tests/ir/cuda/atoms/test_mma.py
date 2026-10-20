"""SM80 register fragments compared with CuTe's MMA traits."""

from tilefoundry.dsl import T

from ._cute_reference import cute_reference, reference_tv, to_tv_layout
from ._operand_layout import (
    _assert_reshard_typeinfer_ok,
    _bound_fragment,
    _check_split_extents_match_mesh,
)


def test_mma_atom():
    """Each operand holds the same elements on the same threads as CuTe."""
    reference = cute_reference()["mma"]
    failures = []
    for role in ("A", "B", "C"):
        atom = T.cuda.sm80.Mma()
        shard = _bound_fragment(atom, role)
        _check_split_extents_match_mesh(shard)
        shape = atom.operand_shapes()[("C", "A", "B").index(role)]
        _assert_reshard_typeinfer_ok(shape, "f32" if role == "C" else "bf16", shard)
        ours = to_tv_layout(atom, role)
        theirs = reference_tv(reference[role])
        if ours != theirs:
            failures.append(f"mma {role}: {ours} != {theirs}")
    assert not failures, "\n".join(failures)
