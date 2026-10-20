"""SM80 register fragments compared with CuTe's MMA traits."""

from tilefoundry.dsl import T

from ._cute_reference import cute_reference, reference_tv, to_tv_layout


def test_mma_atom():
    """Each operand holds the same elements on the same threads as CuTe."""
    reference = cute_reference()["mma"]
    failures = []
    for role in ("A", "B", "C"):
        ours = to_tv_layout(T.cuda.sm80.Mma(), role)
        theirs = reference_tv(reference[role])
        if ours != theirs:
            failures.append(f"mma {role}: {ours} != {theirs}")
    assert not failures, "\n".join(failures)
