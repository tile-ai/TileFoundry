"""Physical-placement proofs that do not require a complete HIR program."""

import isl

from tilefoundry.analysis.allocation import covers_result


def test_union_coverage_does_not_prove_complete_writes_per_iteration() -> None:
    """Complementary partial writes cannot justify destructive result aliasing."""
    outputs = isl.map(
        "{ [iteration, element] -> [element] : "
        "0 <= iteration <= 1 and "
        "((iteration = 0 and 0 <= element <= 1) or "
        "(iteration = 1 and 2 <= element <= 3)) }"
    )
    result_box = isl.set("{ [element] : 0 <= element <= 3 }")
    iterations = isl.set("{ [iteration] : 0 <= iteration <= 1 }")
    full_result = isl.set(
        "{ [iteration, element] : "
        "0 <= iteration <= 1 and 0 <= element <= 3 }"
    )

    assert outputs.range().is_equal(result_box)
    assert not covers_result(outputs.domain(), full_result, outputs, iterations, result_box)
