"""Exact analysis values for programs small enough to compute on paper."""

from __future__ import annotations

from tests.fixtures.placed.hand_checked import InvariantReuse
from tilefoundry.analysis import analyze
from tilefoundry.analysis.report import report_data


def test_invariant_reuse_matches_the_written_arithmetic() -> None:
    result = analyze(
        InvariantReuse,
        InvariantReuse.entry_function(),
        analysis=("memory",),
    )
    data = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )

    traffic = data["function_records"]["memory"]["traffic"]["storage"]["gmem"]
    assert traffic["logical"] == {"read": 64, "write": 0}
    assert traffic["total"] == {"read": 192, "write": 0}
    assert traffic["per_unit"] == [{"read": 192, "write": 0}]
