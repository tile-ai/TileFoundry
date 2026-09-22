"""Exact analysis values for programs small enough to compute on paper."""

from __future__ import annotations

from tests.fixtures.placed.hand_checked import InvariantReuse
from tilefoundry.analysis import analyze
from tilefoundry.analysis.report import report_data


def _checked(data: dict) -> dict:
    """Keep the movement and placement conclusions this fixture hand-checks."""
    function = data["function_records"]["memory"]
    return {
        "loops": data["loops"],
        "calls": data["calls"],
        "function_memory": {
            "topologies": function["topologies"],
            "traffic": function["traffic"],
            "footprint": function["footprint"],
            "peaks": function["peaks"],
            "solver_status": function["solver_status"],
            "errors": function["errors"],
            "advisories": function["advisories"],
        },
    }


INVARIANT_REUSE = {
    "loops": [],
    "calls": [
        {
            "value": "v0",
            "memory": {
                "topologies": ["cta"],
                "traffic": {
                    "storage": {
                        "smem": {
                            "logical": {"read": 0, "write": 16},
                            "total": {"read": 0, "write": 16},
                            "per_unit": [{"read": 0, "write": 16}],
                        }
                    },
                    "communication": {},
                },
                "operands": [
                    {
                        "arg": "result",
                        "name": "v0",
                        "type": "bf16[4,2] smem",
                        "read": 0,
                        "write": 16,
                    }
                ],
                "footprint": None,
            },
        },
        {
            "value": "v1",
            "memory": {
                "topologies": ["cta"],
                "traffic": {
                    "storage": {
                        "rmem": {
                            "logical": {"read": 16, "write": 0},
                            "total": {"read": 16, "write": 0},
                            "per_unit": [{"read": 16, "write": 0}],
                        }
                    },
                    "communication": {},
                },
                "operands": [
                    {
                        "arg": 0,
                        "name": "x",
                        "type": "bf16[8,4] gmem",
                        "read": 0,
                        "write": 0,
                    },
                    {
                        "arg": 1,
                        "name": "tuple",
                        "type": "i64[]+ umat",
                        "read": 16,
                        "write": 0,
                    },
                    {
                        "arg": "result",
                        "name": "v1",
                        "type": "bf16[4,2] gmem",
                        "read": 0,
                        "write": 0,
                    },
                ],
                "footprint": None,
            },
        },
        {
            "value": "v2",
            "memory": {
                "topologies": ["cta"],
                "traffic": {
                    "storage": {
                        "gmem": {
                            "logical": {"read": 16, "write": 0},
                            "total": {"read": 16, "write": 0},
                            "per_unit": [{"read": 16, "write": 0}],
                        },
                        "smem": {
                            "logical": {"read": 0, "write": 16},
                            "total": {"read": 0, "write": 16},
                            "per_unit": [{"read": 0, "write": 16}],
                        },
                    },
                    "communication": {},
                },
                "operands": [
                    {
                        "arg": 0,
                        "name": "v1",
                        "type": "bf16[4,2] gmem",
                        "read": 16,
                        "write": 0,
                    },
                    {
                        "arg": "result",
                        "name": "v2",
                        "type": "bf16[4,2] smem",
                        "read": 0,
                        "write": 16,
                    },
                ],
                "footprint": None,
            },
        },
    ],
    "function_memory": {
        "topologies": ["cta"],
        "traffic": {
            "storage": {
                "gmem": {
                    "logical": {"read": 64, "write": 0},
                    "total": {"read": 192, "write": 0},
                    "per_unit": [{"read": 192, "write": 0}],
                },
                "rmem": {
                    "logical": {"read": 64, "write": 0},
                    "total": {"read": 192, "write": 0},
                    "per_unit": [{"read": 192, "write": 0}],
                },
                "smem": {
                    "logical": {"read": 0, "write": 80},
                    "total": {"read": 0, "write": 208},
                    "per_unit": [{"read": 0, "write": 208}],
                },
            },
            "communication": {},
        },
        "footprint": None,
        "peaks": [
            {
                "memory_level": "gmem",
                "peak_bytes": 80,
                "persistent_bytes": 64,
                "capacity_bytes": 141_000_000_000,
            },
            {
                "memory_level": "rmem",
                "peak_bytes": 0,
                "persistent_bytes": 0,
                "capacity_bytes": 262_144,
            },
            {
                "memory_level": "smem",
                "peak_bytes": 64,
                "persistent_bytes": 0,
                "capacity_bytes": 232_448,
            },
        ],
        "solver_status": "feasible",
        "errors": [],
        "advisories": [],
    },
}


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

    assert _checked(data) == INVARIANT_REUSE
