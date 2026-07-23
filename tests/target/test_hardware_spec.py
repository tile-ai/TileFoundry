from __future__ import annotations

from tilefoundry.target.hardware import format_capabilities, load_h200_sxm_sm90


def test_h200_capabilities_keep_grid_and_capacity_facts_distinct() -> None:
    spec = load_h200_sxm_sm90()
    facts = spec["facts"]

    assert facts["sm_count"]["value"] == 132
    assert facts["max_resident_ctas_per_sm"]["value"] == 32
    assert facts["compiler_policy_max_parallel_ctas"]["value"] == 132
    assert facts["shared_memory_per_sm"]["value"] == 228
    assert facts["shared_memory_per_cta"]["value"] == 227
    assert facts["registers_per_sm"]["value"] == 65_536
    assert facts["smem_bandwidth"]["provenance"] == "unavailable"
    assert "compiler_policy_max_parallel_ctas: 132 CTA [derived]" in format_capabilities(spec)
