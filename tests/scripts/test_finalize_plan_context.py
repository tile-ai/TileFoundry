from pathlib import Path

from scripts.finalize_plan_context import finalize_plan


def test_template_finalizer_keeps_milestone_and_final_gate_scoped(
    tmp_path: Path,
) -> None:
    template = Path(__file__).parents[2] / "docs/plans/TEMPLATE.md"
    canonical, after = finalize_plan(template, write=False)
    assert canonical == after

    plan = tmp_path / "plan.md"
    plan.write_text(canonical.replace("<path>", "src/example.py"))
    before, after = finalize_plan(plan, write=False)
    assert before != after
    assert after.count("Milestone MUST name a `#### Golden Reference`") == 2
    assert after.count("Verification MUST exercise the Golden Reference") == 2
    assert after.count("Touched tests and comments MUST be reviewed") == 2
    assert after.count("Milestone MUST classify public-contract impact") == 2
    assert after.count("Spec section MUST NOT reference plans") == 0
    assert after.count("No touched C++/CUDA files in this plan") == 1

    checked = after.replace(
        "- [ ] Milestone MUST name a `#### Golden Reference`",
        "- [x] Milestone MUST name a `#### Golden Reference`",
        1,
    )
    plan.write_text(checked)
    _, preserved = finalize_plan(plan, write=False)
    assert "- [x] Milestone MUST name a `#### Golden Reference`" in preserved
