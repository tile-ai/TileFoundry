from pathlib import Path

import pytest

from scripts.finalize_plan_context import FinalizeError, finalize_plan


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
    assert after.count("#### Target State Design") == 2
    assert after.count("Touched tests and comments MUST be reviewed") == 2
    assert after.count("MUST list the owning `docs/spec/*.md` path") == 2
    assert after.count("Spec section MUST NOT reference plans") == 0
    assert after.count("No touched C++/CUDA files in this plan") == 1

    legacy = after.replace(
        "- [ ] Touched tests and comments MUST be reviewed",
        "- [x] Touched tests and comments MUST be reviewed",
        1,
    ).replace(
        "<!-- policy_ac: milestone_review-0 -->",
        "<!-- policy_ac: milestone_review-0 -->\n"
        "- [ ] Retired policy check. <!-- policy_ac: milestone_review-1 -->\n"
        "- [ ] Retired policy check. <!-- policy_ac: milestone_review-2 -->",
        1,
    )
    plan.write_text(legacy)
    _, normalized = finalize_plan(plan, write=False)
    assert "- [x] Touched tests and comments MUST be reviewed" in normalized
    assert "milestone_review-1" not in normalized
    assert "milestone_review-2" not in normalized


def test_a_fenced_comment_line_is_quoted_output_not_a_heading(tmp_path: Path) -> None:
    """A plan quotes analyze output, whose lines begin with `#`."""
    template = Path(__file__).parents[2] / "docs/plans/TEMPLATE.md"
    canonical, _ = finalize_plan(template, write=False)
    quoted = canonical.replace(
        "## Description",
        "## Description\n\n"
        "```console\n"
        "$ tilefoundry analyze model.py:Model.layer\n"
        "# traffic gmem=r3322101984/w100868096\n"
        "```",
        1,
    ).replace("<path>", "src/example.py")

    plan = tmp_path / "quoting.md"
    plan.write_text(quoted)
    _, after = finalize_plan(plan, write=False)
    assert after.count("#### Target State Design") == 2
    assert "# traffic gmem=r3322101984/w100868096" in after


def test_every_milestone_requires_a_target_state_design(
    tmp_path: Path,
) -> None:
    template = Path(__file__).parents[2] / "docs/plans/TEMPLATE.md"
    canonical, _ = finalize_plan(template, write=False)
    retired_heading = "Golden" + " Reference"
    invalid = canonical.replace("<path>", "src/example.py").replace(
        "#### Target State Design",
        f"#### {retired_heading}",
        1,
    )

    plan = tmp_path / "missing-target-state-shape.md"
    plan.write_text(invalid)
    with pytest.raises(FinalizeError, match="missing `#### Target State Design`"):
        finalize_plan(plan, write=False)


def test_a_code_only_target_state_design_is_not_empty(tmp_path: Path) -> None:
    template = Path(__file__).parents[2] / "docs/plans/TEMPLATE.md"
    canonical, _ = finalize_plan(template, write=False)
    code_only = canonical.replace(
        "#### Target State Design\n"
        "<!-- Show every part this milestone designs in its delivered form. Use code or\n"
        "     compact pseudocode. -->\n"
        "```python\n"
        "# <delivered code shape>\n"
        "```",
        "#### Target State Design\n```python\nvalue = 1\n```",
        1,
    ).replace("<path>", "src/example.py")

    plan = tmp_path / "code-only-target-state-design.md"
    plan.write_text(code_only)
    finalize_plan(plan, write=False)
