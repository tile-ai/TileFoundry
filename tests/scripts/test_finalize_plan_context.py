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
    assert after.count("The gate request MUST show the Golden Reference") == 2
    assert after.count("Touched tests and comments MUST be reviewed") == 2
    assert after.count("MUST list the owning `docs/spec/*.md` path") == 2
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
    assert after.count("Milestone MUST name a `#### Golden Reference`") == 2
    assert "# traffic gmem=r3322101984/w100868096" in after
