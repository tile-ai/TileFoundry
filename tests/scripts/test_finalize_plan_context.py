from pathlib import Path

import pytest

from scripts.finalize_plan_context import FinalizeError, finalize_plan, main


def _write_plan(
    tmp_path: Path,
    *,
    related_files: list[str] | None = None,
    spec_impact: list[str] | None = None,
    include_spec_impact: bool = True,
) -> Path:
    related_files = related_files or ["src/tilefoundry/example.py"]
    spec_impact = spec_impact or ["N/A: internal behavior-preserving change"]
    related = "\n".join(f"- `{path}`" for path in related_files)
    impact = "\n".join(f"- {item}" for item in spec_impact)
    impact_section = f"\n#### Spec Impact\n{impact}\n" if include_spec_impact else ""
    plan = tmp_path / "plan.md"
    plan.write_text(
        f"""---
type: META
component: test
target_repo: tilefoundry
---

# [META][test] Finalizer fixture

## Description

### Symptom / Motivation

Test fixture.

### Root Cause Analysis

Test fixture.

### Related Files

{related}

## Goal

Exercise plan validation.

## Constraints

- Keep the fixture minimal.

## Milestones

### Milestone M0: Validate fixture

#### Depends

- None

#### Related Files

{related}
{impact_section}
#### Plan

- [ ] Exercise validation.

#### Acceptance Criteria

- [ ] The fixture is accepted or rejected as expected.
<!-- policy_ac:start -->
<!-- policy_ac:end -->

## Execution Preflight

<!-- policy_preflight:start -->
<!-- policy_preflight:end -->
"""
    )
    return plan


def test_accepts_reasoned_na(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)

    finalize_plan(plan, write=False)


def test_accepts_scoped_spec_paths(tmp_path: Path) -> None:
    paths = ["docs/spec/core-ir.md", "docs/spec/types.md"]
    plan = _write_plan(
        tmp_path,
        related_files=["src/tilefoundry/example.py", *paths],
        spec_impact=[f"`{path}`" for path in paths],
    )

    finalize_plan(plan, write=False)


def test_requires_spec_impact_with_milestone_name(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, include_spec_impact=False)

    with pytest.raises(
        FinalizeError,
        match=r"Milestone M0: Validate fixture.*missing `#### Spec Impact`",
    ):
        finalize_plan(plan, write=False)


@pytest.mark.parametrize(
    ("spec_impact", "message"),
    [
        (
            ["N/A: internal change", "`docs/spec/types.md`"],
            "cannot mix an `N/A:` entry with spec paths",
        ),
        (["N/A:"], "must use one reasoned `N/A: <reason>` entry"),
        (
            ["`docs/develop.md`"],
            "must be a `docs/spec/\\*\\.md` path",
        ),
    ],
)
def test_rejects_malformed_spec_impact(
    tmp_path: Path,
    spec_impact: list[str],
    message: str,
) -> None:
    plan = _write_plan(
        tmp_path,
        related_files=["src/tilefoundry/example.py", "docs/spec/types.md"],
        spec_impact=spec_impact,
    )

    with pytest.raises(FinalizeError, match=message):
        finalize_plan(plan, write=False)


def test_rejects_spec_path_absent_from_related_files(tmp_path: Path) -> None:
    plan = _write_plan(
        tmp_path,
        spec_impact=["`docs/spec/types.md`"],
    )

    with pytest.raises(
        FinalizeError,
        match="must also appear in the milestone's effective `#### Related Files`",
    ):
        finalize_plan(plan, write=False)


def test_finalization_remains_idempotent(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)

    assert main([str(plan)]) == 0
    assert main(["--check", str(plan)]) == 0
