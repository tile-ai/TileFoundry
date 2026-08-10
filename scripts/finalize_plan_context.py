#!/usr/bin/env python3
"""Finalize policy-generated regions in an authored plan.

Input is ``docs/plans/<name>.md``. hygiene: required CLI path template.
Related Files select milestone acceptance criteria and final-tree gates.
Handwritten content is preserved; misplaced markers or incomplete milestone
structure fail validation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from get_policy import filter_policies, load_policies  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "docs" / "policies" / "project-policy.json"

MILESTONE_AC_START = "<!-- policy_ac:start -->"
MILESTONE_AC_END = "<!-- policy_ac:end -->"
FINAL_GATE_START = "<!-- final_gate:start -->"
FINAL_GATE_END = "<!-- final_gate:end -->"


INLINE_POLICY_TAG_RE = re.compile(
    r"<!--\s+(?:policy_ac|policy_final|policy_rules|policy_knowledge):\s+[\w\-]+\s+-->"
)
POLICY_CHECK_RE = re.compile(r"^\s*-\s+\[([ xX])\].*<!--\s+policy_(?:ac|final):\s+([\w\-]+)\s+-->")
CODE_SPAN_RE = re.compile(r"`[^`]*`")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


class FinalizeError(Exception):
    """Report a validation failure clearly through a nonzero CLI exit."""


def _strip_inline_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text)


def _split_lines(text: str) -> list[str]:

    return text.split("\n")


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def _mask_fenced(lines: list[str]) -> list[str]:
    """The same lines with fenced blocks blanked, positions preserved.

    Structure is read off this copy, so a fenced line is content whatever it
    starts with. A plan quotes command output whose lines begin with `#`, and
    reading one as a heading would end the section it belongs to.
    """
    masked: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            masked.append("")
            continue
        masked.append("" if in_fence else line)
    return masked


def _heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else None


def _heading_text(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).rstrip()


def _find_section(
    lines: list[str], level: int, name: str, start: int = 0, end: int | None = None
) -> tuple[int, int] | None:
    """Find the bounds of a heading matching *level* and *name*.

    Matching is case-sensitive, with surrounding whitespace ignored. The
    result is ``(heading_index, body_end_exclusive)``. Body ends at the next
    heading with level <= *level* or at *end*.
    """
    if end is None:
        end = len(lines)
    for i in range(start, end):
        lvl = _heading_level(lines[i])
        if lvl == level and _heading_text(lines[i]) == name:
            j = i + 1
            while j < end:
                jl = _heading_level(lines[j])
                if jl is not None and jl <= level:
                    break
                j += 1
            return (i, j)
    return None


def _list_bullets(lines: list[str], start: int, end: int) -> list[str]:
    out = []
    for i in range(start, end):
        line = lines[i]
        m = re.match(r"^\s*-\s+(.*?)\s*$", line)
        if m:
            text = _strip_inline_comments(m.group(1)).strip()
            if text:
                out.append(text)
    return out


def _policy_check_states(lines: list[str], start: int, end: int) -> dict[str, bool]:
    """Return generated checklist completion keyed by its stable marker."""
    states: dict[str, bool] = {}
    for line in lines[start + 1 : end]:
        match = POLICY_CHECK_RE.match(line)
        if match:
            states[match.group(2)] = match.group(1).lower() == "x"
    return states


def _related_files_from_section(
    lines: list[str], section_start: int, section_end: int
) -> list[str]:
    """Collect bullets verbatim from a Related Files section.

    Empty and comment-only lines are skipped. This does not interpret
    ``inherit:`` — that
    is resolved at a higher level so callers can detect explicit
    inheritance vs. concrete paths.
    """
    return _list_bullets(lines, section_start + 1, section_end)


_PATH_FROM_BULLET_RE = re.compile(r"`([^`]+)`")


def _strip_path_bullet(item: str) -> str:
    """Extract the leading repository-relative path from a bullet.

    Related Files commonly uses ``- `<path>` — short description``
    so the matcher pulls the FIRST backtick-wrapped span; an item
    without backticks is taken whole (after whitespace strip).
    """
    s = item.strip()
    m = _PATH_FROM_BULLET_RE.match(s)
    if m:
        return m.group(1)
    return s


class PlanModel:
    def __init__(self, plan_path: Path) -> None:
        self.path = plan_path
        self.text = plan_path.read_text()
        self.lines = _split_lines(self.text)

        self.scan = _mask_fenced(self.lines)
        self._parse()

    def _parse(self) -> None:
        lines = self.scan

        self.plan_related_files: list[str] = []

        self.final_gate_start_idx = self._require_unique_line(FINAL_GATE_START)
        self.final_gate_end_idx = self._require_unique_line(FINAL_GATE_END)
        if self.final_gate_end_idx <= self.final_gate_start_idx:
            raise FinalizeError(f"{self.path}: `final_gate:end` precedes `final_gate:start`.")
        self.final_gate_states = _policy_check_states(
            lines, self.final_gate_start_idx, self.final_gate_end_idx
        )

        for name in ("Description", "Milestones", "Final Gate"):
            span = _find_section(lines, 2, name)
            if span is None:
                raise FinalizeError(f"{self.path}: missing required `## {name}` section.")

        milestones_span = _find_section(lines, 2, "Milestones")
        assert milestones_span is not None
        self.milestones: list[dict[str, Any]] = []
        i = milestones_span[0] + 1
        end = milestones_span[1]
        while i < end:
            lvl = _heading_level(lines[i])
            if lvl == 3 and lines[i].startswith("### Milestone "):
                ms_start = i

                j = i + 1
                while j < end:
                    jl = _heading_level(lines[j])
                    if jl is not None and jl <= 3:
                        break
                    j += 1
                self.milestones.append(self._parse_milestone(ms_start, j))
                i = j
            else:
                i += 1

        if not self.milestones:
            raise FinalizeError(
                f"{self.path}: `## Milestones` block contains no `### Milestone …` entries."
            )

        seen: set[str] = set()
        for m in self.milestones:
            for path in m["related_files"]:
                if path not in seen:
                    seen.add(path)
                    self.plan_related_files.append(path)

    def _require_unique_line(self, marker: str) -> int:
        idxs = [i for i, line in enumerate(self.scan) if line.strip() == marker]
        if not idxs:
            raise FinalizeError(
                f"{self.path}: marker {marker!r} missing — finalize_plan_context "
                "expects the template's marker pair."
            )
        if len(idxs) > 1:
            raise FinalizeError(
                f"{self.path}: marker {marker!r} appears {len(idxs)} times (expected exactly one)."
            )
        return idxs[0]

    def _parse_milestone(self, ms_start: int, ms_end: int) -> dict[str, Any]:
        lines = self.scan
        name = _heading_text(lines[ms_start])

        sections: dict[str, tuple[int, int]] = {}
        for required in (
            "Depends",
            "Target State Design",
            "Related Files",
            "Plan",
            "Acceptance Criteria",
        ):
            span = _find_section(lines, 4, required, ms_start + 1, ms_end)
            if span is None:
                raise FinalizeError(f"{self.path}: milestone {name!r} missing `#### {required}`.")

            body = [
                ln
                for ln in self.lines[span[0] + 1 : span[1]]
                if ln.strip()
                and ln.strip()
                not in (
                    MILESTONE_AC_START,
                    MILESTONE_AC_END,
                )
            ]
            if not body:
                raise FinalizeError(f"{self.path}: milestone {name!r} has empty `#### {required}`.")
            sections[required] = span

        related = sections["Related Files"]
        rel_items = _related_files_from_section(lines, related[0], related[1])
        effective_paths: list[str] = []
        for item in rel_items:
            if item.lower() == "inherit: top-level":
                raise FinalizeError(
                    f"{self.path}: milestone {name!r} says `inherit: top-level`, "
                    "but there is no plan-level `Related Files` to inherit from -- "
                    "the plan's touch surface is the union of its milestones'. "
                    "List the paths this milestone touches."
                )
            effective_paths.append(_strip_path_bullet(item))

        ac_section = sections["Acceptance Criteria"]

        ac_start = ac_end = None
        for k in range(ac_section[0] + 1, ac_section[1]):
            t = lines[k].strip()
            if t == MILESTONE_AC_START:
                ac_start = k
            elif t == MILESTONE_AC_END:
                ac_end = k
        if ac_start is None or ac_end is None:
            raise FinalizeError(
                f"{self.path}: milestone {name!r} is missing the "
                f"`policy_ac:start`/`policy_ac:end` marker pair inside "
                f"`#### Acceptance Criteria`."
            )
        if ac_end <= ac_start:
            raise FinalizeError(
                f"{self.path}: milestone {name!r}: `policy_ac:end` precedes `policy_ac:start`."
            )

        return {
            "name": name,
            "related_files": effective_paths,
            "ac_section": ac_section,
            "policy_ac_start_idx": ac_start,
            "policy_ac_end_idx": ac_end,
            "policy_states": _policy_check_states(lines, ac_start, ac_end),
        }


def _refs_phrase(refs: list[dict[str, str]]) -> str:
    parts = [f"`{r['path']} § {r['section']}`" for r in refs]
    return ", ".join(parts)


def render_policy_ac_body(matched: list[dict[str, Any]], states: dict[str, bool]) -> list[str]:
    items: list[str] = []
    for p in matched:
        for n, ac in enumerate(p.get("ac") or []):
            marker = f"{p['id']}-{n}"
            check = "x" if states.get(marker, False) else " "
            items.append(f"- [{check}] {ac} <!-- policy_ac: {marker} -->")
    return items


def render_final_gate_body(
    matched: list[dict[str, Any]], policies: list[dict[str, Any]], states: dict[str, bool]
) -> list[str]:
    items: list[str] = []
    for p in matched:
        for n, ac in enumerate(p.get("final_ac") or []):
            marker = f"{p['id']}-{n}"
            check = "x" if states.get(marker, False) else " "
            items.append(f"- [{check}] {ac} <!-- policy_final: {marker} -->")
    cf = next((p for p in policies if p.get("id") == "clang_format"), None)
    if cf is not None and cf not in matched:
        marker = "clang_format-na"
        check = "x" if states.get(marker, False) else " "
        items.append(
            f"- [{check}] No touched C++/CUDA files in this plan — clang-format "
            f"gate N/A <!-- policy_final: {marker} -->"
        )
    return items


def _replace_range(lines: list[str], start_idx: int, end_idx: int, body: list[str]) -> list[str]:
    """Replace a range's contents while preserving its marker lines.

    The replaced slice is `lines[start_idx + 1 : end_idx]`.
    """
    return lines[: start_idx + 1] + body + lines[end_idx:]


def finalize_plan(
    plan_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    role: str = "implementer",
    write: bool = True,
) -> tuple[str, str]:
    """Rewrite *plan_path* to canonical form.

    Returns the ``(before, after)`` text pair.

    When *write* is False the plan file is not modified — useful for
    dry-runs in tests and for the ``--check`` mode.
    """
    plan = PlanModel(plan_path)
    policies = load_policies(policy_path)

    plan_matched = filter_policies(policies, plan.plan_related_files)

    rewrites: list[tuple[int, int, list[str]]] = []
    for m in plan.milestones:
        matched = filter_policies(policies, m["related_files"])
        body = render_policy_ac_body(matched, m["policy_states"])
        rewrites.append((m["policy_ac_start_idx"], m["policy_ac_end_idx"], body))
    rewrites.append(
        (
            plan.final_gate_start_idx,
            plan.final_gate_end_idx,
            render_final_gate_body(plan_matched, policies, plan.final_gate_states),
        )
    )
    rewrites.sort(key=lambda r: r[0], reverse=True)

    new_lines = list(plan.lines)
    for start, end, body in rewrites:
        new_lines = _replace_range(new_lines, start, end, body)

    allowed: list[tuple[int, int]] = []
    final_gate_start = next(
        i for i, line in enumerate(new_lines) if line.strip() == FINAL_GATE_START
    )
    final_gate_end = next(i for i, line in enumerate(new_lines) if line.strip() == FINAL_GATE_END)
    allowed.append((final_gate_start, final_gate_end))
    ac_starts = [i for i, line in enumerate(new_lines) if line.strip() == MILESTONE_AC_START]
    ac_ends = [i for i, line in enumerate(new_lines) if line.strip() == MILESTONE_AC_END]
    if len(ac_starts) != len(ac_ends):
        raise FinalizeError(
            f"{plan_path}: unbalanced `policy_ac` marker pairs after rewrite "
            f"(starts={len(ac_starts)}, ends={len(ac_ends)})."
        )
    for s, e in zip(sorted(ac_starts), sorted(ac_ends)):
        allowed.append((s, e))

    diagnostics: list[str] = []
    in_fence = False
    for i, line in enumerate(new_lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped_line = CODE_SPAN_RE.sub("", line)
        if not INLINE_POLICY_TAG_RE.search(stripped_line):
            continue
        if any(lo < i < hi for lo, hi in allowed):
            continue
        diagnostics.append(
            f"{plan_path}: line {i + 1}: stray policy marker outside any "
            f"allowed range: {line.strip()!r}"
        )
    if diagnostics:
        raise FinalizeError("\n".join(diagnostics))

    before = plan.text
    after = _join_lines(new_lines)

    if before.endswith("\n") and not after.endswith("\n"):
        after = after + "\n"
    if write and after != before:
        plan_path.write_text(after)
    return before, after


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("plan", type=Path, help="Path to docs/plans/<name>.md")
    p.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help=f"Policy JSON path (default: {DEFAULT_POLICY}).",
    )
    p.add_argument(
        "--role",
        choices=("implementer", "reviewer"),
        default="implementer",
        help="Role used to filter rules / knowledge refs for the "
        "plan-level Preflight block (default: implementer).",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate only; exit non-zero if a rewrite would change the file.",
    )
    args = p.parse_args(argv)

    try:
        before, after = finalize_plan(
            args.plan,
            policy_path=args.policy,
            role=args.role,
            write=not args.check,
        )
    except FinalizeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if args.check and before != after:
        sys.stderr.write(
            f"{args.plan}: plan is not in canonical form. "
            "Run `scripts/finalize_plan_context.py` to rewrite.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
