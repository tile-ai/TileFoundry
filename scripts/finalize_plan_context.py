#!/usr/bin/env python3
"""Plan finalizer for `docs/plans/<name>.md`.

Reads a plan written against `docs/plans/TEMPLATE.md`, matches the
plan-level and per-milestone ``Related Files`` against
`docs/policies/project-policy.json`, and rewrites two kinds of
generated regions:

- the plan-level ``<!-- policy_preflight:start --> ... <!-- policy_preflight:end -->``
  block carries the plan-wide ``### Policy Rules & Knowledge``
  subsection (refs-only — never the actual rule / knowledge text);
- each milestone's ``<!-- policy_ac:start --> ... <!-- policy_ac:end -->``
  range carries the policy ACs whose ``when.path_glob`` matches that
  milestone's ``#### Related Files`` (with explicit
  ``- inherit: top-level`` fallback).

The finalizer never touches handwritten content outside the marker
ranges. A ``policy_*`` HTML comment appearing outside any allowed
range is a hard validation failure. Every milestone must also declare
its public-contract impact in ``Spec Impact`` as either owning
``docs/spec/*.md`` paths or one reasoned ``N/A:`` entry.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# Allow ``from scripts.get_policy import ...`` when invoked from the
# repo root via either ``python scripts/finalize_plan_context.py`` or
# the test suite's ``import scripts.finalize_plan_context``.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from get_policy import filter_policies, load_policies  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "docs" / "policies" / "project-policy.json"

PREFLIGHT_START = "<!-- policy_preflight:start -->"
PREFLIGHT_END = "<!-- policy_preflight:end -->"
MILESTONE_AC_START = "<!-- policy_ac:start -->"
MILESTONE_AC_END = "<!-- policy_ac:end -->"

# Any ``<!-- policy_(ac|rules|knowledge): <id> -->`` tag. The
# whitespace AFTER the colon is the rendered convention; the marker
# pair sentinels ``policy_*:start`` / ``policy_*:end`` have no space
# after the colon and are therefore NOT matched by this pattern.
INLINE_POLICY_TAG_RE = re.compile(
    r"<!--\s+(?:policy_ac|policy_rules|policy_knowledge):\s+[\w\-]+\s+-->"
)
CODE_SPAN_RE = re.compile(r"`[^`]*`")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


class FinalizeError(Exception):
    """Validation failure surfaced to the CLI as exit-non-zero with
    a clear message."""


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


def _strip_inline_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text)


def _split_lines(text: str) -> list[str]:
    # ``splitlines(keepends=True)`` preserves trailing newline form; the
    # finalizer prefers logical-line manipulation, so we strip then
    # rejoin with `\n` at write time. The plan files are LF.
    return text.split("\n")


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def _heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else None


def _heading_text(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).rstrip()


def _find_section(
    lines: list[str], level: int, name: str, start: int = 0, end: int | None = None
) -> tuple[int, int] | None:
    """Find ``(heading_index, body_end_exclusive)`` for a heading whose
    level is *level* and whose text matches *name* (case-sensitive,
    surrounding whitespace ignored). Body ends at the next heading with
    level <= *level* or at *end*.
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


def _related_files_from_section(
    lines: list[str], section_start: int, section_end: int
) -> list[str]:
    """Collect bullets verbatim from a Related Files section. Skips
    empty / comment-only lines. Does NOT interpret ``inherit:`` — that
    is resolved at a higher level so callers can detect explicit
    inheritance vs. concrete paths.
    """
    return _list_bullets(lines, section_start + 1, section_end)


# ---------------------------------------------------------------------------
# Whole-plan structure model
# ---------------------------------------------------------------------------


_PATH_FROM_BULLET_RE = re.compile(r"`([^`]+)`")


def _strip_path_bullet(item: str) -> str:
    """Extract the leading repo-relative path from a Related Files
    bullet. Authors commonly write ``- `<path>` — short description``
    so the matcher pulls the FIRST backtick-wrapped span; an item
    without backticks is taken whole (after whitespace strip).
    """
    s = item.strip()
    m = _PATH_FROM_BULLET_RE.match(s)
    if m:
        return m.group(1)
    return s


def _is_spec_path(path: str) -> bool:
    """Return whether *path* is a repo-relative Markdown spec path."""
    parts = Path(path).parts
    return (
        len(parts) >= 3
        and parts[:2] == ("docs", "spec")
        and all(part not in ("", ".", "..") for part in parts)
        and path.endswith(".md")
    )


class PlanModel:
    def __init__(self, plan_path: Path) -> None:
        self.path = plan_path
        self.text = plan_path.read_text()
        self.lines = _split_lines(self.text)
        self._parse()

    def _parse(self) -> None:
        lines = self.lines

        # Plan-level Description / Related Files.
        desc = _find_section(lines, 2, "Description")
        if desc is None:
            raise FinalizeError(
                f"{self.path}: missing required `## Description` section."
            )
        rel = _find_section(lines, 3, "Related Files", desc[0] + 1, desc[1])
        if rel is None:
            raise FinalizeError(
                f"{self.path}: missing required `### Related Files` under `## Description`."
            )
        rel_items = _related_files_from_section(lines, rel[0], rel[1])
        if not rel_items:
            raise FinalizeError(
                f"{self.path}: `### Related Files` is empty."
            )
        self.plan_related_files: list[str] = [_strip_path_bullet(r) for r in rel_items]

        # Plan-level Preflight range.
        self.preflight_start_idx = self._require_unique_line(PREFLIGHT_START)
        self.preflight_end_idx = self._require_unique_line(PREFLIGHT_END)
        if self.preflight_end_idx <= self.preflight_start_idx:
            raise FinalizeError(
                f"{self.path}: `policy_preflight:end` precedes `policy_preflight:start`."
            )

        # Plan-level template sections: every level-2 heading the
        # template promises must exist (and have a non-empty body for
        # the prose ones).
        for name in ("Goal", "Constraints", "Milestones", "Execution Preflight"):
            span = _find_section(lines, 2, name)
            if span is None:
                raise FinalizeError(
                    f"{self.path}: missing required `## {name}` section."
                )

        # Milestones: each `### Milestone <name>` block.
        milestones_span = _find_section(lines, 2, "Milestones")
        assert milestones_span is not None  # guarded above
        self.milestones: list[dict[str, Any]] = []
        i = milestones_span[0] + 1
        end = milestones_span[1]
        while i < end:
            lvl = _heading_level(lines[i])
            if lvl == 3 and lines[i].startswith("### Milestone "):
                ms_start = i
                # Find this milestone's end: next level-3 within milestones or block end.
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

    def _require_unique_line(self, marker: str) -> int:
        idxs = [i for i, line in enumerate(self.lines) if line.strip() == marker]
        if not idxs:
            raise FinalizeError(
                f"{self.path}: marker {marker!r} missing — finalize_plan_context "
                "expects the template's marker pair."
            )
        if len(idxs) > 1:
            raise FinalizeError(
                f"{self.path}: marker {marker!r} appears {len(idxs)} times "
                "(expected exactly one)."
            )
        return idxs[0]

    def _parse_milestone(self, ms_start: int, ms_end: int) -> dict[str, Any]:
        lines = self.lines
        name = _heading_text(lines[ms_start])  # e.g. "Milestone M0: Schema"

        # Every level-4 heading the template promises must exist and
        # carry a non-empty body.
        sections: dict[str, tuple[int, int]] = {}
        for required in (
            "Depends",
            "Related Files",
            "Spec Impact",
            "Plan",
            "Acceptance Criteria",
        ):
            span = _find_section(lines, 4, required, ms_start + 1, ms_end)
            if span is None:
                raise FinalizeError(
                    f"{self.path}: milestone {name!r} missing `#### {required}`."
                )
            # Non-emptiness: there must be at least one non-blank, non-
            # marker-only line in the body.
            body = [
                ln
                for ln in lines[span[0] + 1 : span[1]]
                if ln.strip() and ln.strip() not in (
                    MILESTONE_AC_START,
                    MILESTONE_AC_END,
                )
            ]
            if not body:
                raise FinalizeError(
                    f"{self.path}: milestone {name!r} has empty `#### {required}`."
                )
            sections[required] = span

        related = sections["Related Files"]
        rel_items = _related_files_from_section(lines, related[0], related[1])

        # Resolve inheritance.
        effective_paths: list[str] = []
        for item in rel_items:
            if item.lower() == "inherit: top-level":
                effective_paths.extend(self.plan_related_files)
            else:
                effective_paths.append(_strip_path_bullet(item))

        self._validate_spec_impact(
            name,
            sections["Spec Impact"],
            effective_paths,
        )

        ac_section = sections["Acceptance Criteria"]

        # Locate the local policy_ac marker pair inside the AC section.
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
                f"{self.path}: milestone {name!r}: `policy_ac:end` precedes "
                f"`policy_ac:start`."
            )

        return {
            "name": name,
            "related_files": effective_paths,
            "ac_section": ac_section,
            "policy_ac_start_idx": ac_start,
            "policy_ac_end_idx": ac_end,
        }

    def _validate_spec_impact(
        self,
        milestone_name: str,
        section: tuple[int, int],
        related_files: list[str],
    ) -> None:
        items = _list_bullets(self.lines, section[0] + 1, section[1])
        label = f"{self.path}: milestone {milestone_name!r} `#### Spec Impact`"
        if not items:
            raise FinalizeError(
                f"{label} must contain one or more bullet entries."
            )

        na_items = [item for item in items if item.upper().startswith("N/A")]
        if na_items:
            if len(items) != 1:
                raise FinalizeError(
                    f"{label} cannot mix an `N/A:` entry with spec paths."
                )
            if re.fullmatch(r"N/A:\s+\S(?:.*\S)?", items[0], re.IGNORECASE) is None:
                raise FinalizeError(
                    f"{label} must use one reasoned `N/A: <reason>` entry."
                )
            return

        spec_paths: list[str] = []
        for item in items:
            path = _strip_path_bullet(item)
            if not _is_spec_path(path):
                raise FinalizeError(
                    f"{label} entry {item!r} must be a `docs/spec/*.md` path "
                    "or one reasoned `N/A:` entry."
                )
            spec_paths.append(path)

        missing = [path for path in spec_paths if path not in related_files]
        if missing:
            formatted = ", ".join(repr(path) for path in missing)
            raise FinalizeError(
                f"{label} path(s) {formatted} must also appear in the milestone's "
                "effective `#### Related Files`."
            )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _refs_phrase(refs: list[dict[str, str]]) -> str:
    parts = [f"`{r['path']} § {r['section']}`" for r in refs]
    return ", ".join(parts)


def render_preflight_body(matched: list[dict[str, Any]], role: str) -> list[str]:
    """Return the lines that go between the preflight markers
    (exclusive). Empty list when no rules / knowledge applies.

    The body is rendered deterministically: policies appear in their
    original order in the policy file; within each policy, rules
    precede knowledge.
    """
    items: list[str] = []
    for p in matched:
        for kind, marker in (("rules", "policy_rules"), ("knowledge", "policy_knowledge")):
            refs = p.get(kind, {}).get(role) or []
            if not refs:
                continue
            items.append(
                f"- {p['name']} — {p['description']} (see {_refs_phrase(refs)}) "
                f"<!-- {marker}: {p['id']} -->"
            )
    if not items:
        return []
    return ["", "### Policy Rules & Knowledge", "", *items]


def render_policy_ac_body(
    matched: list[dict[str, Any]], policies: list[dict[str, Any]]
) -> list[str]:
    items: list[str] = []
    for p in matched:
        for n, ac in enumerate(p.get("ac") or []):
            items.append(f"- [ ] {ac} <!-- policy_ac: {p['id']}-{n} -->")
    # The clang-format gate is present in every milestone: a milestone that
    # touches C++ gets the rule's AC above (via the matched loop); one that
    # touches no C++ gets an explicit N/A line so the gate is never silently
    # absent from a checklist.
    cf = next((p for p in policies if p.get("id") == "clang_format"), None)
    if cf is not None and cf not in matched:
        items.append(
            "- [ ] No touched C++/CUDA files in this milestone — clang-format "
            "gate N/A <!-- policy_ac: clang_format-na -->"
        )
    return items


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


def _replace_range(
    lines: list[str], start_idx: int, end_idx: int, body: list[str]
) -> list[str]:
    """Return a new list where `lines[start_idx + 1 : end_idx]` is
    replaced by *body*. Marker lines themselves are preserved.
    """
    return lines[: start_idx + 1] + body + lines[end_idx:]


# Stray-marker detection: see `_collect_stray_diagnostics` inside
# ``finalize_plan`` for the canonical implementation used after
# rewrite. We deliberately do not export it — the detection runs
# against the post-rewrite line list so that the allowed ranges have
# stable indices.


def finalize_plan(
    plan_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    role: str = "implementer",
    write: bool = True,
) -> tuple[str, str]:
    """Rewrite *plan_path* to a canonical form. Returns the
    ``(before, after)`` text pair.

    When *write* is False the plan file is not modified — useful for
    dry-runs in tests and for the ``--check`` mode.
    """
    plan = PlanModel(plan_path)
    policies = load_policies(policy_path)

    # Plan-level scope.
    plan_matched = filter_policies(policies, plan.plan_related_files)
    preflight_body = render_preflight_body(plan_matched, role)

    # Build the full list of (start, end, body) rewrites, then apply
    # them from highest-index to lowest. Higher-index rewrites do not
    # shift the positions of lower-index ranges, so a single descending
    # sort lets us splice without recomputing indices.
    rewrites: list[tuple[int, int, list[str]]] = [
        (plan.preflight_start_idx, plan.preflight_end_idx, preflight_body)
    ]
    for m in plan.milestones:
        matched = filter_policies(policies, m["related_files"])
        body = render_policy_ac_body(matched, policies)
        rewrites.append(
            (m["policy_ac_start_idx"], m["policy_ac_end_idx"], body)
        )
    rewrites.sort(key=lambda r: r[0], reverse=True)

    new_lines = list(plan.lines)
    for start, end, body in rewrites:
        new_lines = _replace_range(new_lines, start, end, body)

    # Stray-marker detection: any policy_ac:* / policy_rules:* /
    # policy_knowledge:* tag that lands outside every marker range is
    # a hard failure. Re-derive marker indices on the post-rewrite text
    # so a stale tag inside the now-generated range is naturally caught
    # as "inside an allowed range" while a tag living in author-written
    # prose is caught here.
    final_preflight_start = next(
        i for i, line in enumerate(new_lines) if line.strip() == PREFLIGHT_START
    )
    final_preflight_end = next(
        i for i, line in enumerate(new_lines) if line.strip() == PREFLIGHT_END
    )
    allowed: list[tuple[int, int]] = [(final_preflight_start, final_preflight_end)]
    ac_starts = [i for i, line in enumerate(new_lines) if line.strip() == MILESTONE_AC_START]
    ac_ends = [i for i, line in enumerate(new_lines) if line.strip() == MILESTONE_AC_END]
    if len(ac_starts) != len(ac_ends):
        raise FinalizeError(
            f"{plan_path}: unbalanced `policy_ac` marker pairs after rewrite "
            f"(starts={len(ac_starts)}, ends={len(ac_ends)})."
        )
    for s, e in zip(sorted(ac_starts), sorted(ac_ends)):
        allowed.append((s, e))

    # Stray-marker detection: any inline policy tag
    # `<!-- policy_(ac|rules|knowledge): <id> -->` outside an allowed
    # range is a hard failure. Code fences (``` / ~~~) and inline
    # code spans (`backticks`) are stripped before the check so that
    # prose / documentation MAY mention marker syntax in code spans.
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
    # Preserve the original file's trailing newline behaviour.
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
