#!/usr/bin/env python3
"""Finalize policy-generated regions in an authored plan.

Input is ``docs/plans/<name>.md``. hygiene: required CLI path template.
Related Files select milestone acceptance criteria and final-tree gates.
Handwritten content is preserved; a plan that breaks the template's structure
fails validation instead of being rewritten.

Structure comes from a CommonMark parse rather than a line scan: a plan quotes
command output whose lines begin with `#`, and a scan would read one as a heading
and end the section it belongs to.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from get_policy import filter_policies, load_policies  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "docs" / "policies" / "project-policy.json"

AC_START, AC_END = "<!-- policy_ac:start -->", "<!-- policy_ac:end -->"
GATE_START, GATE_END = "<!-- final_gate:start -->", "<!-- final_gate:end -->"

HEADINGS: dict[int, set[str]] = {
    2: {"Description", "Milestones", "Final Gate"},
    4: {"Depends", "Target State Design", "Related Files"},
    5: {"Delivered", "Accepted by"},
}
CODE_LANGS = frozenset(
    "bash c c++ cc cpp cs csharp go java javascript js kotlin py python rs rust "
    "sh shell swift ts typescript zsh".split()
)

CHECKBOX_RE = re.compile(r"^\s*-\s+\[([ xX])\].*<!--\s+policy_(?:ac|final):\s+([\w\-]+)\s+-->")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
CITATION_RE = re.compile(r"`[^`\s]*/[^`\s]*\.[^`\s]*`")
DECISION_RE = re.compile(r"^D(\d+)\s+.*?\s--\s+\S")
SUPERSEDES_RE = re.compile(r"\bSupersedes\s+D(\d+)\b")


class FinalizeError(Exception):
    """Report a validation failure clearly through a nonzero CLI exit."""


@dataclass(frozen=True)
class Section:
    """One heading and the lines under it, up to the next heading of its depth."""

    level: int
    title: str
    head: int
    end: int


def _front_matter_lines(lines: list[str]) -> int:
    """How many leading lines the YAML block occupies.

    CommonMark reads the closing `---` as a setext underline, so the block is cut
    off before parsing and its length is added back to every token's position.
    """
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i + 1
    return 0


class Plan:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.text = path.read_text()
        self.lines = self.text.split("\n")
        offset = _front_matter_lines(self.lines)
        tokens = MarkdownIt("commonmark").parse("\n".join(self.lines[offset:]))
        for token in tokens:
            if token.map is not None:
                token.map = [token.map[0] + offset, token.map[1] + offset]
        self.tokens = tokens
        self.sections = self._outline()

    def _outline(self) -> list[Section]:
        heads: list[tuple[int, str, int]] = []
        for index, token in enumerate(self.tokens):
            if token.type == "heading_open" and token.map is not None:
                heads.append((int(token.tag[1]), self.tokens[index + 1].content.strip(), token.map[0]))
        out: list[Section] = []
        for position, (level, title, head) in enumerate(heads):
            end = len(self.lines)
            for deeper_level, _, deeper_head in heads[position + 1 :]:
                if deeper_level <= level:
                    end = deeper_head
                    break
            out.append(Section(level, title, head, end))
        return out

    def find(self, level: int, title: str, within: Section | None = None) -> Section | None:
        lo = 0 if within is None else within.head
        hi = len(self.lines) if within is None else within.end
        return next(
            (s for s in self.sections if s.level == level and s.title == title and lo <= s.head < hi),
            None,
        )

    def milestones(self) -> list[Section]:
        block = self.find(2, "Milestones")
        if block is None:
            return []
        return [
            s
            for s in self.sections
            if s.level == 3 and s.title.startswith("Milestone ") and block.head < s.head < block.end
        ]

    def body(self, section: Section) -> list[str]:
        """The section's lines with its heading, HTML comments, and blanks dropped."""
        text = COMMENT_RE.sub("", "\n".join(self.lines[section.head + 1 : section.end]))
        return [line for line in text.split("\n") if line.strip()]

    def _tokens_in(self, section: Section, kind: str) -> list[Any]:
        return [
            t
            for t in self.tokens
            if t.type == kind and t.map is not None and section.head < t.map[0] < section.end
        ]

    def fence_languages(self, section: Section) -> list[str]:
        return [t.info.strip().split()[0].lower() if t.info.strip() else "" for t in self._tokens_in(section, "fence")]

    def bullets(self, section: Section) -> list[str]:
        """One entry per list item in *section*, comments and markup stripped.

        The item's text is the first ``inline`` token after its opener; counting
        list depth instead would have to reckon with closers, which carry no
        position and so cannot be told apart from another section's.
        """
        out: list[str] = []
        for index, token in enumerate(self.tokens):
            if token.type != "list_item_open" or token.map is None:
                continue
            if not section.head < token.map[0] < section.end:
                continue
            inline = next(
                (t for t in self.tokens[index + 1 :] if t.type == "inline"), None
            )
            if inline is None:
                continue
            text = COMMENT_RE.sub("", inline.content).strip()
            if text:
                out.append(text)
        return out

    def marker(self, literal: str) -> int:
        found = [
            t.map[0]
            for t in self.tokens
            if t.type == "html_block" and t.map is not None and t.content.strip() == literal
        ]
        if len(found) != 1:
            raise FinalizeError(
                f"{self.path}: marker {literal!r} appears {len(found)} times; "
                "the template pairs it exactly once."
            )
        return found[0]

    def checkbox_states(self, start: int, end: int) -> dict[str, bool]:
        states: dict[str, bool] = {}
        for line in self.lines[start + 1 : end]:
            match = CHECKBOX_RE.match(line)
            if match:
                states[match.group(2)] = match.group(1).lower() == "x"
        return states

    def related_files(self, milestone: Section) -> list[str]:
        section = self.find(4, "Related Files", milestone)
        if section is None:
            return []
        out: list[str] = []
        for item in self.bullets(section):
            match = re.match(r"`([^`]+)`", item)
            out.append(match.group(1) if match else item)
        return out



def check_headings(plan: Plan) -> None:
    """Only the sections the template names may appear.

    A section the template does not name is where a second statement of acceptance,
    a design note belonging in the spec, or a status log accumulates -- each of them
    a source that will later disagree with the one the template names. Depth 3 is
    free, because that is how `## Description` is grouped.
    """
    for section in plan.sections:
        if section.level in (1, 3):
            continue
        allowed = HEADINGS.get(section.level)
        if allowed is None:
            raise FinalizeError(
                f"{plan.path}:{section.head + 1}: heading depth {section.level} is not used "
                f"by the plan template: `{'#' * section.level} {section.title}`"
            )
        if section.title not in allowed:
            raise FinalizeError(
                f"{plan.path}:{section.head + 1}: `{'#' * section.level} {section.title}` is not "
                f"a section the plan template defines. Allowed at this depth: "
                f"{', '.join(sorted(allowed))}."
            )


def check_skeleton(plan: Plan) -> None:
    """The three top sections exist and `## Milestones` holds at least one."""
    for title in ("Description", "Milestones", "Final Gate"):
        if plan.find(2, title) is None:
            raise FinalizeError(f"{plan.path}: missing required `## {title}` section.")
    if not plan.milestones():
        raise FinalizeError(
            f"{plan.path}: `## Milestones` block contains no `### Milestone …` entries."
        )
    for milestone in plan.milestones():
        for title in ("Depends", "Target State Design", "Related Files"):
            if plan.find(4, title, milestone) is None:
                raise FinalizeError(
                    f"{plan.path}: milestone {milestone.title!r} has no `#### {title}`."
                )


def check_acceptance(plan: Plan, policies: list[dict[str, Any]]) -> None:
    """Every milestone states how its target state is accepted.

    The delivered shape and the way it is accepted are settled together while the
    plan is written. A milestone that omits the second half would leave that choice
    to whoever implements it, so finalizing fails and prints the contract.
    """
    guidance = next((p["guidance"] for p in policies if p.get("guidance")), [])
    for milestone in plan.milestones():
        design = plan.find(4, "Target State Design", milestone)
        assert design is not None
        for title in ("Delivered", "Accepted by"):
            section = plan.find(5, title, design)
            if section is None or not plan.body(section):
                detail = "\n".join(guidance)
                raise FinalizeError(
                    f"{plan.path}: milestone {milestone.title!r} has no `##### {title}` "
                    f"under `#### Target State Design`." + (f"\n\n{detail}" if detail else "")
                )


def check_delivered_shape(plan: Plan) -> None:
    """Every milestone shows its delivered surface as code.

    A predicate or an access map written as prose makes each reader rebuild it, and
    two readers rebuild it differently -- which is how a milestone gets built as
    something the plan never said. A `text` block is printed output, which is what
    the shape produces rather than the shape.
    """
    for milestone in plan.milestones():
        design = plan.find(4, "Target State Design", milestone)
        section = None if design is None else plan.find(5, "Delivered", design)
        if section is None:
            continue
        languages = plan.fence_languages(section)
        if any(language in CODE_LANGS for language in languages):
            continue
        seen = (
            "no fenced block at all"
            if not languages
            else "only " + ", ".join(f"```{lang or '<untagged>'}" for lang in languages)
        )
        raise FinalizeError(
            f"{plan.path}: milestone {milestone.title!r} states its `##### Delivered` in "
            f"prose ({seen}). The template asks for the delivered surface in code or "
            f"compact pseudocode, in a block tagged with a programming language "
            f"({', '.join(sorted(CODE_LANGS))})."
        )


def check_current_state(plan: Plan) -> None:
    """The state the plan is built on is stated, and every claim points somewhere.

    A claim with nothing to point at is the one that turns out to be wrong, and the
    design built on it moves once it does. A citation makes the author's check cheap:
    spot-check three of them.
    """
    description = plan.find(2, "Description")
    assert description is not None
    section = plan.find(3, "Current state", description)
    if section is None or not plan.body(section):
        raise FinalizeError(
            f"{plan.path}: `## Description` has no `### Current state`. State what the code "
            "does today, one bullet per claim, before the design that rests on it."
        )
    items = plan.bullets(section)
    if not items:
        raise FinalizeError(
            f"{plan.path}: `### Current state` states no bullets; one claim per bullet."
        )
    bare = [item for item in items if not CITATION_RE.search(item)]
    if bare:
        raise FinalizeError(
            f"{plan.path}: `### Current state` has {len(bare)} claim(s) citing nothing. "
            f"Each needs a `path` or `path:line` in backticks. First: {bare[0][:80]!r}"
        )


def check_decisions(plan: Plan) -> None:
    """Settled questions are recorded once and superseded rather than rewritten.

    Rewriting the record erases the position it replaced, so a reader cannot tell
    whether the root moved or only a detail did. Ids must resolve, because a
    supersede pointing at nothing is a rewrite wearing the shape of a record.
    """
    description = plan.find(2, "Description")
    assert description is not None
    section = plan.find(3, "Decisions", description)
    if section is None or not plan.body(section):
        raise FinalizeError(
            f"{plan.path}: `## Description` has no `### Decisions`. Record each settled "
            "question, or state `None.` when the plan settled nothing an implementer "
            "would otherwise choose alone."
        )
    if [line.strip() for line in plan.body(section)] == ["None."]:
        return
    items = plan.bullets(section)
    ids: list[int] = []
    for item in items:
        match = DECISION_RE.match(item)
        if match is None:
            raise FinalizeError(
                f"{plan.path}: `### Decisions` record does not read as "
                f"`- D<n> <question> -- <choice>, because <reason>.`: {item[:80]!r}"
            )
        ids.append(int(match.group(1)))
    if not ids:
        raise FinalizeError(f"{plan.path}: `### Decisions` is neither `None.` nor any record.")
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise FinalizeError(
            f"{plan.path}: `### Decisions` reuses id(s) {sorted(duplicated)}; each record is one id."
        )
    for item in items:
        for referenced in SUPERSEDES_RE.findall(item):
            if int(referenced) not in ids:
                raise FinalizeError(
                    f"{plan.path}: `### Decisions` supersedes D{referenced}, which no record "
                    f"states. A supersede names the record it replaces: {item[:80]!r}"
                )



def render_items(
    matched: list[dict[str, Any]], states: dict[str, bool], *, field: str, tag: str
) -> list[str]:
    """One checkbox per criterion the matched policies contribute.

    A criterion the author already ticked keeps its mark, so finalizing a plan
    mid-implementation does not silently reopen what is done.
    """
    items: list[str] = []
    for policy in matched:
        for index, criterion in enumerate(policy.get(field) or []):
            marker = f"{policy['id']}-{index}"
            check = "x" if states.get(marker, False) else " "
            items.append(f"- [{check}] {criterion} <!-- {tag}: {marker} -->")
    return items


def render_final_gate(
    matched: list[dict[str, Any]], policies: list[dict[str, Any]], states: dict[str, bool]
) -> list[str]:
    """The repository-wide gates, plus the one that records a gate as not applying.

    A plan touching no C++ says so rather than leaving the clang-format gate absent,
    because absent reads as forgotten.
    """
    items = render_items(matched, states, field="final_ac", tag="policy_final")
    clang = next((p for p in policies if p.get("id") == "clang_format"), None)
    if clang is not None and clang not in matched:
        marker = "clang_format-na"
        check = "x" if states.get(marker, False) else " "
        items.append(
            f"- [{check}] No touched C++/CUDA files in this plan — clang-format "
            f"gate N/A <!-- policy_final: {marker} -->"
        )
    return items


def finalize_plan(
    plan_path: Path, *, policy_path: Path = DEFAULT_POLICY, write: bool = True
) -> tuple[str, str]:
    """Rewrite *plan_path* to canonical form, returning the ``(before, after)`` pair."""
    plan = Plan(plan_path)
    policies = load_policies(policy_path)
    check_headings(plan)
    check_skeleton(plan)
    check_acceptance(plan, policies)
    check_delivered_shape(plan)
    check_current_state(plan)
    check_decisions(plan)

    every_related: list[str] = []
    rewrites: list[tuple[int, int, list[str]]] = []
    for milestone in plan.milestones():
        related = plan.related_files(milestone)
        every_related.extend(related)
        start = _paired_marker(plan, milestone, AC_START)
        end = _paired_marker(plan, milestone, AC_END)
        states = plan.checkbox_states(start, end)
        rewrites.append(
            (start, end, render_items(filter_policies(policies, related), states, field="ac", tag="policy_ac"))
        )

    gate_start, gate_end = plan.marker(GATE_START), plan.marker(GATE_END)
    if gate_end <= gate_start:
        raise FinalizeError(f"{plan_path}: `final_gate:end` precedes `final_gate:start`.")
    rewrites.append(
        (
            gate_start,
            gate_end,
            render_final_gate(
                filter_policies(policies, every_related),
                policies,
                plan.checkbox_states(gate_start, gate_end),
            ),
        )
    )

    lines = list(plan.lines)
    for start, end, body in sorted(rewrites, key=lambda r: r[0], reverse=True):
        lines = lines[: start + 1] + body + lines[end:]
    after = "\n".join(lines)
    if write and after != plan.text:
        plan_path.write_text(after)
    return plan.text, after


def _paired_marker(plan: Plan, milestone: Section, literal: str) -> int:
    """The one occurrence of *literal* inside *milestone*'s `##### Accepted by`."""
    found = [
        t.map[0]
        for t in plan.tokens
        if t.type == "html_block"
        and t.map is not None
        and t.content.strip() == literal
        and milestone.head < t.map[0] < milestone.end
    ]
    if len(found) != 1:
        raise FinalizeError(
            f"{plan.path}: milestone {milestone.title!r} holds {len(found)} {literal!r} "
            "markers; the template pairs them exactly once inside `##### Accepted by`."
        )
    return found[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("plan", type=Path, help="Path to docs/plans/<name>.md")
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help=f"Policy JSON path (default: {DEFAULT_POLICY}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate only; exit non-zero if a rewrite would change the file.",
    )
    args = parser.parse_args(argv)

    try:
        before, after = finalize_plan(args.plan, policy_path=args.policy, write=not args.check)
    except FinalizeError as error:
        sys.stderr.write(f"{error}\n")
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
