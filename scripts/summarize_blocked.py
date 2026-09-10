#!/usr/bin/env python3
"""Summarize blocked fixtures by the first capability that prevents promotion."""

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_UNSUPPORTED_CALL = re.compile(r"^runtime_expression: unsupported call '([^']+)'")


@dataclass
class Reason:
    """One observed blocker shared by one or more fixture files."""

    state: str
    key: str
    files: list[Path] = field(default_factory=list)
    diagnostics: Counter[str] = field(default_factory=Counter)
    ledgers: set[str] = field(default_factory=set)


def _fields(path: Path) -> dict[str, str]:
    document = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    if document is None:
        raise ValueError(f"{path}: missing module docstring")
    fields = {}
    for line in document.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    return fields


def _ledger_ids(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _reason_key(fields: dict[str, str]) -> tuple[str, str]:
    state = fields["blocked"]
    error = fields.get("error")
    if error is not None:
        match = _UNSUPPORTED_CALL.match(error)
        if match is not None:
            return state, f"unsupported authored call `{match.group(1)}`"
        return state, error
    if state == "mis-analyzed":
        ledger = fields.get("ledger", "untracked")
        return state, f"mis-analysis tracked by {ledger}"
    raise ValueError("refused fixture is missing error")


def _validate(path: Path, fields: dict[str, str]) -> None:
    state = fields.get("blocked")
    if state not in {"refused", "mis-analyzed"}:
        raise ValueError(f"{path}: invalid blocked state {state!r}")
    if fields.get("phase") not in {"load", "selection/analysis"}:
        raise ValueError(f"{path}: invalid or missing phase")
    if state == "refused" and "error" not in fields:
        raise ValueError(f"{path}: refused fixture is missing error")
    if state == "mis-analyzed":
        missing = {"got", "expected", "why"} - fields.keys()
        if missing:
            raise ValueError(f"{path}: mis-analyzed fixture is missing {sorted(missing)}")


def _summary_table(reasons: list[Reason]) -> list[str]:
    lines = [
        "| Rank | State | First blocker | Blocked files | Fixture ledger refs |",
        "|---:|---|---|---:|---|",
    ]
    for rank, reason in enumerate(reasons, 1):
        ledgers = ", ".join(f"`{item}`" for item in sorted(reason.ledgers)) or "untracked"
        lines.append(
            f"| {rank} | {reason.state} | {reason.key} | {len(reason.files)} | {ledgers} |"
        )
    return lines


def _reason_details(reasons: list[Reason], root: Path) -> list[str]:
    lines = []
    for rank, reason in enumerate(reasons, 1):
        lines.extend(
            [
                "",
                f"## {rank}. {reason.key}",
                "",
                f"- State: `{reason.state}`",
                f"- Blocked files: {len(reason.files)}",
                "- Fixture ledger refs: "
                + (", ".join(f"`{item}`" for item in sorted(reason.ledgers)) or "untracked"),
                "- Observed diagnostics:",
                "",
            ]
        )
        lines.extend(
            f"  - {count} x `{diagnostic}`"
            for diagnostic, count in reason.diagnostics.most_common()
        )
        lines.extend(["", "- Files:", ""])
        lines.extend(f"  - `{path.relative_to(root)}`" for path in sorted(reason.files))
    return lines


def _non_gap_details(rows: list[tuple[Path, dict[str, str]]], root: Path) -> list[str]:
    lines = [
        "",
        "# Non-capability classifications",
        "",
        "These fixtures are excluded from repair priority counts.",
        "",
    ]
    for path, fields in sorted(rows):
        lines.extend(
            [
                f"## `{path.relative_to(root)}`",
                "",
                f"- Classification: {fields['classification']}",
                f"- Observed result: `{fields.get('error', fields.get('got', 'unknown'))}`",
                "",
            ]
        )
    return lines


def summarize(root: Path) -> str:
    """Return a deterministic Markdown summary for all blocked fixtures below root."""
    grouped: dict[tuple[str, str], Reason] = {}
    non_gap = []
    paths = sorted(root.rglob("*.blocked.py"))
    for path in paths:
        fields = _fields(path)
        _validate(path, fields)
        if "classification" in fields:
            non_gap.append((path, fields))
            continue
        state, key = _reason_key(fields)
        reason = grouped.setdefault((state, key), Reason(state=state, key=key))
        reason.files.append(path)
        diagnostic = fields.get("error", fields.get("got", "unknown"))
        reason.diagnostics[diagnostic] += 1
        reason.ledgers.update(_ledger_ids(fields.get("ledger")))

    reasons = sorted(grouped.values(), key=lambda item: (-len(item.files), item.state, item.key))
    lines = [
        "# Blocked fixture reasons",
        "",
        "- Generator: `uv run python scripts/summarize_blocked.py tests/fixtures`",
        f"- Blocked fixture files scanned: {len(paths)}",
        f"- Capability blocker groups: {len(reasons)}",
        f"- Non-capability classifications: {len(non_gap)}",
        "- Priority proxy: descending files currently stopped at the first blocker; actual "
        "promotions require a post-repair rerun.",
        "",
        *_summary_table(reasons),
        *_reason_details(reasons, root),
        *_non_gap_details(non_gap, root),
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    print(summarize(args.fixtures), end="")


if __name__ == "__main__":
    main()
