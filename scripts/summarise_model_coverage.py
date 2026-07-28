#!/usr/bin/env python3
"""Print a model coverage report as a table, and fail if it describes nothing.

A report is only evidence if somebody reads it, and an artifact nobody downloads is
not read. So the same numbers go into the run log, where a reader sees what the run
established and where two runs can be diffed without opening anything.

Exits non-zero when the report is empty or when it holds a FAIL. An empty report is
the failure that matters most: it is what a run produces when the model tests were
collected and never executed, and it looks like success everywhere else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ORDER = ("reference", "analyze", "schedule", "sized")


def summarise(payload: dict) -> tuple[str, int]:
    """The report as lines, and how many failures it holds."""
    lines: list[str] = []
    run = payload["run"]
    lines.append(
        f"{run['cases_reported']} cases reported across "
        f"{len(run['models_in_corpus'])} models "
        f"({len(run['modules_in_corpus'])} Modules)"
    )
    failures = 0
    for model_id in sorted(payload["models"]):
        targets = payload["models"][model_id]["targets"]
        if not targets:
            lines.append(f"{model_id}: nothing reported on any target")
            continue
        for target_id in sorted(targets):
            section = targets[target_id]
            counts: list[str] = []
            for kind in _ORDER:
                rows = (
                    section[kind] if kind == "reference" else section[kind]["tested"]
                )
                tally: dict[str, int] = {}
                for row in rows:
                    tally[row["status"]] = tally.get(row["status"], 0) + 1
                failures += tally.get("FAIL", 0)
                if not tally:
                    continue
                stated = " ".join(
                    f"{status.lower()}={count}" for status, count in sorted(tally.items())
                )
                counts.append(f"{kind} {stated}")
            untested = sum(
                len(section[kind]["untested"]) for kind in _ORDER if kind != "reference"
            )
            if untested:
                counts.append(f"untested={untested}")
            lines.append(f"{model_id} @ {target_id}: {'; '.join(counts) or 'nothing'}")
    for model_id in sorted(payload["models"]):
        for target_id, section in payload["models"][model_id]["targets"].items():
            for kind in _ORDER:
                rows = (
                    section[kind] if kind == "reference" else section[kind]["tested"]
                )
                for row in rows:
                    if row["status"] in ("BLOCKED", "FAIL"):
                        lines.append(
                            f"  {row['status']} {model_id}/{target_id}/{kind} "
                            f"{row['case']}: {row.get('reason', '')}"
                        )
    return "\n".join(lines), failures


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: summarise_model_coverage.py MODEL-COVERAGE.JSON", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"no coverage report at {path}", file=sys.stderr)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    text, failures = summarise(payload)
    print(text)
    if not payload["run"]["cases_reported"]:
        print(
            "the report holds no cases: the model tests were collected and never ran",
            file=sys.stderr,
        )
        return 1
    if failures:
        print(f"{failures} case(s) reported FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
