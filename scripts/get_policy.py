#!/usr/bin/env python3
"""Select entries from ``docs/policies/project-policy.json``.

Rules and knowledge return metadata plus references; acceptance criteria return
their contract text. Role and path filters narrow results, and ``--policy``
overrides the repository-relative source file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "docs" / "policies" / "project-policy.json"


def _glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a gitignore-style glob to a fullmatch regex.

    Supported:
    - ``**`` matches zero or more path segments (including ``/``).
    - ``*`` matches anything except ``/``.
    - ``?`` matches a single character except ``/``.
    Everything else is matched literally.
    """
    parts: list[str] = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if glob.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif glob.startswith("/**", i):
            parts.append("(?:/.*)?")
            i += 3
        elif c == "*" and glob[i + 1 : i + 2] == "*":
            parts.append(".*")
            i += 2
        elif c == "*":
            parts.append("[^/]*")
            i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return re.compile("".join(parts))


def path_matches_glob(path: str, glob: str) -> bool:
    return _glob_to_regex(glob).fullmatch(path) is not None


def policy_matches_paths(policy: dict[str, Any], paths: list[str]) -> bool:
    """OR semantics: policy matches when any (path × glob) pair matches.

    Empty *paths* means "no path filter" — every policy passes.
    """
    if not paths:
        return True
    globs = policy.get("when", {}).get("path_glob", [])
    if not globs:
        return False
    regexes = [_glob_to_regex(g) for g in globs]
    return any(r.fullmatch(p) is not None for p in paths for r in regexes)


def load_policies(policy_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(policy_path.read_text())
    return list(raw.get("policies", []))


def filter_policies(policies: list[dict[str, Any]], paths: list[str]) -> list[dict[str, Any]]:
    return [p for p in policies if policy_matches_paths(p, paths)]


def select_ac(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in policies:
        items = p.get("ac")
        if not items:
            continue
        out.append({"id": p["id"], "items": list(items)})
    return out


def select_refs(policies: list[dict[str, Any]], kind: str, role: str) -> list[dict[str, Any]]:
    """Return index entries for ``kind in {"rules", "knowledge"}``.

    Each entry is ``{id, name, description, refs}`` where ``refs`` is the
    role-filtered list of ``{path, section}`` objects. Policies whose
    selected role bucket is empty / missing are dropped — the consumer
    only sees actionable entries.
    """
    assert kind in ("rules", "knowledge")
    out: list[dict[str, Any]] = []
    for p in policies:
        refs = p.get(kind, {}).get(role) or []
        if not refs:
            continue
        out.append(
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "refs": [dict(r) for r in refs],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--type", required=True, choices=("ac", "rules", "knowledge"))
    p.add_argument(
        "--role",
        choices=("implementer", "reviewer"),
        help="Required when --type is rules or knowledge.",
    )
    p.add_argument(
        "--paths",
        nargs="*",
        default=[],
        help="OR-filter policies by when.path_glob. Empty means no filter.",
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help=f"Policy JSON path (default: {DEFAULT_POLICY}).",
    )
    args = p.parse_args(argv)

    if args.type in ("rules", "knowledge") and not args.role:
        p.error("--role is required when --type is rules or knowledge")

    policies = load_policies(args.policy)
    filtered = filter_policies(policies, list(args.paths))

    if args.type == "ac":
        out = select_ac(filtered)
    else:
        out = select_refs(filtered, args.type, args.role)

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
