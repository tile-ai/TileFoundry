#!/usr/bin/env python
"""Lint ``docs/spec/*.md`` against the mechanically-checkable subset of
``docs/SPEC-RULES.md``.

Two check families:

- **Token / header rules** — forbidden tokens and section headers, detected
  without false positives on legitimate spec prose. Anything a term could
  legitimately be (a bare commit hash looks like any hex literal; a bare
  ``#123`` looks like a link anchor) is left to human review.
- **Unified Entry Format** — every fenced ``python`` block must parse, show
  classes as ``class`` definitions (never call form), keep decorators /
  ``ParamDef`` plumbing out (core-ir.md owns that mechanism), and pass ruff's
  pydocstyle rules under the Google convention (skipped when ``ruff`` is not
  on PATH); every fenced ``cpp`` block must carry a Doxygen ``/** ... */``
  comment ahead of each declaration. A block whose first line is a
  ``# example`` / ``// example`` marker is exempt.

``docs/SPEC-RULES.md`` itself is not a spec section and is not linted — it
names the forbidden tokens as examples.

Usage: ``spec_rules_lint.py <file.md> ...`` (the pre-commit hook passes the
staged ``docs/spec/*.md`` files). Exits non-zero and prints ``file:line:
message`` for each violation.
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import textwrap

# Each rule: (compiled regex, message). The regex matches a forbidden token on
# a single line. Header-only rules are applied to heading lines separately.
_TOKEN_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmsg=[0-9a-f]{6,}"), "chat message / thread id (msg=...)"),
    (re.compile(r"æ"), "the literal `æ` annotation marker"),
    (re.compile(r"\bM\d+[a-z]?\b"), "a milestone identifier (e.g. M0 / M1a)"),
    (re.compile(r"\btask #\d+"), "a task id (task #N)"),
    (re.compile(r"\bPR #?\d+\b"), "a pull-request number (PR #N)"),
    (re.compile(r"\b(?:Alice|Bob|ZhengQiHang)\b"), "an agent / human name"),
    (
        re.compile(r"\bV\d+\b"),
        "a version stamp (e.g. V1 / V2); if this is a product identifier, "
        "rephrase or add a documented allow",
    ),
]

# Forbidden section-header terms (matched only on heading lines, so ordinary
# prose like "in the future" is never flagged).
_HEADER_TERMS = re.compile(
    r"\b(?:Non-goals?|非目标|Future|TODO|Out of scope|Tests|Testing|"
    r"Test plan|测试要求)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^\s*#{1,6}\s")


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` for every violation in *text*."""
    violations: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, message in _TOKEN_RULES:
            if pattern.search(line):
                violations.append((lineno, message))
        if _HEADING.match(line):
            m = _HEADER_TERMS.search(line)
            if m:
                violations.append(
                    (lineno, f"a forbidden section header ({m.group(0)})")
                )
    return violations


# ── Unified Entry Format checks (fenced ``python`` / ``cpp`` blocks) ────────

_FENCE = re.compile(r"^\s*```(\w*)\s*$")
_EXAMPLE_MARK = re.compile(r"^\s*(#|//)\s*example\b", re.IGNORECASE)
_CAMEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")
# A C++ declaration start: a template header, an aggregate, or a free-function
# signature (return type + name + open paren). Macro invocations
# (`NAME(...)`) have no return-type token and do not match.
_CPP_DECL = re.compile(
    r"^(template\s*<|struct\s+\w|class\s+\w|enum\s+(class\s+)?\w"
    r"|[A-Za-z_][\w:<>,&*\s]*\s[\w:]+\s*\()"
)


def _fenced_blocks(text: str) -> list[tuple[str, int, list[str]]]:
    """Return ``(language, first_content_lineno, lines)`` per fenced block."""
    blocks: list[tuple[str, int, list[str]]] = []
    lang: str | None = None
    body: list[str] = []
    start = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _FENCE.match(line)
        if m and lang is None:
            lang, body, start = m.group(1), [], lineno + 1
        elif m and lang is not None:
            blocks.append((lang, start, body))
            lang = None
        elif lang is not None:
            body.append(line)
    return blocks


def _ruff_google(src: str) -> list[str]:
    """Run ruff's pydocstyle (google convention) over one extracted block."""
    if shutil.which("ruff") is None:
        return []
    proc = subprocess.run(
        [
            "ruff", "check", "--no-cache", "--quiet",
            "--output-format", "concise",
            "--stdin-filename", "spec_block.py",
            "--select", "D2,D3,D4",
            "--config", 'lint.pydocstyle.convention="google"',
            "-",
        ],
        input=src, capture_output=True, text=True,
    )
    out = []
    for raw in proc.stdout.splitlines():
        m = re.match(r"spec_block\.py:(\d+):\d+:\s*(.*)", raw)
        if m:
            out.append(f"+{int(m.group(1)) - 1}: {m.group(2)}")
    return out


def _lint_python_block(src: str, allow_op_machinery: bool) -> list[tuple[int, str]]:
    """Format checks for one ``python`` block; offsets are 0-based in-block."""
    violations: list[tuple[int, str]] = []
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError as e:
        return [((e.lineno or 1) - 1, f"python block does not parse: {e.msg}")]
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            violations.append(
                (node.lineno - 1,
                 "floating docstring: documentation belongs inside the "
                 "class / def it describes")
            )
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and _CAMEL.match(fn.id):
                violations.append(
                    (node.lineno - 1,
                     f"class `{fn.id}` shown in call form; write its "
                     "`class` definition")
                )
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.decorator_list and not allow_op_machinery:
            violations.append(
                (node.decorator_list[0].lineno - 1,
                 "decorator in a spec interface block (concise interface only)")
            )
    if not allow_op_machinery and re.search(r"\bParamDef\s*\(", src):
        ln = next(i for i, s in enumerate(src.splitlines()) if "ParamDef" in s)
        violations.append(
            (ln, "ParamDef plumbing in a spec interface block "
                 "(owned by core-ir §2.3)")
        )
    for entry in _ruff_google(src):
        off, msg = entry[1:].split(": ", 1)
        violations.append((int(off), f"google docstring style: {msg}"))
    return violations


def _lint_cpp_block(lines: list[str]) -> list[tuple[int, str]]:
    """Each C++ declaration start must follow a closed Doxygen block."""
    violations: list[tuple[int, str]] = []
    documented = False  # a `*/` (or `///`) immediately precedes, blanks aside
    in_doxygen = False
    in_body = 0
    for off, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if in_doxygen or s.startswith("/**"):
            in_doxygen = not s.endswith("*/")
            documented = not in_doxygen
            continue
        if s.startswith(("//", "*")):
            documented = documented or s.startswith("///")
            continue
        in_body += s.count("{") - s.count("}")
        if in_body > (s.count("{") - s.count("}")):  # inside an aggregate body
            continue
        if _CPP_DECL.match(s):
            if not documented:
                violations.append(
                    (off, "C++ declaration without a preceding Doxygen "
                          "`/** ... */` block")
                )
            # A template header documents the aggregate / function it heads.
            documented = s.startswith("template")
        elif not s.startswith("#"):
            documented = False
    return violations


# Sections that OWN a decorator-based mechanism may show decorators /
# ``ParamDef`` in their interface blocks: core-ir §2.3 owns the custom-op
# machinery, visitor-registry owns the ``@register_*`` visitor registries.
_MACHINERY_OWNERS = ("core-ir.md", "visitor-registry.md")


def lint_entry_format(text: str, path: str) -> list[tuple[int, str]]:
    """Unified Entry Format checks over every fenced block in *text*."""
    allow_op_machinery = path.endswith(_MACHINERY_OWNERS)
    violations: list[tuple[int, str]] = []
    for lang, start, body in _fenced_blocks(text):
        first = next((s for s in body if s.strip()), "")
        if _EXAMPLE_MARK.match(first):
            continue
        if lang == "python":
            per_block = _lint_python_block("\n".join(body), allow_op_machinery)
        elif lang == "cpp":
            per_block = _lint_cpp_block(body)
        else:
            continue
        violations.extend((start + off, msg) for off, msg in per_block)
    return violations


def lint_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    found = lint_text(text) + lint_entry_format(text, path)
    return [f"{path}:{ln}: {msg}" for ln, msg in sorted(found)]


def main(argv: list[str]) -> int:
    failures: list[str] = []
    for path in argv:
        failures.extend(lint_file(path))
    if failures:
        sys.stderr.write(
            "spec_rules_lint: docs/spec violates docs/SPEC-RULES.md:\n"
        )
        for f in failures:
            sys.stderr.write(f"  {f}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
