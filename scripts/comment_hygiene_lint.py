#!/usr/bin/env python3
"""Keep source documentation local, compact, and independent of work history.

Python prose belongs in module, class, or function docstrings. C-family prose
uses Doxygen comments. Ordinary comments are reserved for tool directives, and
documentation must not point at transient plans, reviews, or change records.

The hook accepts explicit paths and reports every violation as ``path:line``.
Unreadable files and unsupported suffixes are ignored.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

MAX_PROSE_LINES = 8
MAX_COLUMNS = 100
EXEMPT_PREFIXES = ("tests/models/", "examples/")
DIRECTIVE_PREFIXES = ("ruff:", "noqa", "type:", "pragma:", "mypy:", "fmt:", "isort:")
PYTHON_SUFFIXES = frozenset({".py"})
C_SUFFIXES = frozenset({".h", ".hpp", ".cuh", ".cu", ".cpp", ".cc"})
GOOGLE_SECTION = re.compile(
    r"^\s*(?:Args|Arguments|Attributes|Example|Examples|Note|Notes|Parameters"
    r"|Raises|References|Returns|See Also|Todo|Warns|Warnings|Yields)\s*:\s*$"
)
ALLOW = re.compile(r"\bhygiene:")
NARRATION: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bAC-\d+-\d+\b"), "an acceptance-criterion id"),
    (re.compile(r"\bmilestones?\s+M\d+\b", re.IGNORECASE), "a milestone reference"),
    (re.compile(r"\bplans?\s+`?\d+`?\b", re.IGNORECASE), "a plan reference"),
    (re.compile(r"docs/plans/"), "a plan path"),
    (re.compile(r"\b(?:PR|pull request|issue)\s*#?\d+\b", re.IGNORECASE), "a PR/issue reference"),
    (re.compile(r"\bcommit\s+[0-9a-f]{7,40}\b", re.IGNORECASE), "a commit hash"),
    (
        re.compile(
            r"\b(?:as discussed|per review|review(?:er)? (?:said|asked|wants|requested)"
            r"|address(?:ed|ing) (?:the )?(?:review|feedback)|as agreed)\b",
            re.IGNORECASE,
        ),
        "review narration",
    ),
)
SELF = Path(__file__).resolve()
ROOT = SELF.parent.parent


def is_exempt(path: Path) -> bool:
    """Whether *path* is temporarily outside placement and size checks."""
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return False
    return any(relative.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def comment_tokens(text: str) -> list[tokenize.TokenInfo]:
    """Python comment tokens in *text*, excluding hashes inside strings."""
    try:
        return [
            token
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT
        ]
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return []


def is_directive(token: tokenize.TokenInfo) -> bool:
    """Whether a hash token addresses a tool rather than a reader."""
    body = token.string.lstrip("#").strip()
    return body.startswith(DIRECTIVE_PREFIXES) or (
        token.start[0] == 1 and token.start[1] == 0 and token.string.startswith("#!")
    )


def prose_lines(text: str) -> int:
    """Count nonblank docstring lines before the first Google section."""
    total = 0
    for line in text.splitlines():
        if GOOGLE_SECTION.match(line):
            break
        if line.strip():
            total += 1
    return total


def _docstring_nodes(text: str) -> list[ast.Constant]:
    """String nodes that serve as Python docstrings."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    nodes = []
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for owner in ast.walk(tree):
        if isinstance(owner, owners) and owner.body:
            statement = owner.body[0]
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                nodes.append(statement.value)
    return nodes


def _narration(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Transient process references in numbered prose lines."""
    found = []
    for number, line in lines:
        if ALLOW.search(line):
            continue
        for pattern, what in NARRATION:
            if pattern.search(line):
                found.append((number, f"documentation carries {what}"))
                break
    return found


def narration_findings(text: str) -> list[tuple[int, str]]:
    """Process narration in Python comments and docstrings."""
    lines = [(token.start[0], token.string.lstrip("#").strip()) for token in comment_tokens(text)]
    for node in _docstring_nodes(text):
        lines.extend(
            (node.lineno + offset, line) for offset, line in enumerate(node.value.splitlines())
        )
    return _narration(lines)


def placement_findings(text: str) -> list[tuple[int, str]]:
    """Python comments that are neither tool directives nor a shebang."""
    return [
        (token.start[0], "explanation outside a docstring")
        for token in comment_tokens(text)
        if not is_directive(token)
    ]


def docstring_findings(text: str) -> list[tuple[int, str]]:
    """Python docstrings beyond the prose or column budget."""
    source_lines = text.splitlines()
    found = []
    for node in _docstring_nodes(text):
        spent = prose_lines(node.value)
        if spent > MAX_PROSE_LINES:
            found.append(
                (node.lineno, f"docstring spends {spent} prose lines; limit is {MAX_PROSE_LINES}")
            )
        end = node.end_lineno or node.lineno
        for number in range(node.lineno, end + 1):
            width = len(source_lines[number - 1].rstrip())
            if width > MAX_COLUMNS:
                found.append(
                    (number, f"documentation line is {width} columns; limit is {MAX_COLUMNS}")
                )
    return found


def width_findings(text: str) -> list[tuple[int, str]]:
    """Overwide Python tool directives and shebangs."""
    found = []
    lines = text.splitlines()
    for token in comment_tokens(text):
        if is_directive(token):
            width = len(lines[token.start[0] - 1].rstrip())
            if width > MAX_COLUMNS:
                found.append(
                    (
                        token.start[0],
                        f"documentation line is {width} columns; limit is {MAX_COLUMNS}",
                    )
                )
    return found


def _c_comments(text: str) -> list[tuple[int, str, str, bool]]:
    """C-family comments as line, marker, body, and whole-line status."""
    found = []
    index = 0
    line = 1
    line_start = 0
    size = len(text)
    while index < size:
        char = text[index]
        if char == "\n":
            line += 1
            line_start = index + 1
            index += 1
            continue
        if char in "\"'":
            quote = char
            index += 1
            while index < size:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    break
                if text[index] == "\n":
                    line += 1
                    line_start = index + 1
                index += 1
            continue
        if text.startswith('R"', index):
            delimiter_end = text.find("(", index + 2, index + 19)
            if delimiter_end >= 0:
                delimiter = text[index + 2 : delimiter_end]
                terminator = f'){delimiter}"'
                raw_end = text.find(terminator, delimiter_end + 1)
                stop = size if raw_end < 0 else raw_end + len(terminator)
                segment = text[index:stop]
                line += segment.count("\n")
                last_newline = segment.rfind("\n")
                if last_newline >= 0:
                    line_start = index + last_newline + 1
                index = stop
                continue
        if text.startswith("//", index):
            marker = (
                "///<"
                if text.startswith("///<", index)
                else "///"
                if text.startswith("///", index)
                else "//"
            )
            end = text.find("\n", index)
            end = size if end < 0 else end
            prefix = text[line_start:index]
            found.append((line, marker, text[index + len(marker) : end], not prefix.strip()))
            index = end
            continue
        if text.startswith("/*", index):
            marker = "/**" if text.startswith("/**", index) else "/*"
            end = text.find("*/", index + len(marker))
            end = size if end < 0 else end
            prefix = text[line_start:index]
            body = text[index + len(marker) : end]
            found.append((line, marker, body, not prefix.strip()))
            segment = text[index : min(size, end + 2)]
            line += segment.count("\n")
            last_newline = segment.rfind("\n")
            if last_newline >= 0:
                line_start = index + last_newline + 1
            index = min(size, end + 2)
            continue
        index += 1
    return found


def c_findings(text: str) -> list[tuple[int, str]]:
    """Illegal C-family comments and oversized Doxygen prose."""
    found = []
    narration_lines = []
    line_run: list[tuple[int, str]] = []

    def finish_line_run() -> None:
        if not line_run:
            return
        cleaned = "\n".join(body.strip() for _, body in line_run)
        spent = prose_lines(cleaned)
        if spent > MAX_PROSE_LINES:
            found.append(
                (
                    line_run[0][0],
                    f"Doxygen block spends {spent} prose lines; limit is {MAX_PROSE_LINES}",
                )
            )
        line_run.clear()

    for number, marker, body, whole_line in _c_comments(text):
        narration_lines.extend(
            (number + offset, line) for offset, line in enumerate(body.splitlines())
        )
        legal = (
            marker == "/**"
            and whole_line
            or marker == "///"
            and whole_line
            or marker == "///<"
            and not whole_line
        )
        if not legal:
            finish_line_run()
            found.append((number, "C-family prose must use Doxygen at a declaration"))
            continue
        if marker == "///":
            if line_run and number != line_run[-1][0] + 1:
                finish_line_run()
            line_run.append((number, body))
            continue
        finish_line_run()
        cleaned = "\n".join(line.strip().lstrip("*").strip() for line in body.splitlines())
        spent = prose_lines(cleaned)
        if marker == "/**" and spent > MAX_PROSE_LINES:
            found.append(
                (number, f"Doxygen block spends {spent} prose lines; limit is {MAX_PROSE_LINES}")
            )
    finish_line_run()
    return _narration(narration_lines) + found


def findings(path: Path) -> list[tuple[int, str]]:
    """Every violation in *path*, as a line number and explanation."""
    if path.suffix not in PYTHON_SUFFIXES | C_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if path.suffix in C_SUFFIXES:
        return [] if is_exempt(path) else sorted(c_findings(text))
    found = [] if path.resolve() == SELF else narration_findings(text)
    if not is_exempt(path) or path.resolve() == SELF:
        found.extend(placement_findings(text))
        found.extend(docstring_findings(text))
        found.extend(width_findings(text))
    return sorted(found)


def main(argv: list[str]) -> int:
    """Run the checker for hook-supplied paths."""
    if not argv:
        print("usage: comment_hygiene_lint.py <path> ...", file=sys.stderr)
        return 2
    failed = False
    for name in argv:
        path = Path(name)
        for number, what in findings(path):
            failed = True
            print(f"{path}:{number}: {what}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
