#!/usr/bin/env python
"""Changed-file gate for the spec-authority rule (see docs/SPEC-RULES.md).

Flags a long docstring or comment block that carries normative vocabulary — a
sustained block of such prose in code is a design statement that belongs in the
spec. Reads code only; a short local-mechanics note stays below the length gate.
The pre-commit hook passes the staged ``src/**`` Python files; this is a
per-file gate, not a whole-tree one (legacy debt is a separate sweep), so it
takes explicit paths. Usage: ``spec_entropy_lint.py <path> ...``.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

# A block must reach this many lines before its prose is treated as "long".
# Below the gate, a comment is assumed to be local mechanics and is left alone.
# The gate is set high enough that a short local-mechanics note that happens to
# say "must" in passing is never flagged; only a sustained block of normative
# prose trips it.
_COMMENT_BLOCK_MIN_LINES = 6
_DOCSTRING_MIN_LINES = 8

# RFC-2119 normative keywords are matched case-sensitively (their uppercase form
# is the tell of a contract); the design-vocabulary phrases are matched
# case-insensitively.
_RFC2119 = ("MUST NOT", "MUST", "SHOULD NOT", "SHOULD", "MAY")
_PHRASES = ("contract", "invariant", "dispatch principle", "single source")


def _has_contract_vocab(text: str) -> bool:
    for kw in _RFC2119:
        if kw in text:
            return True
    low = text.lower()
    return any(p in low for p in _PHRASES)


def _comment_violations(src: str) -> list[tuple[int, str]]:
    """Contiguous runs of full-line ``#`` comments that are long and normative."""
    out: list[tuple[int, str]] = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return out
    block: list[tuple[int, str]] = []  # (lineno, comment text)
    prev_line = None

    def flush() -> None:
        if len(block) >= _COMMENT_BLOCK_MIN_LINES:
            joined = "\n".join(t for _, t in block)
            if _has_contract_vocab(joined):
                out.append(
                    (block[0][0],
                     "suspected contract in a code comment block "
                     f"({len(block)} lines) — move it to docs/spec/")
                )
        block.clear()

    for tok in toks:
        if tok.type == tokenize.COMMENT:
            lineno = tok.start[0]
            if prev_line is not None and lineno != prev_line + 1:
                flush()
            block.append((lineno, tok.string))
            prev_line = lineno
        elif tok.type in (tokenize.NL, tokenize.COMMENT):
            continue
        else:
            if tok.type not in (tokenize.NEWLINE, tokenize.INDENT,
                                tokenize.DEDENT, tokenize.ENCODING):
                flush()
                prev_line = None
    flush()
    return out


def _docstring_violations(src: str) -> list[tuple[int, str]]:
    """Module / class / function docstrings that are long and normative."""
    out: list[tuple[int, str]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        nonblank = [ln for ln in doc.splitlines() if ln.strip()]
        if len(nonblank) < _DOCSTRING_MIN_LINES:
            continue
        if not _has_contract_vocab(doc):
            continue
        lineno = getattr(node, "body", [node])[0].lineno if getattr(
            node, "body", None) else getattr(node, "lineno", 1)
        out.append(
            (lineno,
             "suspected contract in a docstring "
             f"({len(nonblank)} lines) — move it to docs/spec/")
        )
    return out


def lint_file(path: Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    return sorted(_comment_violations(src) + _docstring_violations(src))


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: spec_entropy_lint.py <path> ...  "
              "(pass the files to check; the pre-commit hook passes the "
              "staged src/**.py files)", file=sys.stderr)
        return 2
    roots = [Path(a) for a in argv]
    files: list[Path] = []
    for r in roots:
        files.extend([r] if r.is_file() else sorted(r.rglob("*.py")))
    failed = False
    for f in files:
        for lineno, msg in lint_file(f):
            print(f"{f}:{lineno}: {msg}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
