#!/usr/bin/env python3
"""Reject a reference to a spec section that does not point at that section.

A reference is written `[<doc> §<number>](<path>#<github-anchor>)`. The display
text carries the light form a reader wants; the target carries the heading's
name. Both halves are checked against each other and against the document, which
is what makes the reference survive editing: rename a heading and its anchor stops
resolving, renumber one and the display text stops agreeing with what it points
at. A bare `§2.3` has neither property -- it is a number that was true once.

Five things are refused:

* an anchor that names no heading in the target document;
* an anchor that names a heading other than the section the display text numbers,
  including a heading nested under it -- a reference to `§2.3` goes to `§2.3`, not
  to one of its subsections;
* a display text naming a different document than the target;
* a reference link with no anchor at all, `[types §4](./types.md)`, which has
  nothing in it that can break when a heading is renamed;
* a bare `§2.3`, outside a fenced block, where a link belongs.

Headings come from `tilefoundry.utils.markdown`, the same scan
`tilefoundry.cli.spec.sections` reads a document through, so a `#` line inside a
fenced block is a comment here exactly as it is there and the two cannot grow
separate answers to what sections a document has.

Run over the files a commit touches (the pre-commit hook passes them); a
reference is reported with the line it is on, and the exit status is non-zero if
any was found.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

#: A markdown link whose display text carries a section number. The character
#: classes admit newlines, so a reference reflowed across two lines is still one
#: reference rather than two halves that each look like prose.
_REFERENCE = re.compile(r"\[([^\[\]]*?§[0-9][^\[\]]*?)\]\(([^()]*?)\)", re.DOTALL)

#: A section number standing on its own, once the references around it are out
#: of the way. `§2.3` says which section was true when it was written and stops
#: saying anything the moment one is inserted above it.
_BARE = re.compile(r"§[0-9](?:\.[0-9]+)*")

#: A line break plus whatever the next line opens with: a C++ `//`, a block
#: comment `*`, or nothing at all.
_CONTINUATION = re.compile(r"\s*\n\s*(?://+|\*)?\s*")

#: A fence opens or closes a block in which `§2.3` is example text.
_FENCE = re.compile(r"^\s*(```|~~~)")

#: The display text: an optional document, then the number. The document may be
#: written with or without `.md`, and the whole may be wrapped in backticks.
_DISPLAY = re.compile(
    r"^\s*`?\s*(?:(?P<doc>[a-z][\w-]*)(?:\.md)?\s+)?§(?P<number>[0-9][0-9.]*?)\.?\s*`?\s*$",
    re.DOTALL,
)


def _markdown():
    """`tilefoundry.utils.markdown`, loaded from its file.

    The hook runs whatever `python` resolves to, which is not required to have
    `tilefoundry` installed, and importing the package would pull the whole IR
    -- `isl`, torch -- to read markdown headings. The module is a leaf by
    contract: it imports nothing, so loading it by path needs no stand-ins and
    gives the hook the same heading scan the `spec` command uses.
    """
    path = _ROOT / "src" / "tilefoundry" / "utils" / "markdown.py"
    spec = importlib.util.spec_from_file_location("tilefoundry_utils_markdown", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, which is not there for a module loaded by
    # path alone.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


markdown = _markdown()
_ANCHORS: dict[Path, dict[str, str | None]] = {}


def anchors(document: Path) -> dict[str, str | None]:
    """Each of a document's anchors to the section number carrying it."""
    if document not in _ANCHORS:
        text = document.read_text(encoding="utf-8")
        _ANCHORS[document] = {
            heading.anchor: heading.number for heading in markdown.headings(text)
        }
    return _ANCHORS[document]


def _target_document(source: Path, path: str) -> Path | None:
    """The document a reference's target names, or None if it names none.

    A repository-root path is read from the root, anything else from beside the
    referring file. Which of the two a reference uses is a matter of where it is
    written -- relative inside `docs/spec`, root-relative from code -- but both
    resolve the same way wherever they appear, so the check does not need to know
    the convention to enforce the link. An empty path is the same document, which
    only means anything in markdown.
    """
    if not path:
        return source if source.suffix == ".md" else None
    if path.startswith("docs/"):
        return (_ROOT / path).resolve()
    return (source.parent / path).resolve()


def _outside_references(text: str, markdown: bool) -> str:
    """*text* with every reference, and every fenced block, blanked out.

    What is left is prose, so a `§` still standing in it is a bare reference.
    Blanking rather than deleting keeps every offset, so a line number computed
    against this string is a line number in the file. Inside a fence a `§` is
    example text, the same call a heading gets.
    """
    blanked = _REFERENCE.sub(lambda m: " " * len(m.group(0)), text)
    if not markdown:
        return blanked
    lines = blanked.splitlines(keepends=True)
    fenced = False
    for index, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
        elif fenced:
            lines[index] = " " * len(line)
    return "".join(lines)


def findings(path: Path) -> list[tuple[int, str, str]]:
    """Every reference in *path* that does not resolve, as (line, why, text)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # unreadable or binary: nothing to claim about it
    source = path.resolve()
    found = []
    for match in _REFERENCE.finditer(text):
        number = text.count("\n", 0, match.start()) + 1
        whole = _unwrap(match.group(0))
        why = _why(source, match.group(1), match.group(2))
        if why:
            found.append((number, why, whole))
    prose = _outside_references(text, path.suffix == ".md")
    for match in _BARE.finditer(prose):
        number = prose.count("\n", 0, match.start()) + 1
        found.append((
            number,
            f"bare `{match.group(0)}`; a reference MUST name its heading",
            text.splitlines()[number - 1].strip(),
        ))
    return sorted(found)


def _unwrap(text: str) -> str:
    """*text* as one line, whatever a reflow put at the start of the next.

    `clang-format` owns the C++ comments a reference sits in and will break one
    across lines wherever it likes, continuing with `//` or ` * `. Reading those
    as part of the display text would fail a reference that is correct and that
    no author can keep on one line. A markdown or Python wrap has no prefix and
    goes through the same path.
    """
    return _CONTINUATION.sub(" ", text).strip()


def _why(source: Path, display: str, target: str) -> str:
    """Why this reference is refused, or the empty string if it is not."""
    spelled = _DISPLAY.match(_unwrap(display))
    if spelled is None:
        return "display text is not `<doc> §<number>`"
    if "#" not in target:
        return "names no heading; the target needs a `#<anchor>`"
    path, _, fragment = target.partition("#")
    document = _target_document(source, path.strip())
    if document is None or not document.is_file():
        return f"target document {path!r} does not exist"
    named = spelled.group("doc")
    if named and named not in (document.stem, document.name):
        return f"display text names {named!r}, target names {document.stem!r}"
    found = anchors(document)
    if fragment not in found:
        return f"anchor '#{fragment}' names no heading in {document.name}"
    wanted = spelled.group("number")
    numbered = found[fragment]
    if numbered != wanted:
        names = f"§{numbered}" if numbered else "an unnumbered heading"
        return f"anchor names {names} of {document.name}, not §{wanted}"
    return ""


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: spec_refs_lint.py <path> ...  "
            "(the pre-commit hook passes the staged files)",
            file=sys.stderr,
        )
        return 2
    failed = False
    for name in argv:
        path = Path(name)
        for number, why, whole in findings(path):
            failed = True
            print(f"{path}:{number}: {why}")
            print(f"    {whole}")
    if failed:
        print(
            "\nA reference to a spec section MUST be written "
            "`[<doc> §<number>](<path>#<github-anchor>)`, with the anchor naming "
            "the very section the display text numbers -- `./<doc>.md` from "
            "inside docs/spec, `docs/spec/<doc>.md` from anywhere else. Ask "
            "`tilefoundry spec <topic>` for the sections a document has.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
