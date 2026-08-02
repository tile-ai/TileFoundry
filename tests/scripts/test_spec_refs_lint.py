"""The spec-reference checker, held to catching things and to not crying wolf.

The point of the grammar is that a reference breaks when the section it names
moves. So the central test edits a document and asserts the reference to it goes
red -- a checker that only ever runs over a settled tree cannot tell you it has
that property, because so would a checker that looks for nothing.

The refused shapes are the ones this repository actually accumulated: anchors
left behind by a renamed heading, anchors aimed at a subsection of the section
the text names, and links that never carried an anchor at all.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "spec_refs_lint.py"

#: A document with the shapes an anchor has to survive: a numbered heading, one
#: whose name has punctuation in it, an unnumbered subsection, a repeated name,
#: and a `#` line inside a fence that is not a heading at all.
TARGET = """\
# Title

## 1. First section

### 1.1 `ir/hir/nn/`

Text.

```python
## 9. Not a heading
```

## 2. Second section

#### `name`

## 3. Third section

#### `name`
"""


def _lint():
    """The checker, loaded from the script the hook runs."""
    spec = importlib.util.spec_from_file_location("spec_refs_lint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lint():
    return _lint()


@pytest.fixture
def docs(tmp_path):
    """A target document and an empty file beside it to refer to it from."""
    (tmp_path / "target.md").write_text(TARGET, encoding="utf-8")
    return tmp_path


def _check(lint, docs, body: str):
    source = docs / "source.md"
    source.write_text(body, encoding="utf-8")
    return lint.findings(source)


#: References that resolve, in every spelling the grammar allows.
ACCEPTED = [
    "[target §1](./target.md#1-first-section)",
    "[target.md §1](./target.md#1-first-section)",
    "[target §2](target.md#2-second-section)",
    "[target §1.1](./target.md#11-irhirnn)",
    "see ([target §3](./target.md#3-third-section)) for more",
    # Reflowed by an editor across a line break, still one reference.
    "[target\n§1](./target.md#1-first-section)",
]

#: References that do not, one per way of going wrong.
REFUSED = [
    # The heading was renamed and the anchor stayed behind.
    "[target §1](./target.md#1-first-section-renamed)",
    # Aimed at a subsection of the section the display text names.
    "[target §1](./target.md#11-irhirnn)",
    # Aimed at the parent of the section the display text names.
    "[target §1.1](./target.md#1-first-section)",
    # A number that no heading carries.
    "[target §7](./target.md#7-nowhere)",
    # No anchor at all: nothing in it can break when a heading is renamed.
    "[target §1](./target.md)",
    # The display text and the target name different documents.
    "[runtime §1](./target.md#1-first-section)",
    # A document that is not there.
    "[missing §1](./missing.md#1-first-section)",
    # Several numbers in one link cannot name one section.
    "[target §1 / §2](./target.md#1-first-section)",
]


@pytest.mark.parametrize("body", ACCEPTED, ids=range(len(ACCEPTED)))
def test_a_reference_that_resolves_is_left_alone(lint, docs, body) -> None:
    assert _check(lint, docs, body) == [], f"wrongly reported: {body}"


@pytest.mark.parametrize("body", REFUSED, ids=range(len(REFUSED)))
def test_a_reference_that_does_not_resolve_is_reported(lint, docs, body) -> None:
    found = _check(lint, docs, body)

    assert found, f"not reported: {body}"
    assert found[0][0] == 1


def test_renaming_a_heading_breaks_the_references_to_it(lint, docs) -> None:
    """The property the bare `§1.1` form cannot have.

    Nothing about the referring file changes -- only the heading it points at --
    and that alone has to be enough to fail.
    """
    source = docs / "source.md"
    source.write_text("[target §1](./target.md#1-first-section)\n", encoding="utf-8")
    assert lint.findings(source) == []

    target = docs / "target.md"
    target.write_text(
        TARGET.replace("## 1. First section", "## 1. First section, reworded"),
        encoding="utf-8",
    )
    lint._ANCHORS.clear()  # the run that saw the old heading is over

    found = lint.findings(source)

    assert [line for line, _, _ in found] == [1]


def test_renumbering_a_heading_breaks_the_references_to_it(lint, docs) -> None:
    """The other half: the name survives, the number the reader reads does not."""
    source = docs / "source.md"
    source.write_text("[target §2](./target.md#2-second-section)\n", encoding="utf-8")
    assert lint.findings(source) == []

    target = docs / "target.md"
    target.write_text(
        TARGET.replace("## 2. Second section", "## 4. Second section"),
        encoding="utf-8",
    )
    lint._ANCHORS.clear()

    assert lint.findings(source), "renumbering left the reference passing"


def test_a_heading_inside_a_fenced_block_is_not_one(lint, docs) -> None:
    """`## 9.` inside a fence is a comment, so nothing may point at it."""
    assert lint.markdown.anchor("9. Not a heading") == "9-not-a-heading"

    found = _check(lint, docs, "[target §9](./target.md#9-not-a-heading)")

    assert found, "a fenced line was treated as a heading"


def test_the_heading_scan_is_the_one_the_cli_uses(lint) -> None:
    """Both readers resolve a document the same way, or the lint passes
    documents the `spec` command cannot open."""
    from tilefoundry.cli import spec as cli_spec  # noqa: PLC0415

    text = (_ROOT / "docs" / "spec" / "tir.md").read_text(encoding="utf-8")
    numbered = {h.number for h in lint.markdown.headings(text) if h.number}

    assert numbered <= {section.key for section in cli_spec.sections(text)}


def test_the_shared_scan_stays_a_leaf(lint) -> None:
    """The hook loads it by path under an interpreter with nothing installed,
    so an import of anything but the standard library breaks the hook."""
    source = (_ROOT / "src" / "tilefoundry" / "utils" / "markdown.py").read_text(
        encoding="utf-8"
    )
    imported = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, re.MULTILINE)

    assert [name for name in imported if name.split(".")[0] == "tilefoundry"] == []


def test_an_unnumbered_subsection_cannot_be_pointed_at(lint, docs) -> None:
    """`#### \\`name\\`` sits under §2, so `§2` does not name it.

    The grammar addresses sections by number, so an unnumbered heading is not
    addressable -- pointing at one while calling it by its parent's number is
    the deep link the rule refuses.
    """
    assert _check(lint, docs, "[target §2](./target.md#name)")


def test_a_same_document_reference_resolves(lint, tmp_path) -> None:
    """The form M1 converts 132 references into: no path, only an anchor."""
    source = tmp_path / "alone.md"
    source.write_text(
        "## 1. Only section\n\n[§1](#1-only-section)\n", encoding="utf-8"
    )

    assert lint.findings(source) == []

    source.write_text(
        "## 2. Only section\n\n[§1](#1-only-section)\n", encoding="utf-8"
    )
    lint._ANCHORS.clear()

    assert lint.findings(source), "a same-document reference went unchecked"


def test_the_exit_status_and_the_report_name_the_line(lint, docs, capsys) -> None:
    """The hook's own contract: non-zero, and enough on stdout to go fix it."""
    source = docs / "source.md"
    source.write_text(
        "fine\n[target §1](./target.md#nowhere)\n", encoding="utf-8"
    )

    status = lint.main([str(source)])

    assert status == 1
    out = capsys.readouterr().out
    assert f"{source}:2:" in out
    assert "#nowhere" in out


def test_the_spec_is_clean(lint) -> None:
    """Every reference in every spec document, so the grammar is a fact about
    the tree and not only about the commit that happens to be in flight."""
    offenders = {
        document.name: lint.findings(document)
        for document in sorted((_ROOT / "docs" / "spec").glob("*.md"))
    }

    assert not {name: found for name, found in offenders.items() if found}
