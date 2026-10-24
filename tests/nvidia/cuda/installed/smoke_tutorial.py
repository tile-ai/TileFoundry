"""Tests and helpers for the owning target."""

from __future__ import annotations

import pytest

from tests.core.installed.smoke_tutorial import NOTEBOOK_PAGES, _asked_of_the_reader, _reproduce


@pytest.mark.parametrize("page", NOTEBOOK_PAGES)
def test_a_reader_reproduces_the_page_from_the_page(installation, tf, page, tmp_path) -> None:
    """Do what the page tells a reader to do; the outputs must be the page's own.

    Stronger than asserting the page contains some string, and it runs where the
    reader is: an installation, with no checkout to import and no ``scripts/`` to
    fall back on. The first block extracts the programs from the page itself, so
    every command after it is measuring the source the page displays.
    """
    text = tf("tutorial", page).stdout
    (tmp_path / f"{page}.md").write_text(text, encoding="utf-8")
    blocks = _asked_of_the_reader(text)
    assert blocks, f"{page}: the page asks the reader to run nothing"
    for index, match in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        _reproduce(
            match,
            text[match.end() : end],
            python=installation / "bin" / "python",
            cwd=tmp_path,
        )
