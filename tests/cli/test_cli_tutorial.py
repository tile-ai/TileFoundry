"""Tutorial command workflows."""

from __future__ import annotations

import pytest

from tilefoundry import cli

# Named here rather than read from `tutorial.PAGES`: the constant is part of what
# these tests check, so deriving the cases from it would let a page go missing
# from both at once.
PAGES = ("migrate", "run", "optimize")


def test_the_overview_names_the_pages_and_reference_commands(capsys) -> None:
    """A bare tutorial names its pages and the commands it delegates to."""
    assert cli.main(["tutorial"]) == 0
    reported = capsys.readouterr().out

    assert "source to source" in reported
    for page in PAGES:
        assert page in reported
    assert "tilefoundry spec" in reported
    assert "tilefoundry check --help" in reported


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page, capsys) -> None:
    """A rendered page has no unresolved source directive."""
    assert cli.main(["tutorial", page]) == 0
    reported = capsys.readouterr().out

    assert reported.startswith("# ")
    assert "{{fixture:" not in reported
