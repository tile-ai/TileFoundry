"""Generated Parser Spec sections stay inside their ownership markers."""

from __future__ import annotations

import pytest

from tilefoundry.parser.spec import render_parser_document

_DOCUMENT = """# Hand-written heading

kept before grammar

<!-- parser-grammar:start -->
stale grammar
<!-- parser-grammar:end -->

kept between sections

<!-- parser-constraints:start -->
stale constraints
<!-- parser-constraints:end -->

kept after constraints
"""


def test_parser_spec_generation_only_replaces_marked_sections() -> None:
    rendered = render_parser_document(_DOCUMENT)

    assert rendered.startswith("# Hand-written heading\n\nkept before grammar\n")
    assert "\nkept between sections\n" in rendered
    assert rendered.endswith("\nkept after constraints\n")
    assert "stale grammar" not in rendered
    assert "stale constraints" not in rendered


@pytest.mark.parametrize(
    "document",
    (
        _DOCUMENT.replace("<!-- parser-grammar:start -->", ""),
        _DOCUMENT.replace(
            "<!-- parser-constraints:end -->",
            "<!-- parser-constraints:end -->\n<!-- parser-constraints:end -->",
        ),
        _DOCUMENT.replace("<!-- parser-grammar:start -->", "GRAMMAR-MARKER").replace(
            "<!-- parser-grammar:end -->",
            "<!-- parser-grammar:start -->",
        ).replace("GRAMMAR-MARKER", "<!-- parser-grammar:end -->"),
    ),
)
def test_parser_spec_generation_rejects_missing_or_repeated_markers(document: str) -> None:
    with pytest.raises(ValueError, match="exactly one|must precede"):
        render_parser_document(document)
