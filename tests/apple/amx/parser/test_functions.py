"""Source-span checks for Apple fixture programs."""

from pathlib import Path

import pytest

from tests.core.parser._fixtures import check_parsed_call_sources

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
_FIXTURE_SOURCES = tuple(
    path for path in sorted(_FIXTURE_ROOT.rglob("*.py")) if path.name != "__init__.py"
)


@pytest.mark.parametrize(
    "source", _FIXTURE_SOURCES, ids=lambda source: str(source.relative_to(_FIXTURE_ROOT))
)
def test_every_parsed_call_knows_where_it_came_from(source: Path) -> None:
    check_parsed_call_sources(source)
