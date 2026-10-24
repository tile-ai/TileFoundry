"""Source-span checks for Apple fixture programs."""

from pathlib import Path

from tests.core.parser._fixtures import check_parsed_call_sources, fixture_cases


@fixture_cases(__file__)
def test_every_parsed_call_knows_where_it_came_from(source: Path) -> None:
    check_parsed_call_sources(source)
