"""Cover print-import-print fixed points beyond the model corpus."""

from pathlib import Path

import pytest

from tests.core._source import import_dsl
from tests.nvidia.cuda.schedule.test_fixtures import PLAIN_REFUSED
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script


@pytest.mark.parametrize(
    "path",
    sorted(
        path for path in (Path(__file__).parents[1] / "fixtures" / "schedule").rglob("*.py")
        if path.name != "__init__.py"
        and not (path.parent.name == "plain" and path.stem in PLAIN_REFUSED)
    ),
    ids=lambda path: str(path.relative_to(Path(__file__).parents[1] / "fixtures")),
)
def test_schedule_fixtures_roundtrip(path: Path) -> None:
    program = import_dsl(path.read_text())
    import_dsl(as_script(program))
