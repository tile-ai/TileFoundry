"""The viewer builds its DOT graph for the placed rmsnorm programs, without a server."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.inspection.viewer.builder import ViewerBuilder

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


class ViewerCase(Case):
    """A program whose entry function the viewer draws."""

    def guard(self, state, problems):
        graph = ViewerBuilder(state.entry_function(), collapsed=set()).build()

        assert graph.source.strip()


@case
class Rmsnorm(ViewerCase):
    path: Path = PLACED / "rmsnorm.py"
    name: str | None = "RmsnormModule"


@case
class RmsnormSeq2(ViewerCase):
    path: Path = PLACED / "rmsnorm_seq2.py"
    name: str | None = "RmsnormSeq2Module"


@case
class RmsnormQuantSeq2(ViewerCase):
    path: Path = PLACED / "rmsnorm_quant_seq2.py"
    name: str | None = "RmsnormQuantSeq2Module"


CASES = special(ViewerCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_viewer_case(case: ViewerCase) -> None:
    check_case(case)
