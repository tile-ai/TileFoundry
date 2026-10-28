"""Authored TIR entries are preserved, and missing entries are rejected by compile."""

from __future__ import annotations

from pathlib import Path

import pytest

import tilefoundry
from tests.core._corpus import Case, case, check_case, special
from tilefoundry.passes.transforms import LoweringError
from tilefoundry_nvidia.cuda.passes.transforms import insert_default_host_entry


class HostEntryCase(Case):
    """A module given to the default host-entry pass, and what the pass returns for it."""


@case
class ExistingHostEntry(HostEntryCase):
    """A module that already has a host entry is left alone."""

    path: Path = Path(__file__).parents[1] / "fixtures/tir/square.py"

    def guard(self, state, problems):
        assert insert_default_host_entry(state) is state


@case
class MissingHostEntry(HostEntryCase):
    """A hand-authored TIR module must supply its own CPU entry."""

    path: Path = Path(__file__).parents[1] / "fixtures/schedule/tir/loop_outputs.py"

    def guard(self, state, problems):
        assert insert_default_host_entry(state) is state
        with pytest.raises(ValueError, match=r"tilefoundry.build: Module.entry .* is not a CPU host entry"):
            tilefoundry.compile(state)


@case
class ChildHostWithoutRootHost(HostEntryCase):
    path: Path = Path(__file__).parents[1] / "fixtures/passes/child_host_without_root_host.py"
    name: str | None = "NoRootHost"

    def guard(self, state, problems):
        with pytest.raises(ValueError, match=r"Module.entry 'device' is not a CPU host entry"):
            tilefoundry.build(state)


@case
class AmbiguousHIRHost(HostEntryCase):
    path: Path = Path(__file__).parents[1] / "fixtures/passes/ambiguous_hir_host.py"
    name: str | None = "AmbiguousHIRHost"

    def guard(self, state, problems):
        with pytest.raises(LoweringError, match="host 'host' launches 2 HIR functions; expected exactly one"):
            tilefoundry.compile(state)


CASES = special(HostEntryCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_host_entry_case(case: HostEntryCase) -> None:
    check_case(case)
