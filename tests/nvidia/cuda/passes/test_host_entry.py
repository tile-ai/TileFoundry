"""Authored TIR entries are preserved, and missing entries are rejected by compile."""

from __future__ import annotations

from pathlib import Path

import pytest

import tilefoundry
from tests.core._corpus import Case, case, check_case, special
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core.module import Module, ModuleList
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


CASES = special(HostEntryCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_host_entry_case(case: HostEntryCase) -> None:
    check_case(case)


def test_build_does_not_use_a_child_host_as_the_root():
    child = load_authored_ir(str(ExistingHostEntry().path))
    root = Module(
        "NoRootHost", (child.lookup("square_device"),), "square_device",
        target=child.target, topologies=child.topologies, children={"child": ModuleList(child)},
    )
    with pytest.raises(ValueError, match=r"Module.entry 'square_device' is not a CPU host entry"):
        tilefoundry.build(root)
