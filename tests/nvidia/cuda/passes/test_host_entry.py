"""What a module with no host entry gets, and what it keeps.

Both device shapes take one route: the callee's own shape decides whether the
call reads as a launch or as a dispatch, and no function's target is rewritten
on the way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.passes.transforms import insert_default_host_entry

DEVICE_ENTRIES = Path(__file__).parents[1] / "fixtures" / "passes" / "device_entries_without_host.py"


def _launch_of(mod):
    """The one ``Launch`` the synthesized entry's body is made of."""
    (statement,) = mod.entry_function().body.body
    assert isinstance(statement, Evaluate)
    assert isinstance(statement.callable, Launch)
    return statement


def _geometry(evaluate) -> tuple[tuple[int, ...], tuple[int, ...]]:
    extents = tuple(int(arg.value) for arg in evaluate.args[1:7])
    return extents[:3], extents[3:]


class HostEntryCase(Case):
    """A module given to the default host-entry pass, and what the pass returns for it."""


class DeviceEntry(HostEntryCase):
    """A device entry with no host entry, launched on one block of 128 threads."""

    def guard(self, state, problems):
        problems.run("gains a host entry that launches it", self.gains_a_host_entry, state)
        problems.run("leaves every recorded target alone", self.leaves_targets_alone, state)
        problems.run("reads its geometry", self.reads_its_geometry, state)

    def gains_a_host_entry(self, authored):
        lowered = insert_default_host_entry(authored)
        entry = lowered.entry_function()
        assert lowered.entry == "main"
        assert isinstance(entry.target, CpuTarget)
        assert [p.name for p in entry.params] == ["x"]
        assert _launch_of(lowered).args[0].name == authored.entry

    def leaves_targets_alone(self, authored):
        lowered = insert_default_host_entry(authored)
        assert isinstance(lowered.lookup(authored.entry).target, CudaTarget)

    def reads_its_geometry(self, authored):
        assert _geometry(_launch_of(insert_default_host_entry(authored))) == (
            (1, 1, 1),
            (128, 1, 1),
        )


@case
class LoneKernel(DeviceEntry):
    """A lone kernel reads its geometry off its own body."""

    path: Path = DEVICE_ENTRIES
    name: str | None = "_OneKernel"


@case
class Prototype(DeviceEntry):
    """The prototype's own body is empty, so a launch of it would run one thread.

    It reads its geometry off the variants that have a body.
    """

    path: Path = DEVICE_ENTRIES
    name: str | None = "_Prototype"


@case
class ExistingHostEntry(HostEntryCase):
    """A module that already has a host entry is left alone."""

    path: Path = Path(__file__).parents[1] / "fixtures/tir/square.py"

    def guard(self, state, problems):
        assert insert_default_host_entry(state) is state


CASES = special(HostEntryCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_host_entry_case(case: HostEntryCase) -> None:
    check_case(case)
