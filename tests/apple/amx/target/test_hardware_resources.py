"""Pin schemas and identities for independent installed hardware documents.

A wrong hardware number silently prices a plan for a nonexistent machine. These
tests reject malformed documents, unmodelled facts, counterfeit measurements,
and numbers outside documents. Each failure retains a diagnostic useful to the
TOML author.
"""

from __future__ import annotations

import pytest

from tests.core.target._hardware import _BareArchitecture, _BareDevice
from tests.core.target._instructions import instruction_declarations
from tilefoundry_apple.amx.target import AmxTarget
from tilefoundry_apple.amx.target import spec as amx_spec


@pytest.mark.parametrize(
    ("construct", "message"),
    [
        pytest.param(
            lambda: AmxTarget(architecture=_BareArchitecture()),
            "AmxTarget.architecture must be an installed ID string, a document "
            "path, or an AppleAmx; got _BareArchitecture",
            id="amx-architecture",
        ),
        pytest.param(
            lambda: AmxTarget(_BareDevice()),
            "AmxTarget.device must be an installed ID string, a document path, "
            "or an AppleM2Pro; got _BareDevice",
            id="amx-device",
        ),
    ],
)
def test_document_backends_reject_bare_projection_value_markers(construct, message: str) -> None:
    with pytest.raises(TypeError) as rejected:
        construct()
    assert str(rejected.value) == message


def test_no_installed_number_is_repeated_as_a_python_default() -> None:
    """Keep hardware numbers exclusively in documents.

    Value classes declare shape, not content, and cannot be constructed without
    document values.
    """
    for value_type in (
        amx_spec.APPLE_AMX_ID,
        amx_spec.APPLE_M2_PRO_ID,
    ):
        hardware = AmxTarget.hardware
        built = hardware.resolve(value_type).value
        with pytest.raises(TypeError, match="required positional argument"):
            type(built)()


def test_architecture_bandwidth_evidence_distinguishes_known_from_unavailable() -> None:
    owners = {
        "apple.amx": {"register.owner": "amx"},
        "apple.m2_pro": {"memory.unified.owner": "target"},
    }
    for spec_id, expected_owners in owners.items():
        hardware = AmxTarget.hardware
        document = hardware.documents()[spec_id]
        assert not any("policy" in path for path in document.facts)
        assert not any("topology" in path for path in document.facts)
        assert {path: document.fact(path).value for path in expected_owners} == expected_owners
        for fact in document.facts.values():
            assert (fact.value is None) == (not fact.available)


def test_instruction_capabilities_and_execution_resources_are_separate_axes() -> None:
    """Enumerate instruction declarations and reconcile only their capability side."""
    instructions = tuple(
        (op_type, capability)
        for op_type, capability in instruction_declarations()
        if op_type.__module__.startswith("tilefoundry_apple.")
    )
    capabilities = {capability.name for _op_type, capability in instructions}
    architecture_capabilities = {
        capability
        for target in AmxTarget.available()
        for capability in target.architecture.capabilities
    }
    assert capabilities <= architecture_capabilities
