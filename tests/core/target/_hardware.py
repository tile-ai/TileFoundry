"""Hardware projection markers and checks for document-backed targets."""

from __future__ import annotations

import pytest

from tilefoundry.target.base import Architecture, Device


class _BareArchitecture(Architecture):
    pass


class _BareDevice(Device):
    pass


def check_bare_projection(construct, message: str) -> None:
    with pytest.raises(TypeError) as rejected:
        construct()
    assert str(rejected.value) == message


def check_hardware_defaults(hardware, value_types) -> None:
    """Keep hardware numbers exclusively in documents."""
    for value_type in value_types:
        built = hardware.resolve(value_type).value
        with pytest.raises(TypeError, match="required positional argument"):
            type(built)()


def check_document_evidence(hardware, owners) -> None:
    for spec_id, expected_owners in owners.items():
        document = hardware.documents()[spec_id]
        assert not any("policy" in path for path in document.facts)
        assert not any("topology" in path for path in document.facts)
        assert {path: document.fact(path).value for path in expected_owners} == expected_owners
        for fact in document.facts.values():
            assert (fact.value is None) == (not fact.available)


def bare_projection_cases(architecture, device, prefix: str, messages):
    return (
        pytest.param(
            lambda: architecture(_BareArchitecture()), messages[0], id=f"{prefix}-architecture"
        ),
        pytest.param(lambda: device(_BareDevice()), messages[1], id=f"{prefix}-device"),
    )
