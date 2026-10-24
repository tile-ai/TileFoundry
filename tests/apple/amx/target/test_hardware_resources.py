"""Pin schemas and identities for independent installed hardware documents."""

from __future__ import annotations

import pytest

from tests.core.target._hardware import (
    bare_projection_cases,
    check_bare_projection,
    check_document_evidence,
    check_hardware_defaults,
)
from tests.core.target._instructions import check_architecture_capabilities
from tilefoundry_apple.amx.target import AmxTarget
from tilefoundry_apple.amx.target import spec as amx_spec


@pytest.mark.parametrize(
    ("construct", "message"),
    bare_projection_cases(
        lambda architecture: AmxTarget(architecture=architecture),
        AmxTarget,
        "amx",
        (
            "AmxTarget.architecture must be an installed ID string, a document path, or "
            "an AppleAmx; got _BareArchitecture",
            "AmxTarget.device must be an installed ID string, a document path, or an "
            "AppleM2Pro; got _BareDevice",
        ),
    ),
)
def test_document_backends_reject_bare_projection_value_markers(construct, message: str) -> None:
    check_bare_projection(construct, message)


def test_no_installed_number_is_repeated_as_a_python_default() -> None:
    check_hardware_defaults(AmxTarget.hardware, (amx_spec.APPLE_AMX_ID, amx_spec.APPLE_M2_PRO_ID))


def test_architecture_bandwidth_evidence_distinguishes_known_from_unavailable() -> None:
    owners = {
        "apple.amx": {"register.owner": "amx"},
        "apple.m2_pro": {"memory.unified.owner": "target"},
    }
    check_document_evidence(AmxTarget.hardware, owners)


def test_instruction_capabilities_and_execution_resources_are_separate_axes() -> None:
    """Enumerate instruction declarations and reconcile only their capability side."""
    check_architecture_capabilities(AmxTarget, "tilefoundry_apple.")
