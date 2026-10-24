"""Hardware schema diagnostics and target-neutral capability boundaries."""

from __future__ import annotations

import pytest

from tests.core.target._instructions import instruction_declarations
from tilefoundry.target.hardware import DocumentFormatError, EvidenceFormatError, parse_document


def test_instruction_capabilities_and_execution_resources_are_separate_axes() -> None:
    """Enumerate instruction declarations and reconcile only their capability side."""
    instructions = instruction_declarations()
    capabilities = {capability.name for _op_type, capability in instructions}
    resources = {
        resource
        for op_type, capability in instructions
        if isinstance(
            (resource := getattr(capability.declaration or op_type, "resource", None)), str
        )
    }
    assert instructions
    assert capabilities.isdisjoint(resources)
    assert all(resource.endswith("_engine") for resource in resources)

_MINIMAL_DEVICE = """
[spec]
schema = "test.device/v1"
kind = "device"
id = "test.device"

[compatibility]
architectures = ["test.arch"]

[facts.compute.count]
value = 4
unit = "count"
origin = "vendor"
source = "test"
conditions = "test"
"""


def _document(old: str, new: str) -> str:
    """The minimal device document with one substitution applied."""
    assert old in _MINIMAL_DEVICE
    return _MINIMAL_DEVICE.replace(old, new)


@pytest.mark.parametrize(
    ("old", "new", "error", "message"),
    [
        pytest.param(
            'kind = "device"',
            'kind = "accelerator"',
            DocumentFormatError,
            "kind must be one of",
            id="unknown-kind",
        ),
        pytest.param(
            "value = 4",
            'value = 4\nstatus = "unavailable"',
            EvidenceFormatError,
            "either 'value' or 'status'",
            id="value-and-status",
        ),
        pytest.param(
            'architectures = ["test.arch"]',
            'architectures = "test.arch"',
            DocumentFormatError,
            "must be a list of non-empty ID strings",
            id="compatibility-bare-string",
        ),
    ],
)
def test_a_malformed_document_names_exactly_what_is_wrong(
    old: str, new: str, error: type[Exception], message: str
) -> None:
    """Give each malformed shape its own diagnostic.

    Cases cover the envelope, fact evidence, and compatibility list rather than
    collapsing them into a generic parse failure.
    """
    with pytest.raises(error, match=message):
        parse_document(_document(old, new), origin_label="test")
