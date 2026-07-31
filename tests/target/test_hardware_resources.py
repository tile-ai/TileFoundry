"""Installed hardware resources: independent documents, exact schemas, and
the identity a compiled artifact can name them by.

A hardware number that is wrong is not a crash, it is a plan priced against a
machine nobody has, so what is checked here is the boundary that lets a wrong
number in: whether a malformed document loads, whether a fact nobody modelled is
accepted as a fact, whether an absent measurement can pass as a present one, and
whether a number can be written anywhere other than the document. Each failure
keeps its own diagnostic because the reader of one is somebody editing a TOML
file by hand.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from tilefoundry.ir.types import DType
from tilefoundry.target.amx import AmxTarget
from tilefoundry.target.amx import spec as amx_spec
from tilefoundry.target.cuda import CudaTarget
from tilefoundry.target.cuda import spec as cuda_spec
from tilefoundry.target.hardware import (
    HARDWARE_SPECS,
    DocumentFormatError,
    DuplicateRegistrationError,
    EvidenceFormatError,
    HardwareSpecRegistry,
    IncompatiblePairError,
    SchemaValidationError,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)

_SM90 = "nvidia.sm90"
_H200 = "nvidia.h200_sxm"

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


def test_a_target_retains_the_identity_and_digest_of_what_it_resolved() -> None:
    """AC-1-1 and AC-1-3. Each side is a complete document in its own right, and
    a target is the pair composed through a declared compatibility rather than one
    combined record -- neither document carries the other's facts.

    The ID and content digest of each travel with the composed value, and editing
    a document's content changes its digest, which is what lets a compiled
    artifact name the hardware it was built against. Identity stays out of fact
    equality, though: two targets carrying identical facts are equal however each
    was obtained, because codegen groups CUDA functions by comparing their
    targets, and letting the resolved ID into equality would report identical
    hardware as "differing Target facts".
    """
    architecture = HARDWARE_SPECS.document(_SM90)
    device = HARDWARE_SPECS.document(_H200)
    assert (architecture.kind, device.kind) == ("architecture", "device")
    assert device.compatibility == (_SM90,)
    assert not any(path.startswith("throughput.") for path in architecture.facts)
    assert not any(path.startswith("instruction.") for path in device.facts)

    target = CudaTarget("nvidia.h200_sxm")
    assert (target.architecture_id, target.device_id) == (_SM90, _H200)
    assert target.architecture_digest == architecture.digest
    assert re.fullmatch(r"[0-9a-f]{64}", target.device_digest)
    assert target.architecture.max_threads_per_cta == 1024
    assert target.device.sm_count == 132

    original = parse_document(_MINIMAL_DEVICE, origin_label="test")
    edited = parse_document(
        _MINIMAL_DEVICE.replace("value = 4", "value = 5"), origin_label="test"
    )
    assert original.digest != edited.digest

    with pytest.raises(TypeError):
        CudaTarget()
    with pytest.raises(ValueError, match="carries no document"):
        CudaTarget(target.device)

    supplied = CudaTarget(device=target.device, architecture=target.architecture)
    assert supplied.architecture_id is None and supplied.device_id is None
    assert supplied.architecture_digest is None
    assert target == supplied
    assert hash(target) == hash(supplied)
    # Differing facts still separate them.
    assert target != CudaTarget(
        "nvidia.h200_sxm",
        architecture=replace(target.architecture, name="other"),
    )


def test_an_explicitly_loaded_document_stays_out_of_the_installed_namespace(
    tmp_path: Path,
) -> None:
    """AC-1-3. A custom document is loaded by path and must be complete; it
    cannot shadow or join the installed IDs."""
    registry = HardwareSpecRegistry()
    cuda_spec.install(registry)
    before = registry.installed_ids()

    custom = tmp_path / "custom_sm90.toml"
    text = (
        Path(cuda_spec.__file__).parent.parent
        / "hardware"
        / "nvidia_sm90.toml"
    ).read_text(encoding="utf-8")
    custom.write_text(text.replace('id = "nvidia.sm90"', 'id = "vendor.sm90_custom"'))

    resolved = registry.load_path(custom)
    assert resolved.id == "vendor.sm90_custom"
    assert resolved.value.max_threads_per_cta == 1024
    assert registry.installed_ids() == before
    with pytest.raises(UnknownDocumentError):
        registry.resolve("vendor.sm90_custom")


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
            # A bare string is iterable, so an unchecked tuple() would silently
            # turn one ID into a tuple of its characters.
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
    """AC-1-2. One case per class of malformed shape -- the envelope, one fact's
    evidence, and the compatibility list -- each with its own diagnostic rather
    than a generic parse failure."""
    with pytest.raises(error, match=message):
        parse_document(_document(old, new), origin_label="test")


def test_a_schema_rejects_a_fact_it_does_not_model() -> None:
    """AC-1-2. An unknown key under ``facts`` is a spelling mistake, not an
    unused fact, so a document carrying one does not load."""
    text = (
        Path(cuda_spec.__file__).parent.parent / "hardware" / "nvidia_sm90.toml"
    ).read_text(encoding="utf-8")
    stray = text + (
        '\n[facts.compute.max_threads_per_cluster]\n'
        'value = 8\nunit = "count"\norigin = "vendor"\n'
        'source = "test"\nconditions = "test"\n'
    )
    with pytest.raises(SchemaValidationError, match="unknown facts for schema"):
        cuda_spec.build_sm90(parse_document(stray, origin_label="stray"))


def test_registration_and_resolution_failures_are_each_distinguishable() -> None:
    """AC-1-2. Unknown IDs, unknown schemas, duplicate registrations, and
    incompatible pairs are separate diagnostics."""
    registry = HardwareSpecRegistry()
    cuda_spec.install(registry)

    with pytest.raises(UnknownDocumentError, match="no installed hardware document"):
        registry.resolve("nvidia.sm100")
    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        cuda_spec.install(registry)
    with pytest.raises(DuplicateRegistrationError, match="already installed"):
        registry.install(_SM90, "tilefoundry.target.hardware", "nvidia_sm90.toml")

    unschemed = HardwareSpecRegistry()
    unschemed.install("test.device", "tilefoundry.target.hardware", "nvidia_sm90.toml")
    with pytest.raises(UnknownDocumentError, match="declares id="):
        unschemed.resolve("test.device")

    bare = HardwareSpecRegistry()
    bare.install(_SM90, "tilefoundry.target.hardware", "nvidia_sm90.toml")
    with pytest.raises(UnknownSchemaError, match="no registered schema"):
        bare.resolve(_SM90)

    with pytest.raises(IncompatiblePairError, match="declares compatibility with"):
        AmxTarget(architecture="apple.amx", device=_H200)


def test_no_installed_number_is_repeated_as_a_python_default() -> None:
    """AC-1-4. The documents are the only place a hardware number is written:
    the value classes declare shape, not content, so none of them can be
    constructed without one."""
    for value_type in (
        cuda_spec.SM90_ID,
        cuda_spec.H200_SXM_ID,
        amx_spec.APPLE_AMX_ID,
        amx_spec.APPLE_M2_PRO_ID,
    ):
        built = HARDWARE_SPECS.resolve(value_type).value
        with pytest.raises(TypeError, match="required positional argument"):
            type(built)()

    # The one figure that had no installed record is now recorded, so the
    # public peak stays available without a Python literal behind it.
    assert CudaTarget("nvidia.h200_sxm").device.peak_for(DType.f32) == 67_000_000_000_000
    assert HARDWARE_SPECS.document(_H200).fact("throughput.f32").origin == "vendor"


def test_an_unavailable_fact_omits_its_value_and_says_why() -> None:
    """AC-1-5. An unavailable fact is recorded as such rather than carrying a
    placeholder string, and hardware documents hold no compiler policy."""
    architecture = HARDWARE_SPECS.document(_SM90)
    unavailable = architecture.fact("memory.shared.bandwidth")

    assert not unavailable.available
    assert unavailable.value is None
    assert unavailable.status == "unavailable"
    assert unavailable.conditions

    for spec_id in (_SM90, _H200, "apple.amx", "apple.m2_pro"):
        document = HARDWARE_SPECS.document(spec_id)
        assert not any("policy" in path for path in document.facts)
        assert not any("topology" in path for path in document.facts)
        for fact in document.facts.values():
            assert (fact.value is None) == (not fact.available)
