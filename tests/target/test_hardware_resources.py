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

from tilefoundry.analysis.facts import MemoryHierarchyFacts, ThroughputFacts
from tilefoundry.ir.types import DType
from tilefoundry.target.amx import AmxTarget
from tilefoundry.target.amx import spec as amx_spec
from tilefoundry.target.base import Architecture, Device
from tilefoundry.target.cuda import CudaArchitecture, CudaDevice, CudaTarget
from tilefoundry.target.cuda import spec as cuda_spec
from tilefoundry.target.hardware import (
    DocumentFormatError,
    DuplicateRegistrationError,
    EvidenceFormatError,
    HardwareSpec,
    IncompatiblePairError,
    SchemaValidationError,
    UnknownDocumentError,
    UnknownSchemaError,
    hardware_documents,
    parse_document,
)

_SM90 = "nvidia.sm90"
_SM100 = "nvidia.sm100"
_H200 = "nvidia.h200_sxm"
_B200 = "nvidia.b200_sxm"
_EXTERNAL_HARDWARE = Path(__file__).parents[1] / "installed" / "smoke_target" / "hw"

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


def _hardware_text(resource: str) -> str:
    """The installed hardware document *resource* as authored."""
    hardware = Path(cuda_spec.__file__).parent / "hardware"
    return (hardware / resource).read_text(encoding="utf-8")


def _external_hardware_text(resource: str) -> str:
    """One external CUDA document used by the installed smoke workflow."""
    return (_EXTERNAL_HARDWARE / resource).read_text(encoding="utf-8")


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
    architecture = CudaTarget.hardware.documents()[_SM90]
    device = CudaTarget.hardware.documents()[_H200]
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


def test_a_second_cuda_product_composes_from_its_own_documents() -> None:
    """A CUDA product is a pair of documents and nothing else, so naming one
    device is enough to compose a target for hardware the compiler has never been
    pointed at before.

    Both products go through one document format per kind and build one value
    type per kind: every CUDA architecture document states the same fact paths
    and so does every device document, which is what lets one schema validate
    them all. What separates the two products is what their documents record, so
    a Blackwell target needs no type of its own to be a distinct value.
    """
    architecture = CudaTarget.hardware.documents()[_SM100]
    device = CudaTarget.hardware.documents()[_B200]
    assert device.compatibility == (_SM100,)
    assert architecture.schema == CudaTarget.hardware.documents()[_SM90].schema
    assert device.schema == CudaTarget.hardware.documents()[_H200].schema

    target = CudaTarget(_B200)
    hopper = CudaTarget(_H200)
    assert (target.architecture_id, target.device_id) == (_SM100, _B200)
    assert target.architecture_digest == architecture.digest
    assert target.device_digest == device.digest
    assert (target.arch, target.device.name) == ("sm_100", "b200_sxm")
    assert target.device.sm_count == 148
    assert type(target.architecture) is type(hopper.architecture) is CudaArchitecture
    assert type(target.device) is type(hopper.device) is CudaDevice
    assert target != hopper

    # A rate a product's tensor cores can reach, and the recorded absence of one
    # they cannot: the second is a statement about the hardware, so asking for it
    # fails rather than reading as an unpublished number.
    assert target.device.peak_for(DType.f4e2m1) == 9_000_000_000_000_000
    assert CudaTarget.hardware.documents()[_H200].fact("throughput.f4e2m1").status == (
        "unavailable"
    )
    with pytest.raises(ValueError, match="no dense compute-throughput entry"):
        hopper.device.peak_for(DType.f4e2m1)


def test_an_external_document_pair_stays_out_of_the_available_namespace(
    tmp_path: Path,
) -> None:
    """A complete custom pair keeps provenance without joining available IDs."""
    before = tuple(CudaTarget.hardware.documents())
    architecture_path = tmp_path / "vendor_sm70.toml"
    architecture_path.write_text(_external_hardware_text(architecture_path.name))
    device_path = tmp_path / "vendor_v100_sxm2_32gb.toml"
    device_path.write_text(_external_hardware_text(device_path.name))

    target = CudaTarget(device_path, architecture_path)
    assert target.architecture_id == "vendor.sm70"
    assert target.device_id == "vendor.v100_sxm2_32gb"
    assert re.fullmatch(r"[0-9a-f]{64}", target.architecture_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", target.device_digest)
    assert target.architecture.max_threads_per_cta == 1024
    assert target.device.sm_count == 80
    memory = target.get_facts(MemoryHierarchyFacts)
    throughput = target.get_facts(ThroughputFacts)
    assert memory.explicit("gmem").capacity_bytes == 32_000_000_000
    assert memory.explicit("tmem").capacity_bytes is None
    assert throughput.peak_for(DType.f16) == 125_000_000_000_000
    assert throughput.peak_for(DType.bf16) is None
    architecture_document, device_document = hardware_documents(target)
    assert architecture_document.id == target.architecture_id
    assert device_document.id == target.device_id
    assert architecture_document.digest == target.architecture_digest
    assert device_document.digest == target.device_digest
    for document in (architecture_document, device_document):
        for fact in document.facts.values():
            if fact.available:
                assert fact.origin and fact.source and fact.conditions
    for document, path in (
        (architecture_document, "memory.tensor.per_cta"),
        (device_document, "throughput.bf16"),
        (device_document, "throughput.fp8e4m3"),
        (device_document, "throughput.f4e2m1"),
    ):
        fact = document.fact(path)
        assert fact.status == "unavailable" and fact.conditions
    assert tuple(CudaTarget.hardware.documents()) == before
    with pytest.raises(UnknownDocumentError):
        CudaTarget.hardware.resolve("vendor.sm70")


def test_external_cuda_documents_reject_incompatibility_and_a_missing_leaf(
    tmp_path: Path,
) -> None:
    architecture = _external_hardware_text("vendor_sm70.toml")
    architecture_path = tmp_path / "vendor_sm70.toml"
    architecture_path.write_text(architecture)
    device = _external_hardware_text("vendor_v100_sxm2_32gb.toml")
    device_path = tmp_path / "vendor_v100_sxm2_32gb.toml"
    device_path.write_text(device)
    incompatible_path = tmp_path / "incompatible_v100.toml"
    incompatible_path.write_text(
        device.replace(
            'architectures = ["vendor.sm70"]',
            'architectures = ["vendor.other"]',
        )
    )

    with pytest.raises(
        IncompatiblePairError,
        match=r"declares compatibility with \['vendor.other'\], not 'vendor.sm70'",
    ):
        CudaTarget(incompatible_path, architecture_path)

    missing_path = tmp_path / "missing_tensor_memory.toml"
    missing_path.write_text(
        architecture.replace(
            '\n[facts.memory.tensor.per_cta]\nstatus = "unavailable"\n'
            'conditions = "Volta Tensor Cores accumulate into registers and provide '
            'no separately addressable tensor-memory store."\n',
            "\n",
        )
    )
    with pytest.raises(
        SchemaValidationError,
        match=r"required fact 'memory.tensor.per_cta' is missing",
    ):
        CudaTarget(device_path, missing_path)


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
    stray = _hardware_text("nvidia_sm90.toml") + (
        '\n[facts.compute.max_threads_per_cluster]\n'
        'value = 8\nunit = "count"\norigin = "vendor"\n'
        'source = "test"\nconditions = "test"\n'
    )
    with pytest.raises(SchemaValidationError, match="unknown facts for schema"):
        cuda_spec.build_cuda_architecture(parse_document(stray, origin_label="stray"))


def test_a_memory_owner_must_use_the_target_vocabulary() -> None:
    document = _hardware_text("nvidia_h200_sxm.toml").replace(
        'value = "target"', 'value = "warp"'
    )

    with pytest.raises(SchemaValidationError, match=r"memory owner 'warp'.*target.*cta.*thread"):
        cuda_spec.build_cuda_device(parse_document(document, origin_label="bad-owner"))


def test_resolution_failures_are_each_distinguishable() -> None:
    """Unknown IDs, schemas, duplicate IDs, and incompatible pairs differ."""
    with pytest.raises(
        UnknownDocumentError, match="no hardware document 'nvidia.sm70'"
    ):
        CudaTarget("nvidia.sm70")

    with pytest.raises(UnknownDocumentError) as cross_backend:
        CudaTarget("apple.m2_pro")
    assert str(cross_backend.value) == (
        "CudaTarget.device 'apple.m2_pro' is a hardware document owned by "
        "AmxTarget, not CudaTarget"
    )

    with pytest.raises(UnknownDocumentError) as wrong_kind:
        CudaTarget(_SM90)
    assert str(wrong_kind.value) == (
        "CudaTarget.device got hardware document 'nvidia.sm90', which builds "
        "CudaArchitecture; expected CudaDevice"
    )

    document = CudaTarget.hardware.documents()[_SM90]
    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        CudaTarget.hardware.adopt(document)

    unsupported = HardwareSpec("tilefoundry.target.cuda.hardware", {})
    with pytest.raises(UnknownSchemaError, match="unsupported schema"):
        unsupported.resolve(_SM90)

    with pytest.raises(IncompatiblePairError, match="declares compatibility with"):
        CudaTarget(_H200, architecture=_SM100)


class _BareArchitecture(Architecture):
    pass


class _BareDevice(Device):
    pass


@pytest.mark.parametrize(
    ("construct", "message"),
    [
        pytest.param(
            lambda: CudaTarget(_H200, architecture=_BareArchitecture()),
            "CudaTarget.architecture must be an installed ID string, a document "
            "path, or a CudaArchitecture; got _BareArchitecture",
            id="cuda-architecture",
        ),
        pytest.param(
            lambda: CudaTarget(_BareDevice()),
            "CudaTarget.device must be an installed ID string, a document path, "
            "or a CudaDevice; got _BareDevice",
            id="cuda-device",
        ),
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
def test_document_backends_reject_bare_projection_value_markers(
    construct, message: str
) -> None:
    with pytest.raises(TypeError) as rejected:
        construct()
    assert str(rejected.value) == message


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
        hardware = AmxTarget.hardware if value_type.startswith("apple.") else CudaTarget.hardware
        built = hardware.resolve(value_type).value
        with pytest.raises(TypeError, match="required positional argument"):
            type(built)()

    # The one figure that had no installed record is now recorded, so the
    # public peak stays available without a Python literal behind it.
    assert CudaTarget("nvidia.h200_sxm").device.peak_for(DType.f32) == 67_000_000_000_000
    assert CudaTarget.hardware.documents()[_H200].fact("throughput.f32").origin == "vendor"


def test_an_unavailable_fact_omits_its_value_and_says_why() -> None:
    """AC-1-5. An unavailable fact is recorded as such rather than carrying a
    placeholder string, and hardware documents hold no compiler policy."""
    architecture = CudaTarget.hardware.documents()[_SM90]
    unavailable = architecture.fact("memory.shared.bandwidth")

    assert not unavailable.available
    assert unavailable.value is None
    assert unavailable.status == "unavailable"
    assert unavailable.conditions

    owners = {
        _SM90: {
            "memory.shared.owner": "cta",
            "memory.register.owner": "thread",
            "memory.tensor.owner": "cta",
        },
        _SM100: {
            "memory.shared.owner": "cta",
            "memory.register.owner": "thread",
            "memory.tensor.owner": "cta",
        },
        _H200: {"memory.hbm.owner": "target"},
        _B200: {"memory.hbm.owner": "target"},
        "apple.amx": {"register.owner": "amx"},
        "apple.m2_pro": {"memory.unified.owner": "target"},
    }
    for spec_id, expected_owners in owners.items():
        hardware = AmxTarget.hardware if spec_id.startswith("apple.") else CudaTarget.hardware
        document = hardware.documents()[spec_id]
        assert not any("policy" in path for path in document.facts)
        assert not any("topology" in path for path in document.facts)
        assert {
            path: document.fact(path).value for path in expected_owners
        } == expected_owners
        for fact in document.facts.values():
            assert (fact.value is None) == (not fact.available)
