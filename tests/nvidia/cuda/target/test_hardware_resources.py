"""Pin schemas and identities for independent installed hardware documents.

A wrong hardware number silently prices a plan for a nonexistent machine. These
tests reject malformed documents, unmodelled facts, counterfeit measurements,
and numbers outside documents. Each failure retains a diagnostic useful to the
TOML author.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from tests.core.target._hardware import (
    bare_projection_cases,
    check_bare_projection,
    check_document_evidence,
    check_hardware_defaults,
)
from tests.core.target._instructions import check_architecture_capabilities
from tests.core.target.test_hardware_resources import _MINIMAL_DEVICE
from tilefoundry import target
from tilefoundry.ir import expr
from tilefoundry.target import hardware
from tilefoundry_nvidia.cuda import target as cuda_target
from tilefoundry_nvidia.cuda.target import spec

_SM89 = "nvidia.sm89"
_SM90 = "nvidia.sm90"
_SM100 = "nvidia.sm100"
_RTX_6000_ADA = "nvidia.rtx_6000_ada"
_H200 = "nvidia.h200_sxm"
_B200 = "nvidia.b200_sxm"
_EXTERNAL_HARDWARE = Path(__file__).parents[1] / "installed" / "smoke_target" / "hw"


def _hardware_text(resource: str) -> str:
    """The installed hardware document *resource* as authored."""
    hardware_value = Path(spec.__file__).parent / "hardware"
    return (hardware_value / resource).read_text(encoding="utf-8")


def _external_hardware_text(resource: str) -> str:
    """One external CUDA document used by the installed smoke workflow."""
    return (_EXTERNAL_HARDWARE / resource).read_text(encoding="utf-8")


def test_a_target_retains_the_identity_and_digest_of_what_it_resolved() -> None:
    """A target retains each resolved document's identity and digest.

    A target composes two complete documents through declared compatibility.
    Their IDs and content digests travel with the value so artifacts can name
    their hardware. Identity stays out of fact equality, allowing equivalent
    supplied and registry-resolved targets to group during codegen.
    """
    architecture = cuda_target.Target.hardware.documents()[_SM90]
    device = cuda_target.Target.hardware.documents()[_H200]
    assert (architecture.kind, device.kind) == ("architecture", "device")
    assert device.compatibility == (_SM90,)
    assert not any(path.startswith("throughput.") for path in architecture.facts)
    assert not any(path.startswith("instruction.") for path in device.facts)

    tgt = cuda_target.Target("nvidia.h200_sxm")
    assert (tgt.architecture_id, tgt.device_id) == (_SM90, _H200)
    assert tgt.architecture_digest == architecture.digest
    assert re.fullmatch(r"[0-9a-f]{64}", tgt.device_digest)
    assert tgt.architecture.max_threads_per_cta == 1024
    assert tgt.device.sm_count == 132

    original = hardware.parse_document(_MINIMAL_DEVICE, origin_label="test")
    edited = hardware.parse_document(_MINIMAL_DEVICE.replace("value = 4", "value = 5"), origin_label="test")
    assert original.digest != edited.digest

    with pytest.raises(TypeError):
        cuda_target.Target()
    with pytest.raises(ValueError, match="carries no document"):
        cuda_target.Target(tgt.device)

    supplied = cuda_target.Target(device=tgt.device, architecture=tgt.architecture)
    assert supplied.architecture_id is None and supplied.device_id is None
    assert supplied.architecture_digest is None
    assert tgt == supplied
    assert hash(tgt) == hash(supplied)

    assert tgt != cuda_target.Target(
        "nvidia.h200_sxm",
        architecture=replace(tgt.architecture, name="other"),
    )


def test_a_second_cuda_product_composes_from_its_own_documents() -> None:
    """A CUDA product is a pair of documents and nothing else.

    Naming a device composes its compatible target. Products share one document
    format and value type per kind; their recorded facts distinguish them, so a
    new architecture needs no new Python target type.
    """
    architecture = cuda_target.Target.hardware.documents()[_SM100]
    device = cuda_target.Target.hardware.documents()[_B200]
    assert device.compatibility == (_SM100,)
    assert architecture.schema == cuda_target.Target.hardware.documents()[_SM90].schema
    assert device.schema == cuda_target.Target.hardware.documents()[_H200].schema

    tgt = cuda_target.Target(_B200)
    hopper = cuda_target.Target(_H200)
    assert (tgt.architecture_id, tgt.device_id) == (_SM100, _B200)
    assert tgt.architecture_digest == architecture.digest
    assert tgt.device_digest == device.digest
    assert (tgt.arch, tgt.device.name) == ("sm_100", "b200_sxm")
    assert tgt.device.sm_count == 148
    assert type(tgt.architecture) is type(hopper.architecture) is cuda_target.Architecture
    assert type(tgt.device) is type(hopper.device) is cuda_target.Device
    assert tgt != hopper

    assert tgt.device.peak_for(expr.DType.f4e2m1) == 9_000_000_000_000_000
    assert cuda_target.Target.hardware.documents()[_H200].fact("throughput.f4e2m1").status == (
        "unavailable"
    )
    with pytest.raises(ValueError, match="no dense compute-throughput entry"):
        hopper.device.peak_for(expr.DType.f4e2m1)


def test_an_external_document_pair_stays_out_of_the_available_namespace(
    tmp_path: Path,
) -> None:
    """A complete custom pair keeps provenance without joining available IDs."""
    before = tuple(cuda_target.Target.hardware.documents())
    architecture_path = tmp_path / "vendor_sm70.toml"
    architecture_path.write_text(_external_hardware_text(architecture_path.name))
    device_path = tmp_path / "vendor_v100_sxm2_32gb.toml"
    device_path.write_text(_external_hardware_text(device_path.name))

    tgt = cuda_target.Target(device_path, architecture_path)
    assert tgt.architecture_id == "vendor.sm70"
    assert tgt.device_id == "vendor.v100_sxm2_32gb"
    assert re.fullmatch(r"[0-9a-f]{64}", tgt.architecture_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", tgt.device_digest)
    assert tgt.architecture.max_threads_per_cta == 1024
    assert tgt.device.sm_count == 80
    memory = tgt.get_facts(target.MemoryHierarchyFacts)
    throughput = tgt.get_facts(target.ThroughputFacts)
    assert memory.explicit("gmem").capacity_bytes == 32_000_000_000
    assert memory.explicit("tmem") is None
    assert throughput.peak_for(expr.DType.f16) == 125_000_000_000_000
    assert throughput.peak_for(expr.DType.bf16) is None
    architecture_document, device_document = hardware.documents(tgt)
    assert architecture_document.id == tgt.architecture_id
    assert device_document.id == tgt.device_id
    assert architecture_document.digest == tgt.architecture_digest
    assert device_document.digest == tgt.device_digest
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
    assert tuple(cuda_target.Target.hardware.documents()) == before
    with pytest.raises(hardware.UnknownDocumentError):
        cuda_target.Target.hardware.resolve("vendor.sm70")


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
        hardware.IncompatiblePairError,
        match=r"declares compatibility with \['vendor.other'\], not 'vendor.sm70'",
    ):
        cuda_target.Target(incompatible_path, architecture_path)

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
        hardware.SchemaValidationError,
        match=r"required fact 'memory.tensor.per_cta' is missing",
    ):
        cuda_target.Target(device_path, missing_path)


def test_a_schema_rejects_a_fact_it_does_not_model() -> None:
    """Reject unknown fact keys as spelling mistakes, not unused facts."""
    stray = _hardware_text("nvidia_sm90.toml") + (
        "\n[facts.compute.max_threads_per_cluster]\n"
        'value = 8\nunit = "count"\norigin = "vendor"\n'
        'source = "test"\nconditions = "test"\n'
    )
    with pytest.raises(hardware.SchemaValidationError, match="unknown facts for schema"):
        spec.build_cuda_architecture(hardware.parse_document(stray, origin_label="stray"))


def test_a_memory_owner_must_use_the_target_vocabulary() -> None:
    document = _hardware_text("nvidia_h200_sxm.toml").replace('value = "gpu"', 'value = "warp"')

    with pytest.raises(hardware.SchemaValidationError, match=r"memory owner 'warp'.*gpu.*cta.*thread"):
        spec.build_cuda_device(hardware.parse_document(document, origin_label="bad-owner"))


def test_resolution_failures_are_each_distinguishable() -> None:
    """Unknown IDs, schemas, duplicate IDs, and incompatible pairs differ."""
    with pytest.raises(hardware.UnknownDocumentError, match="no hardware document 'nvidia.sm70'"):
        cuda_target.Target("nvidia.sm70")


    with pytest.raises(hardware.UnknownDocumentError) as wrong_kind:
        cuda_target.Target(_SM90)
    assert str(wrong_kind.value) == (
        "cuda.Target.device got hardware document 'nvidia.sm90', which builds "
        "Architecture; expected Device"
    )

    document = cuda_target.Target.hardware.documents()[_SM90]
    with pytest.raises(hardware.DuplicateRegistrationError, match="already registered"):
        cuda_target.Target.hardware.adopt(document)

    unsupported = target.HardwareSpec("tilefoundry_nvidia.cuda.target.hardware", {})
    with pytest.raises(hardware.UnknownSchemaError, match="unsupported schema"):
        unsupported.resolve(_SM90)

    with pytest.raises(hardware.IncompatiblePairError, match="declares compatibility with"):
        cuda_target.Target(_H200, architecture=_SM100)


@pytest.mark.parametrize(
    ("construct", "message"),
    bare_projection_cases(
        lambda architecture: cuda_target.Target(_H200, architecture=architecture),
        cuda_target.Target,
        "cuda",
        (
            "cuda.Target.architecture must be an installed ID string, a document path, "
            "or an Architecture; got _BareArchitecture",
            "cuda.Target.device must be an installed ID string, a document path, or a "
            "Device; got _BareDevice",
        ),
    ),
)
def test_document_backends_reject_bare_projection_value_markers(construct, message: str) -> None:
    check_bare_projection(construct, message)


def test_no_installed_number_is_repeated_as_a_python_default() -> None:
    check_hardware_defaults(cuda_target.Target.hardware, (spec.SM90_ID, spec.H200_SXM_ID))

    assert cuda_target.Target("nvidia.h200_sxm").device.peak_for(expr.DType.f32) == 67_000_000_000_000
    assert cuda_target.Target.hardware.documents()[_H200].fact("throughput.f32").origin == "vendor"


def test_architecture_bandwidth_evidence_distinguishes_known_from_unavailable() -> None:
    """A measured architecture rate and an unknown one remain distinguishable."""
    hopper = cuda_target.Target.hardware.documents()[_SM90].fact("memory.shared.bandwidth")
    unavailable = cuda_target.Target.hardware.documents()[_SM100].fact("memory.shared.bandwidth")

    assert (hopper.value, hopper.unit, hopper.origin) == (
        128,
        "byte/clock/sm",
        "reference",
    )
    assert hopper.source and "microbenchmark" in hopper.conditions
    assert not unavailable.available
    assert unavailable.value is None
    assert unavailable.status == "unavailable"
    assert unavailable.conditions

    owners = {
        _SM89: {
            "memory.shared.owner": "cta",
            "memory.register.owner": "thread",
            "memory.tensor.owner": "cta",
        },
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
        _RTX_6000_ADA: {"memory.hbm.owner": "gpu"},
        _H200: {"memory.hbm.owner": "gpu"},
        _B200: {"memory.hbm.owner": "gpu"},
    }
    check_document_evidence(cuda_target.Target.hardware, owners)


def test_cuda_sm_clocks_are_the_typed_floor_of_existing_device_facts() -> None:
    """The clock fact extracts an existing derivation instead of repeating it."""
    for device_id in (_RTX_6000_ADA, _H200, _B200):
        tgt = cuda_target.Target(device_id)
        document = cuda_target.Target.hardware.documents()[device_id]
        peak = document.fact("throughput.f32").value
        divisor = tgt.device.sm_count * 128 * 2
        clock = tgt.device.sm_clock_hz

        assert clock == peak // divisor
        assert clock * divisor <= peak < (clock + 1) * divisor
        stated = document.fact("compute.sm_clock_hz")
        assert (stated.value, stated.unit, stated.origin) == (clock, "Hz", "derived")
        assert stated.source
        assert f"{peak / 1e12:g}e12 FLOP/s" in stated.conditions
        assert f"{tgt.device.sm_count} SM * 128 FP32 results/clock/SM" in stated.conditions
        assert "* 2 FLOP/result" in stated.conditions

    integer = cuda_target.Target.hardware.documents()[_H200].fact("service.integer")
    assert "facts.compute.sm_clock_hz" in integer.conditions
    assert "67e12 /" not in integer.conditions


def test_instruction_capabilities_and_execution_resources_are_separate_axes() -> None:
    """Enumerate instruction declarations and reconcile only their capability side."""
    check_architecture_capabilities(cuda_target.Target, "tilefoundry_nvidia.")
    assert all("cp.async" in tgt.architecture.capabilities for tgt in cuda_target.Target.from_installed())
