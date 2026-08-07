"""Typed CUDA hardware schemas and the documents this package installs."""

from __future__ import annotations

from tilefoundry.ir.types import DType
from tilefoundry.target.cuda.architecture import SM90, SM100, CudaArchitecture
from tilefoundry.target.cuda.device import B200SXM, H200SXM, CudaDevice
from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    SchemaValidationError,
)
from tilefoundry.target.hardware.registry import HARDWARE_SPECS, HardwareSpecRegistry
from tilefoundry.target.hardware.schema import SchemaReader

ARCHITECTURE_SCHEMA = "tilefoundry.cuda.architecture/v1"
DEVICE_SCHEMA = "tilefoundry.cuda.device/v1"

SM90_ID = "nvidia.sm90"
SM100_ID = "nvidia.sm100"
H200_SXM_ID = "nvidia.h200_sxm"
B200_SXM_ID = "nvidia.b200_sxm"

_PACKAGE = "tilefoundry.target.hardware"

# One document format per kind: every CUDA architecture document states the same
# fact paths and so does every device document, which is what lets one schema
# validate them all. Which value class a document builds is the identity it
# declares, because that identity is the document's own content rather than a
# dispatch on a target name.
_ARCHITECTURE_IDENTITIES: dict[str, type[CudaArchitecture]] = {
    "sm_90": SM90,
    "sm_100": SM100,
}
_DEVICE_IDENTITIES: dict[str, type[CudaDevice]] = {
    "h200_sxm": H200SXM,
    "b200_sxm": B200SXM,
}

# The compute DTypes a CUDA device document states a dense peak rate for. A
# product whose tensor cores have no mode for one of them records it
# unavailable, so the absence is a statement rather than a missing key.
_THROUGHPUT_DTYPE_NAMES = ("f32", "f16", "bf16", "fp8e4m3", "f4e2m1")


def _dtypes(names: tuple[str, ...], document: HardwareDocument) -> tuple[DType, ...]:
    """Resolve recorded dtype names against the IR dtype table."""
    resolved = []
    for name in names:
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise SchemaValidationError(
                f"{document.id}: unknown compute dtype {name!r}"
            )
        resolved.append(dtype)
    return tuple(resolved)


def _identity(
    identities: dict[str, type], name: str, document: HardwareDocument
) -> type:
    """The value class the identity a document declares selects."""
    try:
        return identities[name]
    except KeyError:
        raise SchemaValidationError(
            f"{document.id}: no value class for identity {name!r}; "
            f"this schema builds {sorted(identities)}"
        ) from None


def build_cuda_architecture(document: HardwareDocument) -> CudaArchitecture:
    """Build the immutable CUDA architecture value from its installed document."""
    reader = SchemaReader(document)
    name = reader.text("identity.name")
    architecture_type = _identity(_ARCHITECTURE_IDENTITIES, name, document)
    max_threads_per_cta = reader.integer(
        "compute.max_threads_per_cta", unit="thread"
    )
    max_threads_per_warp = reader.integer(
        "compute.max_threads_per_warp", unit="thread"
    )
    max_warps_per_cta = reader.integer("compute.max_warps_per_cta", unit="count")
    architecture = architecture_type(
        name=name,
        supported_compute_dtypes=_dtypes(
            reader.names("instruction.compute_dtypes"), document
        ),
        instruction_capabilities=reader.names("instruction.capabilities"),
        max_threads_per_cta=max_threads_per_cta,
        max_threads_per_warp=max_threads_per_warp,
        max_warps_per_cta=max_warps_per_cta,
        max_resident_ctas_per_sm=reader.integer(
            "compute.max_resident_ctas_per_sm", unit="count"
        ),
        shared_memory_per_sm_bytes=reader.integer(
            "memory.shared.per_sm", unit="byte"
        ),
        shared_memory_per_cta_bytes=reader.integer(
            "memory.shared.per_cta", unit="byte"
        ),
        unified_l1_shared_per_sm_bytes=reader.integer(
            "memory.unified_l1_shared.per_sm", unit="byte"
        ),
        registers_per_sm_32bit=reader.integer("memory.register.per_sm", unit="register"),
        tensor_memory_per_cta_bytes=reader.optional_integer(
            "memory.tensor.per_cta", unit="byte"
        ),
    )
    reader.declared_unavailable("memory.shared.bandwidth")
    reader.declared_unavailable("memory.register.bandwidth")
    reader.close()

    if max_threads_per_warp * max_warps_per_cta != max_threads_per_cta:
        raise SchemaValidationError(
            f"{document.id}: {max_warps_per_cta} warps of "
            f"{max_threads_per_warp} threads do not fill a "
            f"{max_threads_per_cta}-thread CTA"
        )
    if architecture.shared_memory_per_cta_bytes > architecture.shared_memory_per_sm_bytes:
        raise SchemaValidationError(
            f"{document.id}: shared memory per CTA "
            f"({architecture.shared_memory_per_cta_bytes} B) exceeds the per-SM "
            f"capacity ({architecture.shared_memory_per_sm_bytes} B)"
        )
    if (
        architecture.shared_memory_per_sm_bytes
        > architecture.unified_l1_shared_per_sm_bytes
    ):
        raise SchemaValidationError(
            f"{document.id}: the shared-memory carveout "
            f"({architecture.shared_memory_per_sm_bytes} B) exceeds the unified "
            f"L1 and shared block it is taken from "
            f"({architecture.unified_l1_shared_per_sm_bytes} B)"
        )
    return architecture


def build_cuda_device(document: HardwareDocument) -> CudaDevice:
    """Build the immutable CUDA device value from its installed document."""
    reader = SchemaReader(document)
    name = reader.text("identity.name")
    device_type = _identity(_DEVICE_IDENTITIES, name, document)
    dense_flops: list[tuple[DType, int]] = []
    for dtype_name in _THROUGHPUT_DTYPE_NAMES:
        peak = reader.optional_integer(f"throughput.{dtype_name}", unit="flop/s")
        if peak is not None:
            dense_flops.append((getattr(DType, dtype_name), peak))
    device = device_type(
        name=name,
        sm_count=reader.integer("compute.sm_count", unit="count"),
        hbm_capacity_bytes=reader.integer("memory.hbm.capacity", unit="byte"),
        hbm_bandwidth_bytes_per_second=reader.integer(
            "memory.hbm.bandwidth", unit="byte/s"
        ),
        l2_capacity_bytes=reader.optional_integer("memory.l2.capacity", unit="byte"),
        _dense_flops=tuple(dense_flops),
    )
    reader.declared_unavailable("memory.l2.bandwidth")
    reader.close()
    return device


def install(registry: HardwareSpecRegistry | None = None) -> None:
    """Register the CUDA schemas and documents into *registry*."""
    into = HARDWARE_SPECS if registry is None else registry
    into.register_schema(ARCHITECTURE_SCHEMA, build_cuda_architecture)
    into.register_schema(DEVICE_SCHEMA, build_cuda_device)
    into.install(SM90_ID, _PACKAGE, "nvidia_sm90.toml")
    into.install(SM100_ID, _PACKAGE, "nvidia_sm100.toml")
    into.install(H200_SXM_ID, _PACKAGE, "nvidia_h200_sxm.toml")
    into.install(B200_SXM_ID, _PACKAGE, "nvidia_b200_sxm.toml")


def installed_sm90() -> SM90:
    """The installed SM90 architecture value."""
    return HARDWARE_SPECS.resolve(SM90_ID).value


def installed_h200_sxm() -> H200SXM:
    """The installed H200 SXM device value."""
    return HARDWARE_SPECS.resolve(H200_SXM_ID).value


install()


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "B200_SXM_ID",
    "DEVICE_SCHEMA",
    "H200_SXM_ID",
    "SM90_ID",
    "SM100_ID",
    "build_cuda_architecture",
    "build_cuda_device",
    "install",
    "installed_h200_sxm",
    "installed_sm90",
]
