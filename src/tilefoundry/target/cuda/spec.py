"""Typed CUDA hardware schemas and the documents this package installs."""

from __future__ import annotations

from tilefoundry.ir.types import DType
from tilefoundry.target.cuda.architecture import SM90
from tilefoundry.target.cuda.device import H200SXM
from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    SchemaValidationError,
)
from tilefoundry.target.hardware.registry import HARDWARE_SPECS, HardwareSpecRegistry
from tilefoundry.target.hardware.schema import SchemaReader

ARCHITECTURE_SCHEMA = "tilefoundry.cuda.architecture/v1"
DEVICE_SCHEMA = "tilefoundry.cuda.device/v1"

SM90_ID = "nvidia.sm90"
H200_SXM_ID = "nvidia.h200_sxm"

_PACKAGE = "tilefoundry.target.hardware"


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


def build_sm90(document: HardwareDocument) -> SM90:
    """Build the immutable SM90 value from its installed document."""
    reader = SchemaReader(document)
    max_threads_per_cta = reader.integer(
        "compute.max_threads_per_cta", unit="thread"
    )
    max_threads_per_warp = reader.integer(
        "compute.max_threads_per_warp", unit="thread"
    )
    max_warps_per_cta = reader.integer("compute.max_warps_per_cta", unit="count")
    architecture = SM90(
        name=reader.text("identity.name"),
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


def build_h200_sxm(document: HardwareDocument) -> H200SXM:
    """Build the immutable H200 SXM value from its installed document."""
    reader = SchemaReader(document)
    dense_flops = tuple(
        (getattr(DType, name), reader.integer(f"throughput.{name}", unit="flop/s"))
        for name in ("f32", "f16", "bf16", "fp8e4m3")
    )
    device = H200SXM(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.sm_count", unit="count"),
        hbm_capacity_bytes=reader.integer("memory.hbm.capacity", unit="byte"),
        hbm_bandwidth_bytes_per_second=reader.integer(
            "memory.hbm.bandwidth", unit="byte/s"
        ),
        l2_capacity_bytes=reader.optional_integer("memory.l2.capacity", unit="byte"),
        _dense_flops=dense_flops,
    )
    reader.declared_unavailable("memory.l2.bandwidth")
    reader.close()
    return device


def install(registry: HardwareSpecRegistry | None = None) -> None:
    """Register the CUDA schemas and documents into *registry*."""
    into = HARDWARE_SPECS if registry is None else registry
    into.register_schema(ARCHITECTURE_SCHEMA, build_sm90)
    into.register_schema(DEVICE_SCHEMA, build_h200_sxm)
    into.install(SM90_ID, _PACKAGE, "nvidia_sm90.toml")
    into.install(H200_SXM_ID, _PACKAGE, "nvidia_h200_sxm.toml")


def installed_architecture() -> SM90:
    """The installed SM90 architecture value."""
    return HARDWARE_SPECS.resolve(SM90_ID).value


def installed_device() -> H200SXM:
    """The installed H200 SXM device value."""
    return HARDWARE_SPECS.resolve(H200_SXM_ID).value


install()


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "DEVICE_SCHEMA",
    "H200_SXM_ID",
    "SM90_ID",
    "build_h200_sxm",
    "build_sm90",
    "install",
    "installed_architecture",
    "installed_device",
]
