"""Typed CUDA hardware schema builders."""

from __future__ import annotations

from tilefoundry.ir.types import DType
from tilefoundry.target.cuda.architecture import CudaArchitecture
from tilefoundry.target.cuda.device import CudaDevice
from tilefoundry.target.facts import TARGET_MEMORY_OWNER
from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    SchemaValidationError,
)
from tilefoundry.target.hardware.schema import SchemaReader

ARCHITECTURE_SCHEMA = "tilefoundry.cuda.architecture/v3"
DEVICE_SCHEMA = "tilefoundry.cuda.device/v3"

SM90_ID = "nvidia.sm90"
SM100_ID = "nvidia.sm100"
H200_SXM_ID = "nvidia.h200_sxm"
B200_SXM_ID = "nvidia.b200_sxm"

_THROUGHPUT_DTYPE_NAMES = ("f32", "f16", "bf16", "fp8e4m3", "f4e2m1")


def _memory_owner(reader: SchemaReader, path: str) -> str:
    """Read an owner in the CUDA target's topology vocabulary."""
    owner = reader.text(path)
    allowed = (TARGET_MEMORY_OWNER, "cta", "thread")
    if owner not in allowed:
        raise SchemaValidationError(
            f"{reader.document.id}: memory owner {owner!r} at {path!r} must be "
            f"one of {list(allowed)}"
        )
    return owner


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


def build_cuda_architecture(document: HardwareDocument) -> CudaArchitecture:
    """Build the immutable CUDA architecture value from its installed document."""
    reader = SchemaReader(document)
    max_threads_per_cta = reader.integer(
        "compute.max_threads_per_cta", unit="thread"
    )
    max_threads_per_warp = reader.integer(
        "compute.max_threads_per_warp", unit="thread"
    )
    max_warps_per_cta = reader.integer("compute.max_warps_per_cta", unit="count")
    architecture = CudaArchitecture(
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
        smem_owner=_memory_owner(reader, "memory.shared.owner"),
        unified_l1_shared_per_sm_bytes=reader.integer(
            "memory.unified_l1_shared.per_sm", unit="byte"
        ),
        registers_per_sm_32bit=reader.integer("memory.register.per_sm", unit="register"),
        rmem_owner=_memory_owner(reader, "memory.register.owner"),
        tensor_memory_per_cta_bytes=reader.optional_integer(
            "memory.tensor.per_cta", unit="byte"
        ),
        tmem_owner=_memory_owner(reader, "memory.tensor.owner"),
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


_SERVICE_KINDS = ("integer", "predicate", "select", "special")


def build_cuda_device(document: HardwareDocument) -> CudaDevice:
    """Build the immutable CUDA device value from its installed document.

    ``_SERVICE_KINDS`` names every service a device may state a throughput for.
    One the document leaves out is absent rather than zero, so a consumer that
    needs it refuses instead of pricing that work at nothing.
    """
    reader = SchemaReader(document)
    dense_flops: list[tuple[DType, int]] = []
    for dtype_name in _THROUGHPUT_DTYPE_NAMES:
        peak = reader.optional_integer(f"throughput.{dtype_name}", unit="flop/s")
        if peak is not None:
            dense_flops.append((getattr(DType, dtype_name), peak))
    services: list[tuple[str, int]] = []
    for kind in _SERVICE_KINDS:
        if f"service.{kind}" not in document.facts:
            continue
        stated = reader.optional_integer(f"service.{kind}", unit="op/s")
        if stated is not None:
            services.append((kind, stated))
    device = CudaDevice(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.sm_count", unit="count"),
        hbm_capacity_bytes=reader.integer("memory.hbm.capacity", unit="byte"),
        gmem_owner=_memory_owner(reader, "memory.hbm.owner"),
        hbm_bandwidth_bytes_per_second=reader.integer(
            "memory.hbm.bandwidth", unit="byte/s"
        ),
        l2_capacity_bytes=reader.optional_integer("memory.l2.capacity", unit="byte"),
        _dense_flops=tuple(dense_flops),
        _service_ops=tuple(services),
    )
    reader.declared_unavailable("memory.l2.bandwidth")
    reader.close()
    return device


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "B200_SXM_ID",
    "DEVICE_SCHEMA",
    "H200_SXM_ID",
    "SM100_ID",
    "SM90_ID",
    "build_cuda_architecture",
    "build_cuda_device",
]
