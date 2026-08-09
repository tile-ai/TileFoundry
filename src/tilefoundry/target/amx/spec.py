"""Typed AMX hardware schema builders."""

from __future__ import annotations

from tilefoundry.ir.types import DType
from tilefoundry.target.amx.architecture import AppleAmx
from tilefoundry.target.amx.device import AppleM2Pro
from tilefoundry.target.facts import TARGET_MEMORY_OWNER
from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    SchemaValidationError,
)
from tilefoundry.target.hardware.schema import SchemaReader

ARCHITECTURE_SCHEMA = "tilefoundry.amx.architecture/v2"
DEVICE_SCHEMA = "tilefoundry.amx.device/v2"

APPLE_AMX_ID = "apple.amx"
APPLE_M2_PRO_ID = "apple.m2_pro"


def _memory_owner(reader: SchemaReader, path: str) -> str:
    """Read an owner in the AMX target's topology vocabulary."""
    owner = reader.text(path)
    allowed = (TARGET_MEMORY_OWNER, "core", "amx")
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


def build_apple_amx(document: HardwareDocument) -> AppleAmx:
    """Build the immutable AMX architecture value from its document."""
    reader = SchemaReader(document)
    architecture = AppleAmx(
        name=reader.text("identity.name"),
        supported_compute_dtypes=_dtypes(
            reader.names("instruction.compute_dtypes"), document
        ),
        instruction_capabilities=reader.names("instruction.capabilities"),
        amx_units_per_core=reader.integer("compute.units_per_core", unit="count"),
        staging_bytes=reader.integer("register.x_file", unit="byte"),
        accumulator_bytes=reader.integer("register.z_file", unit="byte"),
        rmem_owner=_memory_owner(reader, "register.owner"),
    )
    y_file = reader.integer("register.y_file", unit="byte")
    reader.text("geometry.f32_outer_product")
    reader.text("geometry.f32_accumulator_tile")
    reader.declared_unavailable("throughput.f32_instruction_peak")
    reader.close()

    if y_file != architecture.staging_bytes:
        raise SchemaValidationError(
            f"{document.id}: the X and Y staging files must be the same size, "
            f"got {architecture.staging_bytes} B and {y_file} B"
        )
    return architecture


def build_apple_m2_pro(document: HardwareDocument) -> AppleM2Pro:
    """Build the immutable Apple M2 Pro value from its document."""
    reader = SchemaReader(document)
    unit_flops = (
        ("amx", ((DType.f32, reader.integer("throughput.amx.f32", unit="flop/s")),)),
        ("neon", ((DType.f32, reader.integer("throughput.neon.f32", unit="flop/s")),)),
    )
    device = AppleM2Pro(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.amx_unit_count", unit="count"),
        performance_core_count=reader.integer(
            "compute.performance_core_count", unit="count"
        ),
        efficiency_core_count=reader.integer(
            "compute.efficiency_core_count", unit="count"
        ),
        l1d_bytes_per_performance_core=reader.integer(
            "memory.l1d.per_performance_core", unit="byte"
        ),
        l1d_bytes_per_efficiency_core=reader.integer(
            "memory.l1d.per_efficiency_core", unit="byte"
        ),
        l2_bytes_per_performance_cluster=reader.integer(
            "memory.l2.per_performance_cluster", unit="byte"
        ),
        l2_bytes_per_efficiency_cluster=reader.integer(
            "memory.l2.per_efficiency_cluster", unit="byte"
        ),
        cache_line_bytes=reader.integer("memory.cache_line", unit="byte"),
        unified_memory_capacity_bytes=reader.integer(
            "memory.unified.capacity", unit="byte"
        ),
        unified_memory_owner=_memory_owner(reader, "memory.unified.owner"),
        unified_memory_bandwidth_bytes_per_second=reader.integer(
            "memory.unified.bandwidth", unit="byte/s"
        ),
        _unit_flops=unit_flops,
    )
    reader.text("geometry.neon_f32_outer_product")
    reader.close()

    if device.sm_count > device.performance_core_count:
        raise SchemaValidationError(
            f"{document.id}: {device.sm_count} AMX units cannot exceed the "
            f"{device.performance_core_count} performance cores that share them"
        )
    return device


__all__ = [
    "APPLE_AMX_ID",
    "APPLE_M2_PRO_ID",
    "ARCHITECTURE_SCHEMA",
    "DEVICE_SCHEMA",
    "build_apple_amx",
    "build_apple_m2_pro",
]
