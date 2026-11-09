"""Typed AMX hardware schema builders."""

from __future__ import annotations

from tilefoundry.ir import expr
from tilefoundry.target import hardware

from .architecture import Architecture
from .device import AppleM2Pro

ARCHITECTURE_SCHEMA = "tilefoundry.amx.architecture/v2"
DEVICE_SCHEMA = "tilefoundry.amx.device/v2"

APPLE_AMX_ID = "apple.amx"
APPLE_M2_PRO_ID = "apple.m2_pro"


def memory_owner(reader: hardware.SchemaReader, path: str) -> str:
    """Read an owner in the AMX target's topology vocabulary."""
    owner = reader.text(path)
    allowed = (expr.TARGET_MEMORY_OWNER, "thread")
    if owner not in allowed:
        raise hardware.SchemaValidationError(
            f"{reader.document.id}: memory owner {owner!r} at {path!r} must be "
            f"one of {list(allowed)}"
        )
    return owner


def compute_dtypes(names: tuple[str, ...], document: hardware.Document) -> tuple[expr.DType, ...]:
    """Resolve recorded dtype names against the IR dtype table."""
    resolved = []
    for name in names:
        dtype = getattr(expr.DType, name, None)
        if dtype is None:
            raise hardware.SchemaValidationError(f"{document.id}: unknown compute dtype {name!r}")
        resolved.append(dtype)
    return tuple(resolved)


def build_apple_amx(document: hardware.Document) -> Architecture:
    """Build the immutable AMX architecture value from its document."""
    reader = hardware.SchemaReader(document)
    architecture = Architecture(
        name=reader.text("identity.name"),
        supported_compute_dtypes=compute_dtypes(reader.names("instruction.compute_dtypes"), document),
        capabilities=reader.names("capabilities"),
        staging_bytes=reader.integer("register.x_file", unit="byte"),
        accumulator_bytes=reader.integer("register.z_file", unit="byte"),
        rmem_owner=memory_owner(reader, "register.owner"),
    )
    y_file = reader.integer("register.y_file", unit="byte")
    reader.text("geometry.f32_outer_product")
    reader.text("geometry.f32_accumulator_tile")
    reader.declared_unavailable("throughput.f32_instruction_peak")
    reader.close()

    if y_file != architecture.staging_bytes:
        raise hardware.SchemaValidationError(
            f"{document.id}: the X and Y staging files must be the same size, "
            f"got {architecture.staging_bytes} B and {y_file} B"
        )
    return architecture


def build_apple_m2_pro(document: hardware.Document) -> AppleM2Pro:
    """Build the immutable Apple M2 Pro value from its document."""
    reader = hardware.SchemaReader(document)
    unit_flops = (
        ("amx", ((expr.DType.f32, reader.integer("throughput.amx.f32", unit="flop/s")),)),
        ("neon", ((expr.DType.f32, reader.integer("throughput.neon.f32", unit="flop/s")),)),
    )
    device = AppleM2Pro(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.amx_unit_count", unit="count"),
        performance_core_count=reader.integer("compute.performance_core_count", unit="count"),
        efficiency_core_count=reader.integer("compute.efficiency_core_count", unit="count"),
        l1d_bytes_per_performance_core=reader.integer(
            "memory.l1d.per_performance_core", unit="byte"
        ),
        l1d_bytes_per_efficiency_core=reader.integer("memory.l1d.per_efficiency_core", unit="byte"),
        l2_bytes_per_performance_cluster=reader.integer(
            "memory.l2.per_performance_cluster", unit="byte"
        ),
        l2_bytes_per_efficiency_cluster=reader.integer(
            "memory.l2.per_efficiency_cluster", unit="byte"
        ),
        cache_line_bytes=reader.integer("memory.cache_line", unit="byte"),
        unified_memory_capacity_bytes=reader.integer("memory.unified.capacity", unit="byte"),
        unified_memory_owner=memory_owner(reader, "memory.unified.owner"),
        unified_memory_bandwidth_bytes_per_second=reader.integer(
            "memory.unified.bandwidth", unit="byte/s"
        ),
        unit_flops=unit_flops,
        performance_amx_f16=reader.optional_integer("performance.amx.f16", unit="flop/s"),
        performance_host_lane_events=reader.optional_integer(
            "performance.host.lane_events", unit="op/s"
        ),
    )
    reader.text("geometry.neon_f32_outer_product")
    reader.close()

    if device.sm_count > device.performance_core_count:
        raise hardware.SchemaValidationError(
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
