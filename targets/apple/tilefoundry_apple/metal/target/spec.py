"""Typed Metal hardware schema builders."""

from __future__ import annotations

from tilefoundry.target import hardware

from .architecture import Architecture
from .device import Device

ARCHITECTURE_SCHEMA = "tilefoundry.metal.architecture/v1"
DEVICE_SCHEMA = "tilefoundry.metal.device/v1"

APPLE_M2_PRO_GPU_ID = "apple.m2_pro.gpu"


def build_architecture(document: hardware.Document) -> Architecture:
    """Build the immutable GPU architecture value from its document."""
    reader = hardware.SchemaReader(document)
    architecture = Architecture(
        name=reader.text("identity.name"),
        capabilities=reader.names("capabilities"),
        simd_width=reader.integer("simd.width", unit="count"),
    )
    reader.close()
    return architecture


def build_device(document: hardware.Document) -> Device:
    """Build the immutable GPU device value from its document."""
    reader = hardware.SchemaReader(document)
    device = Device(
        name=reader.text("identity.name"),
        max_threads_per_threadgroup=reader.integer("compute.max_threads_per_threadgroup", unit="count"),
        threadgroup_memory_bytes=reader.integer("memory.threadgroup", unit="byte"),
        unified_memory_capacity_bytes=reader.integer("memory.unified.capacity", unit="byte"),
    )
    reader.integer("memory.unified.bandwidth", unit="byte/s")
    reader.declared_unavailable("compute.resident_threadgroups")
    reader.declared_unavailable("memory.register_per_thread")
    reader.close()
    return device


__all__ = [
    "APPLE_M2_PRO_GPU_ID",
    "ARCHITECTURE_SCHEMA",
    "DEVICE_SCHEMA",
    "build_architecture",
    "build_device",
]
