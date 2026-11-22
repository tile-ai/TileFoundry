"""Typed ANE hardware schema builders."""

from __future__ import annotations

from tilefoundry.target import hardware

from .architecture import Architecture
from .device import Device

ARCHITECTURE_SCHEMA = "tilefoundry.ane.architecture/v1"
DEVICE_SCHEMA = "tilefoundry.ane.device/v1"

APPLE_M2_PRO_ANE_ID = "apple.m2_pro.ane"


def build_architecture(document: hardware.Document) -> Architecture:
    """Build the immutable ANE architecture value from its document."""
    reader = hardware.SchemaReader(document)
    architecture = Architecture(
        name=reader.text("identity.name"),
        capabilities=reader.names("capabilities"),
        max_channels=reader.integer("tensor.max_channels", unit="count"),
        max_height=reader.integer("tensor.max_height", unit="count"),
        max_width=reader.integer("tensor.max_width", unit="count"),
        max_kernel=reader.integer("conv.max_kernel", unit="count"),
    )
    reader.close()
    return architecture


def build_device(document: hardware.Document) -> Device:
    """Build the immutable ANE device value from its document."""
    reader = hardware.SchemaReader(document)
    device = Device(
        name=reader.text("identity.name"),
        core_count=reader.integer("compute.cores", unit="count"),
        unified_memory_capacity_bytes=reader.integer("memory.unified.capacity", unit="byte"),
        f16_throughput=reader.integer("throughput.f16", unit="flop/s"),
    )
    reader.integer("memory.unified.bandwidth", unit="byte/s")
    reader.declared_unavailable("memory.l2.capacity")
    reader.close()
    return device


__all__ = [
    "APPLE_M2_PRO_ANE_ID",
    "ARCHITECTURE_SCHEMA",
    "DEVICE_SCHEMA",
    "build_architecture",
    "build_device",
]
