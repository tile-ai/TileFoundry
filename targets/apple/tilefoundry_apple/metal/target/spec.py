"""Exact hardware document schemas for Metal."""
from tilefoundry.target.hardware.schema import SchemaReader
from tilefoundry.target.hardware.envelope import SchemaValidationError
from .architecture import MetalArchitecture
from .device import MetalDevice

ARCHITECTURE_SCHEMA = "tilefoundry.metal.architecture/v1"
DEVICE_SCHEMA = "tilefoundry.metal.device/v1"
APPLE_GPU8_ID = "apple.gpu8"
APPLE_M2_PRO_GPU_ID = "apple.m2_pro.gpu"

def build_architecture(document):
    r = SchemaReader(document)
    result = MetalArchitecture(r.text("identity.name"), r.names("capabilities"),
                       r.integer("simd.width", unit="count"),
                       r.integer("language.version", unit="version"))
    r.close()
    if result.simd_width != 32:
        raise SchemaValidationError("MetalArchitecture layout profile requires SIMD32")
    return result

def build_device(document):
    r = SchemaReader(document)
    result = MetalDevice(
        r.text("identity.name"), r.optional_integer("compute.gpu_cores", unit="count"),
        r.integer("compute.max_threads_per_threadgroup", unit="count"),
        r.optional_integer("memory.threadgroup", unit="byte"),
        r.optional_integer("memory.unified", unit="byte"),
        r.optional_integer("memory.bandwidth", unit="byte/s"),
        r.optional_integer("compute.resident_threadgroups", unit="count"),
        r.optional_integer("memory.register_per_thread", unit="byte"),
    )
    r.close()
    return result
