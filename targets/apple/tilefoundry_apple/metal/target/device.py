"""Device resources read from the installed hardware document."""
from dataclasses import dataclass

from tilefoundry.target.base import Device


@dataclass(frozen=True)
class MetalDevice(Device):
    name: str
    gpu_core_count: int | None
    max_threads_per_threadgroup: int
    threadgroup_memory_bytes: int | None
    unified_memory_bytes: int | None
    memory_bandwidth_bytes_per_second: int | None
    resident_threadgroups: int | None
    register_bytes_per_thread: int | None
