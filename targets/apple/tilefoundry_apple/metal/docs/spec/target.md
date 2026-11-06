# Metal Target contract

```python
from __future__ import annotations

class MetalTarget(Target):
    def __init__(self, device: Device | str | Path,
                 architecture: Architecture | str | Path | None = None) -> None: ...

    @classmethod
    def available(cls) -> tuple[MetalTarget, ...]: ...

    def get_facts(self, facts_type: type, query: object | None = None): ...
    def to_python(self) -> PythonExpr: ...
```

Construction requires an explicit device. IDs and paths are resolved through
HardwareSpec and SchemaReader; immutable MetalArchitecture and MetalDevice
values are also accepted. An omitted architecture is resolved from device
compatibility metadata. Incompatible document IDs, malformed schemas, missing
required facts and unconsumed fields are errors. Construction/import neither
initializes Metal nor requires an SDK.

The installed inventory contains architecture `apple.gpu8` and physical device
`apple.m2_pro.gpu` only. MLX source profiles are not devices. The architecture
stores name, capabilities, SIMD width (32) and language version; the device
stores resource facts. Document IDs/digests are retained for identity and
reconstruction. `identity` is the device ID when available; `arch` returns the
architecture name. See the [generic contract](../../../../../../docs/spec/target.md).

## Topology and Facts

TopologyFacts describes threadgroup → thread, with threadgroup as parallel level.
The thread ceiling is 1024; actual pipelines may impose a smaller bound.
Resident threadgroups remain None, independently of the 19 GPU cores.
MemoryExecutionFacts projects those topology names. TopologyLevelFacts queries
select the named level; unsupported queries follow the base Target contract.

| MemoryHierarchyFacts level | Owner | Capacity |
| --- | --- | --- |
| gmem | target | 34359738368 bytes shared system DRAM |
| smem | threadgroup | 32768 bytes |
| rmem | thread | None |

There is no second host pool. Shared DRAM is not an allocation budget.
ThroughputFacts contains no compute rates and records vendor package bandwidth
200000000000 byte/s, not measured GPU throughput. PerformanceServiceFacts
provides no invented rates and rejects unknown topology names. Register capacity,
residency and compute rates stay unknown, rather than zero or guessed values.

Capabilities gate the normal instruction registry. The real `apple.m2_pro.gpu` inventory does
not advertise 16×32×16 operations. Their static/logic contracts do not prove
hardware availability. MSL language requirements and source conditions live in
the hardware documents. No codegen, self-authored runtime, submission service or
calibration is implied by this description-only target.
