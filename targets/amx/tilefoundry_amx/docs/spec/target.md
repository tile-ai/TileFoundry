# TileFoundry AMX Spec — Target

## 7. `AppleAmx`

```python
class AppleAmx:
    """Describe AMX compilation identity and structural capabilities."""

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    capabilities: tuple[str, ...]
    amx_units_per_core: int
    staging_bytes: int
    accumulator_bytes: int
    rmem_owner: str

    def supports_compute_dtype(self, dtype: DType) -> bool: ...

    def topology_limit(self, name: str) -> int: ...
```

- constraints:
  - `name` MUST be the architecture identity used by AMX compilation.
  - AppleAmx MUST own the supported compute DTypes and the per-core AMX unit
    count. The modelled atom catalogue MAY be narrower than the supported
    compute DTypes.
  - AppleAmx MUST own the X/Y staging and Z accumulator register files. They are
    ISA geometry, so every part carrying this coprocessor shares them and a
    device MUST NOT restate them. `staging_bytes` MUST be the size of one
    staging file, the X and Y files being equal.
  - `rmem_owner` MUST be `amx`: each AMX unit owns its register-file values.
  - Product- and frequency-dependent throughput values MUST NOT be stored on
    AppleAmx.
  - AMX has no CTA thread level, so AppleAmx MUST carry no CTA thread limit.
  - No field MAY carry a default: every value comes from the installed
    document ([target §1](../../../../../docs/spec/target.md#1-target)0).

## 8. `AppleM2Pro`

```python
class AppleM2Pro:
    """Describe the apple_m2_pro package's fixed hard resource limits."""

    name: str
    sm_count: int
    performance_core_count: int
    efficiency_core_count: int
    l1d_bytes_per_performance_core: int
    l1d_bytes_per_efficiency_core: int
    l2_bytes_per_performance_cluster: int
    l2_bytes_per_efficiency_cluster: int
    cache_line_bytes: int
    unified_memory_capacity_bytes: int
    unified_memory_owner: str
    unified_memory_bandwidth_bytes_per_second: int
    _unit_flops: tuple[tuple[str, tuple[tuple[DType, int], ...]], ...]

    performance_amx_f16: int | None
    performance_host_lane_events: int | None

    def throughput_for(self, unit: str, dtype: DType) -> int: ...
```

- constraints:
  - AppleM2Pro MUST describe one package and MUST NOT carry a machine count.
  - `sm_count` MUST be the number of independent AMX units, which is the
    parallel-unit count a makespan divides work over. It MUST NOT be read as a
    core count: the performance cores outnumber the units and share them, so it
    MUST NOT exceed `performance_core_count`.
  - Cache and core facts MUST distinguish the performance core from the
    efficiency core, and every value MUST come from the installed document
    ([target §1](../../../../../docs/spec/target.md#1-target)0). No field MAY carry a default.
  - A core-level tile's resident footprint MUST be bounded by
    `l1d_bytes_per_performance_core`. The AMX register files bound one atom
    instance instead, which the storage filter enforces rather than a per-tile
    capacity, so the two MUST NOT be conflated.
  - `unified_memory_owner` MUST be `target`; both the `host` and `gmem`
    explicit levels project that one target-wide ownership fact.
  - `throughput_for` MUST be keyed by execution unit as well as DType, because
    the AMX coprocessor and the core's NEON pipes have separate measured rates.
  - A tile's traffic MUST be charged against
    `unified_memory_bandwidth_bytes_per_second`, which unified memory backs.
  - `throughput_for` MUST return a measured per-unit throughput recorded in the
    installed document, and MUST raise an actionable error for a unit or compute
    DType with no measured entry rather than return an estimate.
  - `performance_amx_f16` and `performance_host_lane_events` MUST read their
    available or unavailable entries from the installed document. Estimates MUST
    state their sources and assumptions separately from measured throughput and
    MUST NOT populate `throughput_for`'s measured-rate map.
  - AMX `PerformanceServiceFacts` MUST use the measured NEON FP32 rate for host
    arithmetic. FP16 service MUST divide the documented per-AMX-unit estimate
    across core streams using the AMX unit count and performance-core count.
    Non-FLOP work MUST use the documented per-core lane-event estimate.
    Global-memory bandwidth MUST be shared across the same core streams. AMX
    reads the level of a `(level, active)` query and ignores `active`.
    Unrecorded DTypes and services MUST remain absent.


## 9. `AmxTarget`

```python
class AmxTarget(Target):
    """Compose one AMX target from one architecture and one device."""

    name: ClassVar[str] = "amx"
    architecture: Architecture
    device: Device
    architecture_id: str | None
    device_id: str | None
    architecture_digest: str | None
    device_digest: str | None
    arch: str

    def __init__(
        self,
        device: Device | str | Path | None = None,
        architecture: Architecture | str | Path | None = None,
    ) -> None: ...

    def topology_limit(self, name: str) -> int: ...

    def validate_program_topology(self, topology: Topology) -> None: ...
    def get_analyzer(self, selector: str) -> Analyzer: ...
    def get_facts(self, facts_type: type[FactsT], query=None) -> FactsT: ...
```

- constraints:
  - `device` and `architecture` MUST accept an installed document ID, a
    document path, or a concrete value, on the same terms as
    [target §4](../../../../cuda/tilefoundry_cuda/docs/spec/target.md#4-cudatarget). An omitted architecture MUST be read from the device
    document's sole compatibility declaration.
  - `AmxTarget()` MUST select the installed `apple.amx` and `apple.m2_pro`
    documents, and `arch` MUST equal
    `architecture.name`.
  - `AmxTarget.available()` MUST contain one value per device document whose
    sole compatible architecture document is available. Its `identity` MUST be
    that device document's ID.
  - `get_facts(TopologyFacts).topologies` MUST name `("core", "amx")`: the
    performance core one tile stream runs on, and the AMX unit inside that core
    which issues one atom. Neither level sets `from_target`.
  - `topology_limit("core")` MUST equal `device.performance_core_count` and
    `topology_limit("amx")` MUST equal `architecture.amx_units_per_core`.
  - Declared topology extents MUST be positive static integers within their
    level's limit. AMX has no launch shape, so a deferred or symbolic extent
    MUST NOT be admitted at either level.
  - Unsupported topology levels MUST raise an actionable error naming the
    supported levels, from both the limit lookup and topology validation.
