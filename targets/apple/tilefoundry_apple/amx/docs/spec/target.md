# TileFoundry AMX Spec — Target

## 7. `AppleAmx`

```python
class AppleAmx:
    """Describe AMX compilation identity and structural capabilities."""

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    capabilities: tuple[str, ...]
    staging_bytes: int
    accumulator_bytes: int
    rmem_owner: str

    def supports_compute_dtype(self, dtype: DType) -> bool: ...

    def topology_limit(self, name: str) -> int: ...
```

- constraints:
  - `name` MUST be the architecture identity used by AMX compilation.
  - AppleAmx MUST own the supported compute DTypes and instruction capabilities.
    AMX capabilities MUST NOT introduce a separate program topology level.
    The modelled atom catalogue MAY be narrower than the supported compute DTypes.
  - AppleAmx MUST own the X/Y staging and Z accumulator register files. They are
    ISA geometry, so every part carrying this coprocessor shares them and a
    device MUST NOT restate them. `staging_bytes` MUST be the size of one
    staging file, the X and Y files being equal.
  - `rmem_owner` MUST be `thread`: logical register state belongs to the issuing
    CPU thread. This does not imply one physical AMX engine per thread.
  - Product- and frequency-dependent throughput values MUST NOT be stored on
    AppleAmx.
  - AMX has no CTA thread level, so AppleAmx MUST carry no CTA thread limit.
  - No field MAY carry a default: every value comes from the installed
    document ([target §1](../../../../../../docs/spec/target.md#1-target)0).

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
    ([target §1](../../../../../../docs/spec/target.md#1-target)0). No field MAY carry a default.
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
  - Generic `thread` PerformanceServiceFacts MUST leave rates unknown while
    placement between performance and efficiency cores is unspecified. Existing
    P-core observations and estimates MUST NOT be projected as equal rates for
    all CPU threads. Stored AMX rates and unit-count estimates MUST NOT be
    multiplied by the CPU thread count.


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

    def validate_program_topology(self, topology: Topology) -> None: ...
    def get_analyzer(self, selector: str) -> Analyzer: ...
    def get_facts(self, facts_type: type[FactsT], query=None) -> FactsT: ...
```

- constraints:
  - `device` and `architecture` MUST accept an installed document ID, a
    document path, or a concrete value, on the same terms as
    [target §4](../../../../../nvidia/tilefoundry_nvidia/cuda/docs/spec/target.md#4-cudatarget). An omitted architecture MUST be read from the device
    document's sole compatibility declaration.
  - `AmxTarget()` MUST select the installed `apple.amx` and `apple.m2_pro`
    documents, and `arch` MUST equal
    `architecture.name`.
  - `AmxTarget.available()` MUST contain one value per device document whose
    sole compatible architecture document is available. Its `identity` MUST be
    that device document's ID.
  - `get_facts(TopologyFacts).topologies` MUST contain only `thread`, with logical
    and physical execution-position limits equal to performance plus efficiency
    core counts (12 for the installed M2 Pro). It MUST NOT set `from_target`.
    `parallel_level` MUST be `thread`. These limits describe program workers,
    not independent AMX matrix engines.
  - Declared extents MUST be positive and bounded by that limit, including the
    complete admitted range of a symbolic extent. `amx` MUST be rejected as a
    topology name. Memory execution MUST use the same `thread` vocabulary.
  - Unsupported topology levels MUST raise an actionable error naming the
    supported levels, from both the limit lookup and topology validation.


## Full instruction catalogue

The installed architecture document admits each declared instruction capability
as `amx_<mnemonic>`. Per-instruction Patterns check legal dtype/mode combinations;
full catalogue support includes F16/BF16/F32/F64 and signed/unsigned8/16/32-bit
views where the particular instruction defines them. A dtype listed by the
architecture does not make every instruction accept it.

X and Y each have the `staging_bytes` capacity, while Z has
`accumulator_bytes`. The combined physical RMEM capacity is therefore
`2 * staging_bytes + accumulator_bytes` (5120B for this architecture), not one
staging bank plus Z. The evaluator's logical lifecycle token is excluded. Instruction operands describe logical issue fragments; the full physical bank state is internal to emulation.

Instruction semantics are encoded by static versioned Ops, selected only by
architecture capabilities. There are 39 registered Ops; each architecture lists
24 capabilities: 15 unchanged instruction names and nine versioned names of the
form `amx_<mnemonic>_vN`. Neither a public instruction version parameter nor a
Target generation Facts bridge participates in selection.

| Architecture ID | LDX/LDY | EXTRX/EXTRY | MATFP | VECFP | MATINT | VECINT | GENLUT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apple.amx_m1` | v1 | v1 | v1 | v1 | v1 | v1 | v1 |
| `apple.amx` (M2) | v2 | v2 | v2 | v2 | v1 | v2 | v2 |
| `apple.amx_m3` | v3 | v2 | v2 | v2 | v2 | v2 | v2 |
| `apple.amx_m4` | v3 | v3 | v2 | v3 | v2 | v3 | v2 |

The unchanged capabilities are SET, CLR, LDZ, LDZI, STX, STY, STZ, STZI,
FMA16/32/64, FMS16/32/64, and MAC16, retaining names such as `amx_fma32`.
The installed `apple.amx` ID remains the M2 capability catalogue, preserving the
existing `apple.m2_pro` compatibility reference and default `AmxTarget()`.
The additional architecture documents reuse the same schema and `AppleAmx`
structure; they do not add M1/M3/M4 device specifications or measured rates.

These capability partitions come from the original C emulator and its
[emulate.h](https://github.com/corsix/amx/blob/main/emulate.h) convention
`AMX_VER_M1=1`, `AMX_VER_M2=2`, `AMX_VER_M3=3`, `AMX_VER_M4=4`.
The reference [README](https://github.com/corsix/amx/blob/main/README.md) records
BF16 added with M2, extra load/MATINT modes with M3, and changed extraction/vector
offset behavior with M4. They are reverse-engineered source claims, not native
measurements, Apple's official ISA numbering, or proof of loadability. CPU
oracle comparisons do not discover or exercise native AMX capabilities.
Existing measured performance entries remain unchanged; no new per-instruction
rate is invented. The runtime header is for agent reference and no AMX codegen
or automatic native dispatch is provided.
