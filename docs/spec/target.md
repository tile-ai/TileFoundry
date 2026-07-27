# TileFoundry Spec — Target

A `Target` is the immutable capability context used by compilation and
Target-owned HIR services. Architecture describes compilation identity and
instruction structure. Device describes fixed product resources. A target's
private service bindings are selected by exact interface identity and stage.

## 1. `Target`

```python
class Target:
    """Identify a compilation backend and its private stage services."""

    name: str
    _services: tuple[tuple[type, str, object], ...] = ()

    def service(self, interface: type, stage: str) -> object: ...

    def as_facts(self, facts_type: type, query: object = None) -> object: ...
```

- constraints:
  - `name` MUST be the stable backend identifier used for target resolution and
    codegen grouping.
  - `as_facts` MUST project this target's specification into the immutable
    aggregate a requesting algorithm declares, under the rules of
    [§11](#11-target-facts-projection).
  - `_services` MUST be immutable and populated only by target construction.
    It MUST NOT participate in equality, hashing, or `repr`.
  - `service` MUST require a non-empty stage string and match the interface by
    object identity plus one exact stage string.
  - Missing or duplicate matches MUST raise an actionable built-in error that
    names the target, interface, and stage.
  - Target values MUST NOT own code emission, linking, loading, or the public
    compile/build/jit entry points.

### 1.1 `Architecture`

```python
class Architecture:
    """Describe compilation architecture identity and structural facts."""

    name: str
    max_threads_per_cta: int
```

- constraints:
  - Concrete architecture values MUST be immutable.
  - `name` MUST be the stable architecture identity used by compilation.
  - `max_threads_per_cta` MUST describe the architecture's static CTA thread
    limit when the architecture has a CTA thread level.

### 1.2 `Device`

```python
class Device:
    """Describe one concrete device's fixed resource facts."""

    name: str
    sm_count: int
```

- constraints:
  - Concrete device values MUST be immutable and describe one device.
  - `name` MUST be the stable product identity.
  - Device-specific capacity, bandwidth, and compute-throughput facts belong
    to concrete subclasses.
  - The split between the two is by what the fact is a property of, not by
    which consumer reads it. An Architecture owns instruction legality and the
    per-parallel-unit structural limits, which every product built on it
    shares. A Device owns how many such units the product has, its memory
    system, and its measured or published throughput. A fact MUST be recorded on
    exactly one side, and the other side MUST NOT restate it.

## 2. `SM90`

```python
class SM90:
    """SM90 compilation identity and structural capabilities."""

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    instruction_capabilities: tuple[str, ...]
    max_threads_per_cta: int
    max_threads_per_warp: int
    max_warps_per_cta: int
    max_resident_ctas_per_sm: int
    shared_memory_per_sm_bytes: int
    shared_memory_per_cta_bytes: int
    registers_per_sm_32bit: int

    def supports_compute_dtype(self, dtype: DType) -> bool: ...

    def topology_limit(self, name: str) -> int: ...
```

- constraints:
  - `name` MUST be the architecture identity used by CUDA compilation.
  - SM90 MUST own supported compute DTypes, instruction capabilities, and the
    thread/CTA structural limits.
  - SM90 MUST own the per-SM resource limits: resident CTAs, shared-memory
    capacity per SM and per CTA, and register-file capacity per SM. These are
    properties of the microarchitecture, so every product built on it shares
    them, and a device MUST NOT restate them.
  - Storage and scale DTypes `f4e2m1` and `f8e8m0` MUST NOT be reported as
    compute DTypes by SM90.
  - Device-frequency-dependent FLOP/s values MUST NOT be stored on SM90.
  - No field MAY carry a default: every value comes from the installed
    document (§10), so the class declares shape and never content.

## 3. `H200SXM`

```python
class H200SXM:
    """One H200 SXM device with fixed hard resource limits."""

    name: str
    sm_count: int
    hbm_capacity_bytes: int
    hbm_bandwidth_bytes_per_second: int

    def peak_for(self, dtype: DType) -> int: ...
```

- constraints:
  - H200SXM MUST describe one device and MUST NOT carry a GPU count.
  - H200SXM MUST describe how many SMs the product has and how its memory
    system and compute units perform. Per-SM structural limits belong to the
    architecture (§2).
  - `peak_for` MUST expose a dense integer FLOP/s entry for each of `f32`,
    `f16`, `bf16`, and `fp8e4m3`, each value taken from the installed document.
  - `f4e2m1` and `f8e8m0` MUST have no compute-throughput entry.
  - Unknown compute DTypes MUST raise an actionable error.
  - No field MAY carry a default, and no resource value MAY be written as a
    Python literal: the installed document is the single source (§10).
    Selecting a different installed document by ID is not an override; supplying
    a partial or edited number without a document behind it is, and is not
    admitted.

## 4. `CudaTarget`

```python
class CudaTarget(Target):
    """CUDA target composed from one architecture and one device."""

    name: str = "cuda"
    architecture: Architecture
    device: Device
    architecture_id: str | None
    device_id: str | None
    architecture_digest: str | None
    device_digest: str | None
    arch: str
    topology_levels: tuple[str, ...]

    def __init__(
        self,
        architecture: Architecture | str | None = None,
        device: Device | str | None = None,
    ) -> None: ...

    def topology_limit(self, name: str) -> int | None: ...

    def validate_program_topology(self, topology: Topology) -> None: ...
```

- constraints:
  - `architecture` and `device` MUST each accept an installed document ID or a
    concrete value. An ID MUST resolve immediately to the typed value, and the
    resolved ID and content digest MUST be retained (§10.2).
  - `CudaTarget()` MUST select the installed `nvidia.sm90` and
    `nvidia.h200_sxm` documents, and `arch` MUST equal `architecture.name`.
  - A pair selected by ID MUST be checked for declared compatibility. A value
    supplied directly carries no document, so it has no ID or digest and is
    exempt from that check: it is a distinct hardware value rather than a
    revision of an installed one.
  - `topology_levels` MUST be `("cta", "thread")` for this single-device
    target. Warp/lane/warpgroup structure belongs in thread mesh layouts.
  - `topology_limit("cta")` MUST be `None`: the CUDA grid is a launch shape
    rather than an SM allocation, so its static extent is unbounded here.
    `topology_limit("thread")` MUST equal `architecture.max_threads_per_cta`.
  - Each `CudaTarget` instance MUST bind exactly one private
    `(Schedule, "cta")` service, including instances constructed with custom
    `Device` or `Architecture` values. The concrete service implementation is
    not part of the public `schedule` package.
  - The CTA-level Analysis service MUST report a tile capacity of
    `architecture.shared_memory_per_cta_bytes`.
  - Static declared topology extents MUST be positive integers within their
    target resource limits. `Topology("cta", None)` MUST remain valid for the
    handwritten dynamic-launch compile path.
  - Unsupported topology levels MUST fail at the generic lowering boundary.
  - A `CudaTarget` MUST currently expose no concrete CTA scheduling service.

## 5. `CpuTarget`

```python
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name: str = "cpu"
```

- constraints:
  - `name` MUST be `"cpu"`.
  - CPU host Functions MAY coexist with CUDA Functions in one module and are
    exempt from CUDA hardware-fact equality checks.

## 6. Target ownership and compile resolution

- `tilefoundry.target` MUST be the sole Target implementation package. The IR
  package MUST NOT own Target classes or Target imports.
- `resolve_target("cuda")` MUST return a default `CudaTarget`,
  `resolve_target("amx")` MUST return a default `AmxTarget`,
  `resolve_target("cpu")` MUST return a `CpuTarget`, and a Target object MUST
  pass through unchanged.
- A `Target` MUST be declared by the `Module` that owns the functions running
  on it, never by an authored HIR `Function`. Analyze, Schedule, and compile
  MUST obtain it through `Module.resolve_target()`
  ([core-ir §1](./core-ir.md#1-module)).
- Only the outermost `Module` of a tree declares a `Target`; every Module
  below it inherits that one declaration and MUST NOT declare its own. A
  Module that is reused both as an owned child and as an independently
  analysed root therefore declares its Target only in the second role.
- `Module.resolve_target()` MUST fail when no Module in the owner chain
  declares a Target.
- Analyze and Schedule MUST obtain the Target from `Module.resolve_target()`
  and from nowhere else. Neither accepts a bare `Function`, and neither
  resolves an undeclared Target to a default: both report hardware-dependent
  results, so measuring or scheduling against a device the author never
  declared is a silent wrong answer. In particular neither reads a Target out
  of `Module.metadata`; the `metadata["target"]` the compile pipeline carries
  is the codegen boundary's own record
  ([passes §6](./passes.md#6-top-level-api)), not a Target source
  for Analyze or Schedule.
- The compile boundary MAY resolve that omission to the default CUDA target
  for lowering, because `jit(fn)` on a plain Function is a documented entry
  point ([runtime §1.3](./runtime.md#13-jit-api)) and codegen selects
  its emitter from the lowered `PrimFunction.target` rather than from the
  Module.
- A lowered TIR `PrimFunction` retains its own `target`: after lowering it
  selects the emitter that lowers it, which is how one Module's host and
  device functions reach different backends.
- After target resolution, CUDA Functions in one compilation group MUST carry
     equal architecture and device facts. A mismatch MUST fail before codegen
     grouping.

## 7. `AppleAmx`

```python
class AppleAmx:
    """Describe AMX compilation identity and structural capabilities."""

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    instruction_capabilities: tuple[str, ...]
    amx_units_per_core: int
    staging_bytes: int
    accumulator_bytes: int

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
  - Product- and frequency-dependent throughput values MUST NOT be stored on
    AppleAmx.
  - AMX has no CTA thread level, so AppleAmx MUST carry no CTA thread limit.
  - No field MAY carry a default: every value comes from the installed
    document (§10).

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
    unified_memory_bandwidth_bytes_per_second: int

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
    (§10). No field MAY carry a default.
  - A core-level tile's resident footprint MUST be bounded by
    `l1d_bytes_per_performance_core`. The AMX register files bound one atom
    instance instead, which the storage filter enforces rather than a per-tile
    capacity, so the two MUST NOT be conflated.
  - `throughput_for` MUST be keyed by execution unit as well as DType, because
    the AMX coprocessor and the core's NEON pipes have separate measured rates.
  - A tile's traffic MUST be charged against
    `unified_memory_bandwidth_bytes_per_second`, which unified memory backs.
  - `throughput_for` MUST return a measured per-unit throughput recorded in the
    installed document, and MUST raise an actionable error for a unit or compute
    DType with no measured entry rather than return an estimate.

## 9. `AmxTarget`

```python
class AmxTarget(Target):
    """Compose one AMX target from one architecture and one device."""

    name: str = "amx"
    architecture: Architecture
    device: Device
    architecture_id: str | None
    device_id: str | None
    architecture_digest: str | None
    device_digest: str | None
    arch: str
    topology_levels: tuple[str, ...]

    def __init__(
        self,
        architecture: Architecture | str | None = None,
        device: Device | str | None = None,
    ) -> None: ...

    def topology_limit(self, name: str) -> int: ...

    def validate_program_topology(self, topology: Topology) -> None: ...
```

- constraints:
  - `architecture` and `device` MUST accept an installed document ID or a
    concrete value, on the same terms as §4.
  - `AmxTarget()` MUST select the installed `apple.amx` and `apple.m2_pro`
    documents, and `arch` MUST equal
    `architecture.name`.
  - `topology_levels` MUST be `("core", "amx")`: the performance core one tile
    stream runs on, and the AMX unit inside that core which issues one atom.
  - `topology_limit("core")` MUST equal `device.performance_core_count` and
    `topology_limit("amx")` MUST equal `architecture.amx_units_per_core`.
  - Declared topology extents MUST be positive static integers within their
    level's limit. AMX has no launch shape, so a deferred or symbolic extent
    MUST NOT be admitted at either level.
  - Unsupported topology levels MUST raise an actionable error naming the
    supported levels, from both the limit lookup and topology validation.
  - Each `AmxTarget` instance MUST bind exactly one private
    `(Analysis, "core")` and one private `(Schedule, "core")` service. The
    concrete implementations are not part of the public `analysis` or
    `schedule` packages.
  - The core Analysis service MUST list an op's atom candidates by hard
    filtering the registered catalogue, and MUST NOT rank them. The filter is
    shape divisibility, operand DType, operand layout, and the storage level
    the atom's operand roles need — the last is what separates a
    register-resident atom from one streaming through cache, so an op too wide
    for the register files lists only the streaming atom.
  - An op that clears no filter MUST report an empty candidate list, which is a
    covered op with no usable atom rather than an error. Only an op kind or a
    target the bridge does not model at all MUST raise.
  - The core Schedule service MUST decide resources over the schedule tree
    extracted from its root and report the objective in ns. It materializes
    nothing at this stage, so the module it returns MUST be the module it was
    given.

## 10. Installed hardware resources

Architecture and Device documents are the canonical authored hardware
database, and the only place a hardware number is written. Each is a complete
document in its own right; a target is the pair composed through a declared
compatibility, never a single combined record.

### 10.1 Document envelope

```toml
[spec]
schema = "tilefoundry.cuda.device/v1"
kind = "device"
id = "nvidia.h200_sxm"

[compatibility]
architectures = ["nvidia.sm90"]

[facts.memory.hbm.bandwidth]
value = 4800000000000
unit = "byte/s"
origin = "vendor"
source = "https://www.nvidia.com/en-us/data-center/h200/"
conditions = "4.8 TB/s peak HBM3e bandwidth, decimal"

[facts.memory.l2.bandwidth]
status = "unavailable"
conditions = "No validated number."
```

- constraints:
  - The envelope MUST carry exactly `schema`, `kind`, and `id`. `kind` MUST be
    `architecture` or `device`. An unknown envelope key MUST fail.
  - An architecture document MUST declare compatibility under `devices` and a
    device document under `architectures`. A pair MUST compose only when at
    least one side names the other; neither MUST be inferred.
  - Tables under `facts` are freely nestable namespaces owned by the target
    package named by `schema`. A leaf is identified by carrying `value` or an
    explicit `status`.
  - An available leaf MUST carry `value` and `origin`. An unavailable leaf MUST
    omit `value`, record `status = "unavailable"`, and state the reason in
    `conditions`. The string `"unavailable"` MUST NOT be used as a value, so no
    caller can read a placeholder as a number.
  - `origin` MUST name how the value was obtained: `vendor` from the vendor's
    published figure, `measured` on the described host, `reference` from a cited
    third party, `derived` from other facts, or `estimated` where it is a
    reading that no source states. A value not measured on the described host
    MUST NOT be recorded as `measured`; a reading rather than a citation MUST be
    recorded as `estimated`. `derived` and `estimated` MUST state how in
    `conditions`.
  - Compiler policy and program Topology MUST NOT appear in a hardware
    document. They are inputs to scheduling, not immutable hardware truth: a
    fixed-wave parallel capacity is a scheduling policy even when its current
    value equals a device count.

### 10.2 Registry and resolution

- constraints:
  - `HardwareSpecRegistry` MUST resolve documents by exact ID. There MUST be no
    search path, no overlay, and no partial document.
  - A target package MUST register its typed schemas and installed documents as
    an import side effect, into the same shared registry.
  - A typed schema MUST validate exact fact paths, value types, units, required
    fields, and cross-field invariants, and MUST reject any leaf the document
    carries that the schema does not model, so a misspelled key cannot become
    an unused fact.
  - Units MUST be normalized while constructing the typed value: algorithms see
    canonical integers such as bytes and bytes per second, never source strings
    or unit conversion.
  - Resolution MUST retain each document's ID and content digest on the
    composed value, so a compiled artifact can name the exact resources it was
    built against. Editing any recorded value or its evidence MUST change the
    digest.
  - A custom document MUST be loadable through an explicit path API, MUST be
    complete, and MUST NOT enter the installed-ID namespace, so it can neither
    shadow nor replace an installed resource.
  - Unknown IDs, unknown schemas, unmodelled or malformed facts, malformed
    envelopes, duplicate registrations, and incompatible pairs MUST each raise
    their own actionable diagnostic rather than one shared parse failure.
  - Reporting the resources behind a target MUST name both documents and their
    digests. A target composed from a directly supplied value has no document to
    report and MUST say so rather than name the installed resource it resembles.

## 11. Target Facts projection

A target-aware algorithm declares the immutable aggregate of facts it needs; the
Target package registers the conversion that builds it.
[`Target.as_facts`](#1-target) is the one boundary between a hardware
specification and an algorithm's own view of it, and delegates to this registry.

```python
class TargetFactsRegistry:
    """Conversions from a concrete Target to an algorithm's Facts aggregate."""

    def register(
        self, target_type: type, facts_type: type, conversion: FactsConversion
    ) -> None: ...

    def project(
        self, target: Target, facts_type: type, query: object = None
    ) -> object: ...


def register_target_facts(
    target_type: type, facts_type: type, conversion: FactsConversion
) -> None: ...
```

- constraints:
  - A conversion MUST be registered under the exact
    `(Target concrete type, Facts type)` pair. Resolution MUST use the target's
    exact concrete type: a base-class registration MUST NOT serve a subclass,
    because two targets sharing a base can describe different hardware, and a
    Facts type MUST be identified by the class itself rather than by its name.
  - A missing conversion MUST fail immediately. A target-aware algorithm MUST
    NOT fall back to a default projection; only an algorithm explicitly declared
    target-independent may run without Target Facts.
  - A duplicate registration for one exact pair MUST fail, so a projection
    cannot depend on import order.
  - A Facts aggregate MUST be a frozen dataclass. The constraint MUST be checked
    when the conversion is registered rather than trusted at each projection.
    Aggregates MUST NOT inherit one universal Facts base.
  - Every conversion MUST have the same call shape, `as_facts(FactsType,
    query=None)`. `query` is owned by the requesting algorithm: a hardware-only
    projection MUST require it to be absent, while a program-dependent one MAY
    validate its own private query type. There MUST be no common query base and
    no mandatory public program-view type.
  - A conversion returning a value that is not an instance of the requested
    Facts type MUST fail at the boundary, not inside the algorithm.
  - Projection MUST be a read. It MUST NOT analyze IR, build a constraint model,
    solve, export a plan, or mutate the Target, the IR, the registry, or runtime
    state. It only converts what the specification already records.
  - The registry MUST be generic: the common code MUST NOT import or name a
    concrete target, architecture, or device class, so adding a backend adds a
    registration rather than a branch.
  - The hardware-specification registry (§10.2), the algorithm registry, and the
    Target Facts registry MUST remain distinct module-level registries.
