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
```

- constraints:
  - `name` MUST be the stable backend identifier used for target resolution and
    codegen grouping.
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

## 2. `SM90`

```python
class SM90:
    """SM90 compilation identity and structural capabilities."""

    name: str = "sm_90"
    supported_compute_dtypes: tuple[DType, ...] = ...
    instruction_capabilities: tuple[str, ...] = ...
    max_threads_per_cta: int = 1024
    max_threads_per_warp: int = 32
    max_warps_per_cta: int = 32

    def supports_compute_dtype(self, dtype: DType) -> bool: ...

    def topology_limit(self, name: str) -> int: ...
```

- constraints:
  - `name` MUST be the architecture identity used by CUDA compilation.
  - SM90 MUST own supported compute DTypes, instruction capabilities, and
    thread/CTA structural limits.
  - Storage and scale DTypes `f4e2m1` and `f8e8m0` MUST NOT be reported as
    compute DTypes by SM90.
  - Device-frequency-dependent FLOP/s values MUST NOT be stored on SM90.

## 3. `H200SXM`

```python
class H200SXM:
    """One H200 SXM device with fixed hard resource limits."""

    name: str = "h200_sxm"
    sm_count: int = 132
    hbm_capacity_bytes: int = 141_000_000_000
    hbm_bandwidth_bytes_per_second: int = 4_800_000_000_000

    def peak_for(self, dtype: DType) -> int: ...
```

- constraints:
  - H200SXM MUST describe one device and MUST NOT carry a GPU count.
  - The resource values MUST be fixed to the stated decimal-SI constants;
    callers MUST NOT provide lower effective SM-count, bandwidth, or capacity
    overrides.
  - `peak_for` MUST expose the dense integer FLOP/s map:
    `f32: 67_000_000_000_000`, `f16: 989_500_000_000_000`,
    `bf16: 989_500_000_000_000`, and
    `fp8e4m3: 1_979_000_000_000_000`.
  - `f4e2m1` and `f8e8m0` MUST have no compute-throughput entry.
  - Unknown compute DTypes MUST raise an actionable error.

## 4. `CudaTarget`

```python
class CudaTarget(Target):
    """CUDA target composed from one architecture and one device."""

    name: str = "cuda"
    architecture: SM90 = SM90()
    device: H200SXM = H200SXM()
    arch: str
    topology_levels: tuple[str, ...]

    def topology_limit(self, name: str) -> int: ...

    def validate_program_topology(self, topology: Topology) -> None: ...
```

- constraints:
  - `CudaTarget()` MUST use SM90 and H200SXM, and `arch` MUST equal
    `architecture.name`.
  - `topology_levels` MUST be `("cta", "thread")` for this single-device
    target. Warp/lane/warpgroup structure belongs in thread mesh layouts.
  - `topology_limit("cta")` MUST equal `device.sm_count` and
    `topology_limit("thread")` MUST equal `architecture.max_threads_per_cta`.
  - Each `CudaTarget` instance MUST bind exactly one private
    `(Schedule, "cta")` service, including instances constructed with custom
    `Device` or `Architecture` values. The concrete service implementation is
    not part of the public `schedule` package.
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
- Authored HIR `Function.target` MUST default to `None`. A normal compile
  boundary MAY resolve that omission to the default CUDA target for lowering,
  but scheduling lookup MUST NOT apply that fallback.
- After target resolution, CUDA Functions in one compilation group MUST carry
     equal architecture and device facts. A mismatch MUST fail before codegen
     grouping.

## 7. `AppleAmx`

```python
class AppleAmx:
    """Describe AMX compilation identity and structural capabilities."""

    name: str = "apple_amx"
    supported_compute_dtypes: tuple[DType, ...] = ...
    instruction_capabilities: tuple[str, ...] = ...
    amx_units_per_core: int = 1

    def supports_compute_dtype(self, dtype: DType) -> bool: ...

    def topology_limit(self, name: str) -> int: ...
```

- constraints:
  - `name` MUST be the architecture identity used by AMX compilation.
  - AppleAmx MUST own the supported compute DTypes and the per-core AMX unit
    count. The modelled atom catalogue MAY be narrower than the supported
    compute DTypes.
  - Product- and frequency-dependent throughput values MUST NOT be stored on
    AppleAmx.
  - AMX has no CTA thread level, so AppleAmx MUST carry no CTA thread limit.

## 8. `AppleM2Pro`

```python
class AppleM2Pro:
    """Describe the apple_m2_pro package's fixed hard resource limits."""

    name: str = "apple_m2_pro"
    sm_count: int = 2
    performance_core_count: int = 8
    l1d_bytes_per_performance_core: int = 131_072
    l2_bytes_per_performance_cluster: int = 16_777_216
    unified_memory_bandwidth_bytes_per_second: int = 200_000_000_000
    amx_staging_bytes: int = 512
    amx_accumulator_bytes: int = 4096
    l1_capacity_bytes: int
    l2_bandwidth_bytes_per_second: int

    def throughput_for(self, dtype: DType) -> int: ...
```

- constraints:
  - AppleM2Pro MUST describe one package and MUST NOT carry a machine count.
  - `sm_count` MUST be the number of independent AMX units, which is the
    parallel-unit count a makespan divides work over. It MUST NOT be read as a
    core count: the performance cores outnumber the units and share them.
  - Cache and core facts MUST describe the performance core, not the efficiency
    core, and MUST be fixed to the values recorded in the installed hardware
    specification. Callers MUST NOT provide overrides.
  - `l1_capacity_bytes` MUST be the capacity one tile's resident footprint is
    bounded by, which on this device is the AMX accumulator file
    `amx_accumulator_bytes`. The staging and bulk levels of the AMX storage
    hierarchy MUST NOT be conflated with it.
  - `l2_bandwidth_bytes_per_second` MUST be the bandwidth a tile's traffic is
    charged against, which on this device is
    `unified_memory_bandwidth_bytes_per_second`.
  - `throughput_for` MUST return a measured per-unit AMX throughput recorded in
    the installed hardware specification, and MUST raise an actionable error for
    a compute DType with no measured entry rather than return an estimate.

## 9. `AmxTarget`

```python
class AmxTarget(Target):
    """Compose one AMX target from one architecture and one device."""

    name: str = "amx"
    architecture: AppleAmx = AppleAmx()
    device: AppleM2Pro = AppleM2Pro()
    arch: str
    topology_levels: tuple[str, ...]

    def topology_limit(self, name: str) -> int: ...

    def validate_program_topology(self, topology: Topology) -> None: ...
```

- constraints:
  - `AmxTarget()` MUST use AppleAmx and AppleM2Pro, and `arch` MUST equal
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

## 10. Installed hardware facts

- Each installed hardware specification is the source-attributed record of the
  device and architecture facts one target stands on.
- constraints:
  - Every fact MUST carry `value`, `unit`, `provenance`, `conditions` and
    `source`.
  - `provenance` MUST name how the value was obtained: `measured` on the
    described host, `vendor-spec` from the vendor's published figure, `direct`
    from a cited reference, `derived` from other facts in the same
    specification, `estimated` where it is a reading that no source states, or
    `unavailable` where no value exists.
  - A value that was not measured on the described host MUST NOT be recorded as
    `measured`, and a value that is a reading rather than a citation MUST be
    recorded as `estimated`.
  - A fact with no available value MUST still be recorded, as `unavailable`
    with the reason in `conditions`, so that the gap is explicit.
  - A compiler policy MUST be recorded as a fact in its own right, named as
    policy and attributed to the compiler rather than to the hardware.
  - `load_hardware_spec` MUST resolve one exact authored target, and MUST raise
    an actionable error naming the device and architecture when no
    specification is installed for it.
