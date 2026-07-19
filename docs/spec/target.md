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
  `resolve_target("cpu")` MUST return a `CpuTarget`, and a Target object MUST
  pass through unchanged.
- Authored HIR `Function.target` MUST default to `None`. A normal compile
  boundary MAY resolve that omission to the default CUDA target for lowering,
  but scheduling lookup MUST NOT apply that fallback.
- After target resolution, CUDA Functions in one compilation group MUST carry
  equal architecture and device facts. A mismatch MUST fail before codegen
  grouping.
