# TileFoundry CUDA Spec — Target

## 4. `CudaTarget`

```python
class CudaTarget(Target):
    """CUDA target composed from one architecture and one device."""

    name: ClassVar[str] = "cuda"
    architecture: Architecture
    device: Device
    architecture_id: str | None
    device_id: str | None
    architecture_digest: str | None
    device_digest: str | None
    arch: str

    def __init__(
        self,
        device: Device | str | Path,
        architecture: Architecture | str | Path | None = None,
        *,
        arch: str | None = None,
    ) -> None: ...

    def validate_program_topology(self, topology: Topology) -> None: ...
    def topology_limit(self, name: str) -> int: ...
    def get_analyzer(self, selector: str) -> Analyzer: ...
    def torch_device_type(self) -> str: ...
    def get_facts(self, facts_type: type[FactsT], query=None) -> FactsT: ...
    def __repr__(self) -> str: ...
```

- constraints:
  - `device` and `architecture` MUST each accept an installed document ID or a
    concrete value. An ID MUST resolve immediately to the typed value, and the
    resolved ID and content digest MUST be retained
    ([target §10.2](../../../../../docs/spec/target.md#102-registry-and-resolution)).
  - `arch`, when provided, MUST equal the resolved architecture's `name`; it is
    a consistency check and MUST NOT select or override an architecture.
  - `device` MUST be required. The constructor MUST NOT select hardware for a
    caller who named none: a target nobody stated would answer about a machine
    nobody has.
  - An omitted `architecture` MUST be read from the device document's declared
    compatibility, and MUST fail unless that document names exactly one. A
    `Device` supplied directly carries no document, so it MUST be given an
    architecture as well.
  - `arch` MUST equal `architecture.name`.
  - A pair selected by ID MUST be checked for declared compatibility. A value
    supplied directly carries no document, so it has no ID or digest and is
    exempt from that check: it is a distinct hardware value rather than a
    revision of an installed one.
  - `CudaTarget` MUST compose any installed architecture and device pair that
    declares compatibility. A further CUDA product is its two documents
    ([target §10.2](../../../../../docs/spec/target.md#102-registry-and-resolution)) and nothing else: no Target subclass,
    and no architecture or device type of its own. The services and Facts are
    selected by the value already, and the numbers are in the documents, so
    either addition would carry nothing.
  - `CudaTarget.available()` MUST contain one value per device document whose
    sole compatible architecture document is available. Its `identity` MUST be
    that device document's ID.
  - CUDA MUST select its standard Analyzer and Facts through the corresponding
    getters. These selections MUST NOT use
    `Target.name`, an exact-concrete-type table, or a second extension
    registration.
  - CUDA's `ThroughputFacts` and `PerformanceServiceFacts` MUST both read
    `rates(target, unit, active)`, the one source of CUDA rates: what `active`
    units of `unit` get through together. A query is a level name or
    `(level, active, inflight)`. `active` counts only at `cta`, where it MUST be
    capped at the units the deployment holds; any other level, and a bare level,
    runs all of its units. `inflight` is the bytes one CTA keeps in flight for
    the access being priced. `ThroughputFacts` is `rates` itself, the
    whole deployment when asked about no level; `PerformanceServiceFacts` is
    `rates` divided by `active`, rounding down. CUDA MUST name its default unit
    `cta`; `TopologyLevelFacts.max_physical_units` MUST NOT enter this
    derivation.
  - `rates` MUST scale each deployment dense-FLOP rate, each service rate
    (`unit_ops` from the device document, per SM) and the shared-memory
    bandwidth (the architecture's byte/clock/SM rate times the device's typed SM
    clock) by `active` over the units the deployment holds. Global-memory
    bandwidth MUST be the HBM peak and, at `cta` on a device stating
    `gmem_latency_ns`, no more than `active * inflight / gmem_latency_ns`: by
    Little's law one CTA with `inflight` bytes outstanding moves at most that
    many bytes per latency. One 1024-thread CTA with 16 B per thread in flight
    moves about 46 GB/s on H200, so about 103 CTAs reach its 4.8 TB/s. A device
    without the fact, or a query without `inflight`, has no cap. `unit_ops` is not a division of a device peak, because no vendor
    publishes a device-wide integer, predicate, select, special or local-move
    rate to divide.
  - `topology_limit("cta")` MUST equal `device.sm_count`, and
    `topology_limit("thread")` MUST equal the architecture's corresponding
    structural limit. An unsupported name MUST be refused.
  - `__repr__` MUST use the concrete class name and return a constructor
    expression that rebuilds an equal Target. A subclass retaining the CUDA
    constructor shape inherits it; a subclass with another constructor MUST
    override `__repr__`.
  - The store the threads of one CTA cooperate in MUST be projected as
    `architecture.shared_memory_per_cta_bytes`, and MUST be reported as belonging
    to the `cta` scope even when the level being asked about is `thread`.
  - The tensor-memory level MUST be projected with
    `architecture.tensor_memory_per_cta_bytes` only where the architecture states
    a capacity, and MUST be absent from `explicit_levels` where it states `None`.
    A level on hardware that has no such store would offer a plan somewhere to
    hold accumulators that does not exist.

#### Topology levels

A target's topology levels define the names a program may declare. The program
hierarchy stops at those levels; warp, lane, and warpgroup structure belongs in
thread mesh layouts.

- constraints:
  - `get_facts(TopologyFacts).topologies` MUST name `("gpu", "cta", "thread")`,
    outermost first. A program names `gpu` when its data is divided across
    cards.
  - A topology level whose `from_target` is set takes its extent and its
    program ids from the target instance rather than from the hardware
    document or a register. `"gpu"` MUST be the only CUDA level that sets it.
  - A declared program topology name MUST be one of the levels its target
    states. A name outside that set MUST be refused naming the levels the
    target declares.
  - `get_facts(TopologyLevelFacts, "cta").max_logical_units` MUST be `None`:
    the CUDA grid is a launch shape rather than an SM allocation, so its static
    extent is unbounded here. The `"thread"` Facts projection MUST equal
    `architecture.max_threads_per_cta`.
  - `Topology.size` MUST be an explicit `ShapeDim`; construction with `None`
    MUST fail for every topology level.
  - Static declared topology extents MUST be positive integers within their
    target resource limits.
  - Unsupported topology levels MUST fail at the generic lowering boundary.

#### Instruction support and execution resources

An instruction declaration keeps two independent names:

| Axis | Question | Vocabulary |
|---|---|---|
| `capability` | May this architecture select the instruction family? | Emitted ISA name |
| `resource` | Which execution engine performs it? | An `_engine` name |

CUDA declarations use `mma.sync`, `wgmma.mma_async`, `ldmatrix`, `stmatrix`, `cp.async`,
`cp.async.bulk`, `cp.async.bulk.tensor`, `mbarrier.expect_tx` and
`mbarrier.try_wait` as capabilities. Admission MUST refuse an instruction whose
declared capability the target's architecture does not list, naming the
capability and the target; where an attribute selects one of several
declarations (an MMA atom), only the selected declaration's capability applies.
A target without an architecture lists none, and admission without a target
checks nothing. The resource-bearing declarations
map the two MMA families, matrix load/store, and tensor-map copy respectively to
`tensor_core_engine`, `wgmma_engine`, `tensor_core_engine`, and `tma_engine`;
the current model gives `cp.async` no separately priced engine. The capability
and resource vocabularies MUST be disjoint. Architecture documents list only
capabilities under `[facts.capabilities]`; an instruction declaration is the
single source of any resource mapping it states. A resource declaration does
not imply a throughput fact, and MUST NOT be priced as a service unless a
consumer defines a quantity with matching units.

### 4.1 `CudaArchitecture`

```python
class CudaArchitecture(Architecture):
    """What one CUDA architecture states about itself."""

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    capabilities: tuple[str, ...]
    max_threads_per_cta: int
    max_threads_per_warp: int
    max_warps_per_cta: int
    max_resident_ctas_per_sm: int
    shared_memory_per_sm_bytes: int
    shared_memory_per_cta_bytes: int
    smem_owner: str
    shared_memory_bandwidth_bytes_per_clock_per_sm: int | None
    unified_l1_shared_per_sm_bytes: int
    registers_per_sm_32bit: int
    rmem_owner: str
    tensor_memory_per_cta_bytes: int | None
    tmem_owner: str

    def supports_compute_dtype(self, dtype: DType) -> bool: ...

    def topology_limit(self, name: str) -> int: ...
```

- constraints:
  - One value type MUST answer for every CUDA architecture, and a concrete value
    MUST be immutable. What separates one architecture from another is what its
    installed document records ([§4.1.1](#411-sm90),
    [§4.1.2](#412-sm100), [§4.1.3](#413-sm89)), so an architecture MUST NOT be
    given a type of its own: a class holding no number would restate an
    identity the value already carries.
  - `name` MUST be the architecture identity CUDA compilation uses.
  - A CUDA architecture MUST own supported compute DTypes, instruction
    capabilities, and the thread/CTA structural limits. `capabilities` MUST use
    instruction-family names from the emitted ISA, such as `mma.sync`,
    `wgmma.mma_async`, `ldmatrix`, `stmatrix`, `cp.async`, and `cp.async.bulk.tensor`.
  - It MUST own the per-SM resource limits: resident CTAs, shared-memory
    capacity per SM and per CTA, and register-file capacity per SM. These are
    properties of the microarchitecture, so every product built on it shares
    them, and a device MUST NOT restate them.
  - A shared-memory bandwidth stated in byte/clock/SM MUST remain an
    architecture fact. Turning it into byte/s requires the paired device's SM
    clock and happens only in the CUDA `rates` projection.
  - It MUST NOT carry a compute-throughput rate. A FLOP/s figure depends on the
    clock of one product, so it is a device fact ([§4.2](#42-cudadevice)) even
    though the instruction it rates is the architecture's.
  - `unified_l1_shared_per_sm_bytes` MUST be the size of the one physical block
    the shared-memory carveout and the L1 data cache are both taken from, and
    MUST be at least `shared_memory_per_sm_bytes`. The architecture MUST NOT
    state an L1 capacity: how much L1 remains depends on how much shared memory
    a program asked for, which is not a property of the hardware.
  - `tensor_memory_per_cta_bytes` MUST be the whole tensor-memory store on an
    architecture whose MMA accumulates in one, because that store is allocated
    in columns spanning every lane and one CTA may hold all of them. It MUST be
    `None` where the architecture has no such store, which is the same statement
    the document makes by recording that leaf unavailable.
  - CUDA explicit memory ownership MUST be read from the installed hardware
    values as `smem -> cta`, `rmem -> thread`, and `tmem -> cta`. `scope` MUST
    NOT be used as an ownership map: register capacity is stated per SM while
    each thread owns its register values.
  - Every architecture document MUST declare the leaf behind every field, so a
    value the architecture does not have is recorded as absent rather than left
    out ([target §10.2](../../../../../docs/spec/target.md#102-registry-and-resolution)).
  - No field MAY carry a default: every value comes from the installed document
    ([target §10](../../../../../docs/spec/target.md#10-installed-hardware-resources)), so the type declares shape and
    never content.

#### 4.1.1 SM90

The installed `nvidia.sm90` architecture document.

- constraints:
  - Its recorded identity MUST be `sm_90`.
  - Storage and scale DTypes `f4e2m1` and `f8e8m0` MUST NOT be recorded as
    compute DTypes.
  - It MUST record no tensor-memory capacity: the SM90 MMA accumulates in the
    register file, so there is no separate store to size.

#### 4.1.2 SM100

The installed `nvidia.sm100` architecture document.

- constraints:
  - Its recorded identity MUST be `sm_100`.
  - It MUST record `f4e2m1` as a compute DType, because the SM100 MMA takes
    4-bit operands directly. `f8e8m0` MUST NOT be recorded as one: it scales
    those operands rather than being multiplied.
  - It MUST record a tensor-memory capacity, and its instruction capabilities
    MUST name the MMA family that accumulates there rather than the SM90 family,
    which this architecture does not run.

#### 4.1.3 SM89

The installed `nvidia.sm89` architecture document.

- constraints:
  - Its recorded identity MUST be `sm_89`.
  - Its instruction capabilities MUST NOT name `wgmma.mma_async`,
    `cp.async.bulk`, `cp.async.bulk.tensor`, `mbarrier.expect_tx` or
    `mbarrier.try_wait`: this architecture runs none of them, so its MMA is
    `mma.sync` and its copy from global to shared memory is `cp.async`.
  - Storage and scale DTypes `f4e2m1` and `f8e8m0` MUST NOT be recorded as
    compute DTypes.
  - It MUST record no tensor-memory capacity: `mma.sync` accumulates in the
    register file, so there is no separate store to size.

### 4.2 `CudaDevice`

```python
class CudaDevice(Device):
    """One CUDA device: how many SMs, and the memory and compute rates."""

    name: str
    sm_count: int
    sm_clock_hz: int | None
    hbm_capacity_bytes: int
    gmem_owner: str
    hbm_bandwidth_bytes_per_second: int
    l2_capacity_bytes: int | None
    _dense_flops: tuple[tuple[DType, int], ...]
    _service_ops: tuple[tuple[str, int], ...]
    gmem_latency_ns: int | None = None

    def peak_for(self, dtype: DType) -> int: ...

    @property
    def service_ops_per_second(self) -> dict[str, int]: ...
```

- constraints:
  - One value type MUST answer for every CUDA device, on the same terms as the
    architecture ([§4.1](#41-cudaarchitecture)): a product is its document
    ([§4.2.1](#421-h200sxm), [§4.2.2](#422-b200sxm),
    [§4.2.3](#423-rtx-6000-ada)), not a type.
  - A concrete value MUST be immutable, MUST describe one device, and MUST NOT
    carry a GPU count.
  - It MUST describe how many SMs the product has and how its memory system and
    compute units perform. Per-SM structural limits belong to the architecture
    ([§4.1](#41-cudaarchitecture)).
  - `gmem_latency_ns` MUST be the device's global-memory load latency in ns,
    read from the optional leaf `memory.gmem.latency`. It is a cited figure,
    not a measurement on the described host: H200 records 353 ns from the
    pointer-chase plateau of the same GH100 die (arXiv 2608.15764), with the
    cycle counts of two other studies as cross-checks. A document that omits
    the leaf or records it unavailable states no latency.
  - `sm_clock_hz` MUST be a typed device fact when a per-clock architecture
    rate needs conversion to a per-second service rate. A clock derived from an
    existing peak MUST record that derivation once; other service evidence MUST
    refer to this fact rather than repeat the arithmetic in prose.
  - `peak_for` MUST answer from the installed document for every compute DType
    the product's tensor cores have a mode for, and MUST raise an actionable
    error for any other DType. A product with no such mode records that leaf
    unavailable, so asking for its rate fails instead of reading as a rate
    nobody published.
  - `_dense_flops` MUST hold a dense integer FLOP/s entry per DType. A published
    sparse peak MUST be halved and the division stated as the document's
    evidence, so no plan is priced against a rate structured sparsity is
    required to reach.
  - `_service_ops` MUST hold one integer results/s entry per service kind the
    installed document records, stated per CTA rather than per device. A kind
    the document does not record MUST be absent rather than zero, so a consumer
    that needs it refuses instead of pricing the work at nothing.
  - `l2_capacity_bytes` MUST be `None` when the installed document records no
    value for it. A recorded absence and a number are both statements about the
    product; a substituted figure would not be.
  - `gmem_owner` MUST be `target`: all execution units of this single-device
    Target share one global-memory allocation.
  - The `memory.hbm.*` facts and the `hbm_*` fields describe the device's
    global memory whatever its technology. A product whose global memory is not
    HBM records its figures there and names the technology in `conditions`.
  - No field MAY carry a default, and no resource value MAY be written as a
    Python literal: the installed document is the single source
    ([target §10](../../../../../docs/spec/target.md#10-installed-hardware-resources)). Selecting a different installed
    document by ID is not an override; supplying a partial or edited number
    without a document behind it is, and is not admitted.

#### 4.2.1 H200SXM

The installed `nvidia.h200_sxm` device document.

- constraints:
  - Its recorded identity MUST be `h200_sxm`, and it MUST declare `nvidia.sm90`
    as the architecture it composes with.
  - It MUST record a dense peak for each of `f32`, `f16`, `bf16`, and
    `fp8e4m3`.
  - `memory.gmem.latency` MUST record 353 ns as a `reference` figure with its
    source and cross-checks in `conditions`.
  - `throughput.f4e2m1` MUST be recorded unavailable rather than omitted: the
    Hopper tensor cores have no FP4 mode, so the product has no such rate, which
    is a fact about it and not a number nobody published.
  - It MUST record a per-CTA service rate for each of `integer`, `predicate`,
    `select`, and `special`, because a program that compares, selects or
    indexes asks for work no FLOP/s figure prices. Each MUST state its derivation in `conditions`: the instruction
    throughput in results/clock/SM from the vendor's arithmetic-instruction
    table, times `compute.sm_clock_hz`, stated per CTA. That clock MUST be the
    floor of the value the published `f32` peak implies
    (`67e12 / (132 SM * 128 results/clock * 2 FLOP/result)`).
    These are peak-style analytical envelopes rather than measured
    calibrations, and `conditions` MUST say so, along with the proxy each one
    stands for. A service rate MUST NOT stand in for a bandwidth: movement at a
    level with no published one is stated as traffic and left untimed.

#### 4.2.2 B200SXM

The installed `nvidia.b200_sxm` device document.

- constraints:
  - Its recorded identity MUST be `b200_sxm`, and it MUST declare `nvidia.sm100`
    as the architecture it composes with.
  - It MUST record a dense peak for each of `f32`, `f16`, `bf16`, `fp8e4m3`, and
    `f4e2m1`.
  - `memory.gmem.latency` MUST be recorded unavailable: no cited figure is
    recorded for this product, so its global-memory rate has no per-CTA cap.
  - Resource facts the vendor does not publish for this product, its SM count
    and its L2 capacity among them, MUST be recorded as measured on the
    described host rather than estimated or borrowed from a related part
    ([target §10.1](../../../../../docs/spec/target.md#101-document-envelope)).
  - `compute.sm_clock_hz` MUST be the floor of
    `75e12 / (148 SM * 128 results/clock * 2 FLOP/result)`, using the dense
    `throughput.f32` and `compute.sm_count` already recorded by the document.

#### 4.2.3 RTX 6000 Ada

The installed `nvidia.rtx_6000_ada` device document.

- constraints:
  - Its recorded identity MUST be `rtx_6000_ada`, and it MUST declare
    `nvidia.sm89` as the architecture it composes with.
  - It MUST record a dense peak for each of `f32`, `f16`, `bf16`, and
    `fp8e4m3`, and record `throughput.f4e2m1` unavailable: the Ada tensor cores
    have no FP4 mode.
  - `memory.hbm.*` MUST record its GDDR6 capacity and bandwidth and name GDDR6
    in `conditions` ([§4.2](#42-cudadevice)).
  - `memory.gmem.latency` MUST be recorded unavailable: no cited figure is
    recorded for this product, so its global-memory rate has no per-CTA cap.
  - `compute.sm_clock_hz` MUST be the floor of
    `91.1e12 / (142 SM * 128 results/clock * 2 FLOP/result)`, and its service
    rates MUST follow [§4.2.1](#421-h200sxm) with the compute capability 8.9
    column of the vendor's arithmetic-instruction table.
