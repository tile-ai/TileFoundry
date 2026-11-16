# TileFoundry CUDA Spec — Target

## 4. `cuda.Target`

```python
class Target(target.Target):
    """CUDA target composed from one architecture and one device."""

    name: ClassVar[str] = "cuda"
    architecture: Architecture
    device: Device
    device_count: int | None
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
        device_count: int | None = None,
    ) -> None: ...

    @classmethod
    def from_installed(cls) -> tuple[Target, ...]: ...
    @property
    def identity(self) -> str: ...
    def validate_program_topology(self, topology: Topology) -> None: ...
    def topology_limit(self, name: str) -> int: ...
    def get_analyzer(self, selector: str) -> Analyzer: ...
    def torch_device_type(self) -> str: ...
    def get_facts(self, facts_type: type[FactsT], query=None) -> FactsT: ...
    def __repr__(self) -> str: ...
```

- constraints:
  - `device` and `architecture` MUST each accept an installed document ID or a
    concrete value. An ID MUST resolve immediately to the typed value. The
    target MUST retain the resolved ID and content digest ([target
    §10.2](../../../../../../docs/spec/target.md#102-registry-and-resolution)).
  - `arch`, when provided, MUST equal the resolved architecture's `name`. It is
    a consistency check. It MUST NOT select or override an architecture.
  - `device` MUST be required. The constructor MUST NOT select hardware for a
    caller who named none. A target with no stated hardware would answer about a
    machine nobody has.
  - An omitted `architecture` MUST be read from the device document's declared
    compatibility. The read MUST fail unless that document names exactly one. A
    `Device` supplied directly carries no document, so it MUST be given an
    architecture as well.
  - `arch` MUST equal `architecture.name`.
  - A pair selected by ID MUST be checked for declared compatibility. A value
    supplied directly carries no document. It has no ID or digest. It is exempt
    from that check. It is a distinct hardware value, not a revision of an
    installed one.
  - `cuda.Target` MUST compose any installed architecture and device pair that
    declares compatibility. A further CUDA product is its two documents ([target
    §10.2](../../../../../../docs/spec/target.md#102-registry-and-resolution))
    and nothing else. It MUST NOT add a Target subclass, or an architecture or
    device type of its own. The value already selects the services and Facts.
    The documents hold the numbers. Either addition would carry nothing.
  - `cuda.Target.from_installed()` MUST contain one value per device document
    whose sole compatible architecture document is available. Its `identity`
    MUST be that device document's ID.
  - CUDA MUST select its standard Analyzer and Facts through the corresponding
    getters. These selections MUST NOT use `Target.name`, an exact-concrete-type
    table, or a second extension registration.
  - `device_count`, when given, MUST be a positive `int`. `None` admits any
    extent at the `gpu` level. `topology_limit("gpu")` is `device_count`, or 1
    when it is `None`.
  - `get_analyzer("sync")` returns the `sync` family ([analysis
    §1](./analysis.md#1-sync)). That family requires `memory`. Every other
    selector is the base Target's.
  - CUDA's `ThroughputFacts` and `PerformanceServiceFacts` MUST both read
    `rates(target, unit, active)`. It is the one source of CUDA rates. It gives
    what `active` units of `unit` get through together. A query is a level name
    or `(level, active, inflight)`. `active` counts only at `cta`. There it MUST
    be capped at the units the deployment holds. Any other level runs all of its
    units. A bare level runs all of its units. `inflight` is the bytes one CTA
    keeps in flight for the access being priced. `ThroughputFacts` is `rates`
    itself. It covers the whole deployment when asked about no level.
    `PerformanceServiceFacts` is `rates` divided by `active`, rounding down.
    CUDA MUST name its default unit `cta`. `TopologyLimits.max_physical_units`
    MUST NOT enter this derivation.
  - `rates` MUST scale each deployment dense-FLOP rate by `active` over the
    units the deployment holds. It MUST scale each service rate by the same
    factor. A service rate is `unit_ops` from the device document, per SM. It
    MUST scale the shared-memory bandwidth by the same factor. That bandwidth is
    the architecture's byte/clock/SM rate times the device's typed SM clock.
    Global-memory bandwidth MUST be the HBM peak. At `cta` on a device stating
    `gmem_latency_ns`, it MUST also be no more than
    `active * inflight / gmem_latency_ns`. By Little's law, one CTA with
    `inflight` bytes outstanding moves at most that many bytes per latency. One
    1024-thread CTA with 16 B per thread in flight moves about 46 GB/s on H200.
    About 103 CTAs reach its 4.8 TB/s. A device without the fact has no cap. A
    query without `inflight` has no cap. `unit_ops` is not a division of a
    device peak. No vendor publishes a device-wide integer, predicate, select,
    special or local-move rate to divide.
  - `topology_limit("cta")` MUST equal `device.sm_count`.
    `topology_limit("thread")` MUST equal the architecture's corresponding
    structural limit. The target MUST refuse an unsupported name.
  - `__repr__` MUST use the concrete class name. It MUST return a constructor
    expression that rebuilds an equal Target. A subclass that keeps the CUDA
    constructor shape inherits it. A subclass with another constructor MUST
    override `__repr__`.
  - The target MUST project the store the threads of one CTA cooperate in as
    `architecture.shared_memory_per_cta_bytes`. It MUST report that store as
    belonging to the `cta` scope, even when the level being asked about is
    `thread`.
  - The target MUST project the tensor-memory level with
    `architecture.tensor_memory_per_cta_bytes` only where the architecture
    states a capacity. `explicit_levels` MUST NOT contain that level where the
    architecture states `None`. A level for hardware with no such store would
    offer a plan accumulator space that does not exist.

#### Topology levels

A target's topology levels define the names a program may declare. The program hierarchy
stops at those levels. Thread mesh layouts hold warp, lane, and warpgroup structure.

- constraints:
  - `get_facts(TopologyFacts).topologies` MUST name `("gpu", "cta", "thread")`,
    outermost first. A program names `gpu` when its data is divided across
    cards.
  - A topology level whose `from_target` is set takes its extent and its unit
    ids from the target instance. It does not take them from the hardware
    document or a register. `"gpu"` MUST be the only CUDA level that sets it.
  - A declared program topology name MUST be one of the levels its target
    states. A name outside that set MUST be refused. The refusal MUST name the
    levels the target declares.
  - `get_facts(TopologyLimits, "cta").max_logical_units` MUST be `None`. The
    CUDA grid is a launch shape, not an SM allocation, so its static extent is
    unbounded here. The `"thread"` Facts projection MUST equal
    `architecture.max_threads_per_cta`.
  - `Topology.size` MUST be an explicit `ShapeDim`. Construction with `None`
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
`cp.async.bulk`, `cp.async.bulk.tensor`, `mbarrier.expect_tx` and `mbarrier.try_wait` as
capabilities. Admission MUST refuse an instruction whose declared capability the
target's architecture does not list. The refusal MUST name the capability and the
target. Where an attribute selects one of several declarations (an MMA atom), only the
selected declaration's capability applies. A target without an architecture lists none.
Admission without a target checks nothing. The resource-bearing declarations map the two
MMA families, matrix load/store, and tensor-map copy respectively to
`tensor_core_engine`, `wgmma_engine`, `tensor_core_engine`, and `tma_engine`. The
current model gives `cp.async` no separately priced engine. The capability and resource
vocabularies MUST be disjoint. Architecture documents list only capabilities under
`[facts.capabilities]`. An instruction declaration is the single source of any resource
mapping it states. A resource declaration does not imply a throughput fact. A resource
declaration MUST NOT be priced as a service unless a consumer defines a quantity with
matching units.

### 4.1 `cuda.Architecture`

```python
class Architecture(target.Architecture):
    """What one CUDA architecture states about itself."""

    name: str
    gpu_architecture: str
    gpu_code: tuple[str, ...]
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
  - One value type MUST answer for every CUDA architecture. A concrete value
    MUST be immutable. The installed document records what separates one
    architecture from another ([§4.1.1](#411-sm90), [§4.1.2](#412-sm100),
    [§4.1.3](#413-sm89)). An architecture MUST NOT be given a type of its own. A
    class holding no number would restate an identity the value already carries.
  - `name` MUST be the architecture identity. `gpu_architecture` and `gpu_code`
    MUST be the values that CUDA compilation passes verbatim to nvcc
    `--gpu-architecture` and `--gpu-code`. They are recorded apart from `name`,
    because an architecture may need a feature set its identity does not name.
    The nvcc single-option shorthand for that feature set also emits the
    portable PTX. That PTX cannot hold its instructions.
  - A CUDA architecture MUST own supported compute DTypes, instruction
    capabilities, and the thread/CTA structural limits. `capabilities` MUST use
    instruction-family names from the emitted ISA, such as `mma.sync`,
    `wgmma.mma_async`, `ldmatrix`, `stmatrix`, `cp.async`, and
    `cp.async.bulk.tensor`.
  - It MUST own the per-SM resource limits. They are resident CTAs,
    shared-memory capacity per SM and per CTA, and register-file capacity per
    SM. They are properties of the microarchitecture. Every product built on it
    shares them. A device MUST NOT restate them.
  - A shared-memory bandwidth stated in byte/clock/SM MUST remain an
    architecture fact. Turning it into byte/s requires the paired device's SM
    clock. That conversion happens only in the CUDA `rates` projection.
  - It MUST NOT carry a compute-throughput rate. A FLOP/s figure depends on the
    clock of one product. The figure is a device fact ([§4.2](#42-cudadevice)),
    even though the instruction it rates is the architecture's.
  - `unified_l1_shared_per_sm_bytes` MUST be the size of the one physical block.
    The shared-memory carveout and the L1 data cache are both taken from that
    block. It MUST be at least `shared_memory_per_sm_bytes`. The architecture
    MUST NOT state an L1 capacity. How much L1 remains depends on how much
    shared memory a program asked for. That amount is not a property of the
    hardware.
  - `tensor_memory_per_cta_bytes` MUST be the whole tensor-memory store on an
    architecture whose MMA accumulates in one. That store is allocated in
    columns that span every lane. One CTA may hold all of them. It MUST be
    `None` where the architecture has no such store. The document makes the same
    statement by recording that leaf unavailable.
  - CUDA explicit memory ownership MUST be read from the installed hardware
    values as `smem -> cta`, `rmem -> thread`, and `tmem -> cta`. `scope` MUST
    NOT be used as an ownership map. Register capacity is stated per SM. Each
    thread owns its register values.
  - Every architecture document MUST declare the leaf behind every field. A
    value the architecture does not have is recorded as absent, not left out
    ([target
    §10.2](../../../../../../docs/spec/target.md#102-registry-and-resolution)).
  - No field MAY carry a default. Every value comes from the installed document
    ([target
    §10](../../../../../../docs/spec/target.md#10-installed-hardware-resources)).
    The type declares shape and never content.

#### 4.1.1 SM90

The installed `nvidia.sm90` architecture document.

- constraints:
  - Its recorded identity MUST be `sm_90`. It MUST compile with
    `gpu_architecture` `compute_90a` and `gpu_code` `sm_90a`, `compute_90a`.
    Wgmma and setmaxnreg exist only in the `sm_90a` feature set.
  - Storage and scale DTypes `f4e2m1` and `f8e8m0` MUST NOT be recorded as
    compute DTypes.
  - It MUST record no tensor-memory capacity. The SM90 MMA accumulates in the
    register file. There is no separate store to size.

#### 4.1.2 SM100

The installed `nvidia.sm100` architecture document.

- constraints:
  - Its recorded identity MUST be `sm_100`. It MUST compile with
    `gpu_architecture` `compute_100` and `gpu_code` `sm_100`, `compute_100`.
  - It MUST record `f4e2m1` as a compute DType. The SM100 MMA takes 4-bit
    operands directly. `f8e8m0` MUST NOT be recorded as one. It scales those
    operands rather than being multiplied.
  - It MUST record a tensor-memory capacity. Its instruction capabilities MUST
    name the MMA family that accumulates there, not the SM90 family. This
    architecture does not run the SM90 family.

#### 4.1.3 SM89

The installed `nvidia.sm89` architecture document.

- constraints:
  - Its recorded identity MUST be `sm_89`. It MUST compile with
    `gpu_architecture` `compute_89` and `gpu_code` `sm_89`, `compute_89`.
  - Its instruction capabilities MUST NOT name `wgmma.mma_async`,
    `cp.async.bulk`, `cp.async.bulk.tensor`, `mbarrier.expect_tx` or
    `mbarrier.try_wait`. This architecture runs none of them. Its MMA is
    `mma.sync`. Its copy from global to shared memory is `cp.async`.
  - Storage and scale DTypes `f4e2m1` and `f8e8m0` MUST NOT be recorded as
    compute DTypes.
  - It MUST record no tensor-memory capacity. `mma.sync` accumulates in the
    register file. There is no separate store to size.

### 4.2 `cuda.Device`

```python
class Device(target.Device):
    """One CUDA device: how many SMs, and the memory and compute rates."""

    name: str
    sm_count: int
    sm_clock_hz: int | None
    hbm_capacity_bytes: int
    gmem_owner: str
    hbm_bandwidth_bytes_per_second: int
    l2_capacity_bytes: int | None
    dense_flops: tuple[tuple[DType, int], ...]
    service_ops: tuple[tuple[str, int], ...]
    gmem_latency_ns: int | None = None

    @property
    def dense_flops_per_second(self) -> dict[DType, int]: ...

    def peak_for(self, dtype: DType) -> int: ...

    @property
    def service_ops_per_second(self) -> dict[str, int]: ...
```

- constraints:
  - One value type MUST answer for every CUDA device, on the same terms as the
    architecture ([§4.1](#41-cudaarchitecture)). A product is its document
    ([§4.2.1](#421-h200sxm), [§4.2.2](#422-b200sxm),
    [§4.2.3](#423-rtx-6000-ada)), not a type.
  - A concrete value MUST be immutable, MUST describe one device, and MUST NOT
    carry a GPU count.
  - It MUST describe how many SMs the product has and how its memory system and
    compute units perform. Per-SM structural limits belong to the architecture
    ([§4.1](#41-cudaarchitecture)).
  - `gmem_latency_ns` MUST be the device's global-memory load latency in ns,
    read from the optional leaf `memory.gmem.latency`. It is a cited figure, not
    a measurement on the described host. H200 records 353 ns from the
    pointer-chase plateau of the same GH100 die (arXiv 2608.15764). The cycle
    counts of two other studies serve as cross-checks. A document that omits the
    leaf or records it unavailable states no latency.
  - `sm_clock_hz` MUST be a typed device fact when a per-clock architecture rate
    needs conversion to a per-second service rate. A clock derived from an
    existing peak MUST record that derivation once. Other service evidence MUST
    refer to this fact. It MUST NOT repeat the arithmetic in prose.
  - `peak_for` MUST answer from the installed document for every compute DType
    the product's tensor cores have a mode for. It MUST raise an actionable
    error for any other DType. A product with no such mode records that leaf
    unavailable. Asking for its rate then fails. It does not read as a rate
    nobody published.
  - `dense_flops` MUST hold a dense integer FLOP/s entry per DType.
    `dense_flops_per_second` returns it as a dict. A published sparse peak MUST
    be halved. The document's evidence MUST state the division. No plan is then
    priced against a rate that needs structured sparsity.
  - `service_ops` MUST hold one integer results/s entry per service kind that
    the installed document records. Each entry MUST be stated per CTA rather
    than per device. A kind the document does not record MUST be absent rather
    than zero. A consumer that needs it then refuses. It does not price the work
    at nothing.
  - `l2_capacity_bytes` MUST be `None` when the installed document records no
    value for it. A recorded absence and a number are both statements about the
    product. A substituted figure would not be.
  - `gmem_owner` MUST be `gpu`. Each card's global memory belongs to that card.
    A `gpu`-level `Split` on gmem is therefore private. Moving it between cards
    is communication. The levels below `gpu` share the card's allocation.
  - The `memory.hbm.*` facts and the `hbm_*` fields describe the device's global
    memory, whatever its technology. A product whose global memory is not HBM
    records its figures there. It names the technology in `conditions`.
  - No field MAY carry a default. No resource value MAY be written as a Python
    literal. The installed document is the single source ([target
    §10](../../../../../../docs/spec/target.md#10-installed-hardware-resources)).
    Selecting a different installed document by ID is not an override. Supplying
    a partial or edited number without a document behind it is an override. It
    is not admitted.

#### 4.2.1 H200SXM

The installed `nvidia.h200_sxm` device document.

- constraints:
  - Its recorded identity MUST be `h200_sxm`, and it MUST declare `nvidia.sm90`
    as the architecture it composes with.
  - It MUST record a dense peak for each of `f32`, `f16`, `bf16`, and `fp8e4m3`.
  - `memory.gmem.latency` MUST record 353 ns as a `reference` figure with its
    source and cross-checks in `conditions`.
  - `throughput.f4e2m1` MUST be recorded unavailable rather than omitted. The
    Hopper tensor cores have no FP4 mode. The product has no such rate. That is
    a fact about it, not a number nobody published.
  - It MUST record a per-CTA service rate for each of `integer`, `predicate`,
    `select`, and `special`. A program that compares, selects or indexes asks
    for work that no FLOP/s figure prices. Each rate MUST state its derivation
    in `conditions`. The derivation is the instruction throughput in
    results/clock/SM from the vendor's arithmetic-instruction table, times
    `compute.sm_clock_hz`, stated per CTA. That clock MUST be the floor of the
    value the published `f32` peak implies
    (`67e12 / (132 SM * 128 results/clock * 2 FLOP/result)`). These are
    peak-style analytical envelopes rather than measured calibrations.
    `conditions` MUST say so, along with the proxy each one stands for. A
    service rate MUST NOT stand in for a bandwidth. Movement at a level with no
    published bandwidth is stated as traffic and left untimed.

#### 4.2.2 B200SXM

The installed `nvidia.b200_sxm` device document.

- constraints:
  - Its recorded identity MUST be `b200_sxm`, and it MUST declare `nvidia.sm100`
    as the architecture it composes with.
  - It MUST record a dense peak for each of `f32`, `f16`, `bf16`, `fp8e4m3`, and
    `f4e2m1`.
  - `memory.gmem.latency` MUST be recorded unavailable. No cited figure is
    recorded for this product. Its global-memory rate has no per-CTA cap.
  - The vendor does not publish some resource facts for this product. Each such
    fact MUST be recorded as measured on the described host ([target
    §10.1](../../../../../../docs/spec/target.md#101-document-envelope)). Its SM
    count and its L2 capacity are among them. Such a fact MUST NOT be estimated
    or borrowed from a related part.
  - `compute.sm_clock_hz` MUST be the floor of
    `75e12 / (148 SM * 128 results/clock * 2 FLOP/result)`, using the dense
    `throughput.f32` and `compute.sm_count` already recorded by the document.

#### 4.2.3 RTX 6000 Ada

The installed `nvidia.rtx_6000_ada` device document.

- constraints:
  - Its recorded identity MUST be `rtx_6000_ada`, and it MUST declare
    `nvidia.sm89` as the architecture it composes with.
  - It MUST record a dense peak for each of `f32`, `f16`, `bf16`, and `fp8e4m3`.
    It MUST record `throughput.f4e2m1` unavailable. The Ada tensor cores have no
    FP4 mode.
  - `memory.hbm.*` MUST record its GDDR6 capacity and bandwidth and name GDDR6
    in `conditions` ([§4.2](#42-cudadevice)).
  - `memory.gmem.latency` MUST be recorded unavailable. No cited figure is
    recorded for this product. Its global-memory rate has no per-CTA cap.
  - `compute.sm_clock_hz` MUST be the floor of
    `91.1e12 / (142 SM * 128 results/clock * 2 FLOP/result)`. Its service rates
    MUST follow [§4.2.1](#421-h200sxm) with the compute capability 8.9 column of
    the vendor's arithmetic-instruction table.
