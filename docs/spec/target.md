# TileFoundry Spec — Target

A `Target` is the immutable capability context compilation and the compiler
algorithms read. Architecture describes compilation identity and instruction
structure. Device describes fixed product resources. A target is a value that
answers questions about hardware; it does not own the operations that ask, and it
answers only by projecting the facts an asking algorithm declared.

## 1. `Target`

```python
TargetT = TypeVar("TargetT", bound="Target")
FactsT = TypeVar("FactsT")
T = TypeVar("T")


class Target:
    """Identify a compilation backend."""

    name: ClassVar[str]

    @property
    def identity(self) -> str: ...

    @classmethod
    def available(cls) -> tuple[Target, ...]: ...

    def get_analyzer(self, selector: str) -> Analyzer: ...
    def resolve(self, base: type[T], *args) -> T: ...
    def torch_device_type(self) -> str: ...
    def validate_program_topology(self, topology: Topology) -> None: ...
    def get_facts(
        self,
        facts_type: type[FactsT],
        query: object | None = None,
    ) -> FactsT: ...


def register_target(cls: type[TargetT]) -> type[TargetT]: ...


def registered_targets() -> Mapping[str, type[Target]]: ...


def bind(target: type[Target], base: type) -> Callable[[type[T]], type[T]]: ...
```

- constraints:
  - A custom Target MUST use one of two modes. A document-backed product adds
    complete Architecture and Device documents to an existing Target class; it
    MUST NOT add a product-specific Target subclass. A new backend implements
    the Target interface directly and answers its hardware through `get_facts`;
    it need not expose Architecture, Device, or hardware documents.
  - Inheriting bare Architecture or Device values and injecting them into an
    existing document-backed Target is not a supported custom Target mode. It
    bypasses both document schema validation and the `facts_result` projection
    boundary.
  - `bind(target, base)` MUST register a subclass implementation under the exact
    `(Target class, base class)` pair. A second binding for the pair MUST raise
    `ValueError` naming the Target, base, and existing implementation.
  - `resolve(base, *args)` MUST call `create(self, *args)` on the implementation
    bound to `(type(self), base)`, or on `base` when unbound. Bindings MUST live
    at implementation class definitions; the Target layer MUST NOT import
    codegen implementations. Target subclasses do not inherit another class's
    bindings implicitly.
  - `Target.get_analyzer` MUST select the standard compute-cost, memory,
    roofline, and performance analyzers for every Target. Those algorithms consume
    only requested Facts, so a backend reuses them by answering `get_facts`, not
    by inheriting a backend-specific analysis base class. A missing Facts
    projection MUST fail when the selected analyzer requests it.
  - `facts_result`, `TargetFactsError`, `TopologyFacts`, `TopologyLevelFacts`,
    `MemoryExecutionFacts`,
    `MemoryHierarchyFacts`, `ThroughputFacts`, and `PerformanceServiceFacts`
    MUST be importable from `tilefoundry.target` for provider implementations.
  - `name` MUST be a non-empty class variable declared directly by every
    concrete registered Target class. It is the stable class registration
    identity, not a backend-family selector and not an instance value field.
  - `@register_target` MUST be the only Target registration form. It takes the
    class directly, accepts no decorator arguments, never constructs the class,
    and returns it unchanged.
  - Re-registering a class with the same module, qualified name, and registered
    name MUST be idempotent, including after a provider reload. A different
    provider claiming the same name MUST fail rather than replace the owner.
  - `registered_targets()` MUST expose one read-only `name -> class` view. The
    view MAY be used for inspection but MUST NOT construct a Target.
  - `identity` MUST name one concrete Target value. It MUST be the device
    document ID for a document-backed Target and the registered class `name`
    for a Target with no device document.
  - `available()` MUST return every value of that registered class which can be
    constructed in the current environment. A class with no hardware documents
    MUST return its one parameterless value by default.
  - Authored Target parameters MUST accept a constructed Target instance or
    their documented omitted state. A string MUST fail and MUST NOT be resolved
    through registration.
  - Target values MUST remain immutable hardware values. Their service getters
    select immutable descriptors and Facts; normal Python inheritance carries
    those selections to a subclass unless it overrides or refuses them.
  - `get_facts(TopologyFacts)` MUST list every topology level the Target
    admits, coarsest first, as the `TopologyLevelFacts` of each. A Target that
    states none MUST answer the empty tuple: it runs one program per call.
  - `validate_program_topology` MUST reject a level outside
    `get_facts(TopologyFacts)` and any resolved static extent that is not
    positive or exceeds that level's finite `max_logical_units`. The shared
    program check MUST use this method rather than reproduce a backend's
    topology limits.
  - A provider MAY import `Analyzer` from `tilefoundry.target` to construct
    getter results. Codegen base classes MUST be imported from their owning
    `codegen` modules.
  - `torch_device_type()` MUST return the torch device type holding the
    program's tensor arguments. The base implementation returns `"cpu"`,
    inherited by CPU. A backend MUST override it for another device type. CLI input generation MUST
    use this answer and reject an unavailable non-CPU device.
  - A missing getter capability MUST fail and name the concrete Target class,
    its registration name, and the requested selector, topology, or Facts type.
  - Target values MUST NOT own code emission, linking, loading, or the public
    Analyze, compile, build, or jit orchestration.

### 1.1 `Architecture`

```python
class Architecture:
    """Describe compilation architecture identity and structural facts."""

    name: str
    max_threads_per_cta: int
```

- constraints:
  - `Architecture` is a Target-level marker for the architecture-side value
    passed into a Facts projection and for its Python reconstruction support;
    it is not a provider extension point.
  - A document-backed Target MUST accept direct values only through its
    backend-specific concrete Architecture type. A further product adds a
    document of that backend's schema; a further backend implements `Target`
    and `get_facts` directly.
  - Concrete backend architecture values MUST be immutable.
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
  - `Device` is a Target-level marker for the device-side value passed into a
    Facts projection and for its Python reconstruction support; it is not a
    provider extension point.
  - A document-backed Target MUST accept direct values only through its
    backend-specific concrete Device type. A further product adds a document
    of that backend's schema; a further backend implements `Target` and
    `get_facts` directly.
  - Concrete backend device values MUST be immutable and describe one device.
  - `name` MUST be the stable product identity.
  - Device-specific capacity, bandwidth, and compute-throughput facts belong
    to concrete subclasses.
  - The split between the two is by what the fact is a property of, not by
    which consumer reads it. An Architecture owns instruction legality and the
    per-parallel-unit structural limits, which every product built on it
    shares. A Device owns how many such units the product has, its memory
    system, and its measured or published throughput. A fact MUST be recorded on
    exactly one side, and the other side MUST NOT restate it.

### 1.3 Target loading and extensions

```python
class TargetLoadError(RuntimeError): ...


def load_targets() -> Mapping[str, type[Target]]: ...
```

- constraints:
  - `load_targets()` MUST load `tilefoundry.targets` entry points in name order.
    Each `<name>` MUST point to `tilefoundry_<vendor>.<name>`; importing it MUST
    complete target, operation, visitor, evaluator and pass registrations.
  - A backend MUST ship as `tilefoundry-<vendor>` with only the top-level package
    `tilefoundry_<vendor>` and depend on `tilefoundry==V` at its own version `V`.
    `load_targets()` MUST reject other top-level packages.
  - A load failure MUST raise `TargetLoadError` naming the entry point or
    distribution and cause. A failed load MUST allow retry; a successful load
    MUST make later calls no-ops.
  - Core MUST NOT import or re-export plugin implementations. CPU MUST remain
    available from `tilefoundry.target`; other backends MUST export target values
    from `tilefoundry_<vendor>`.
  - Importing `tilefoundry` MUST call `load_targets()` once at the end of core
    initialization. CLI, DSL lookup, stub generation and spec discovery MUST
    use the registrations established by that import, without loading again.
    A plugin package already present in `sys.modules` MUST be skipped: when a
    user imports a plugin first, that partially initialized package completes
    its own registrations after core initialization returns.
  - A plugin MUST install its specifications as package data under `docs/spec/`.
    After loading targets, `tilefoundry spec` MUST discover those files through
    `importlib.resources.files(entry_point.value)` in source and installed modes,
    naming each topic `<entry_point.name>/<filename stem>`.
  - `EmitContext`, `LinkableModule`, `LinkedFunction` and `LinkedModule` MAY be
    subclassed and bound with `@bind(TargetClass, BaseClass)`. The implementation's
    `create(target, *args)` MUST provide the instance selected by `resolve`.
    `ConvertHIRToTIR` MAY be bound the same way; its `create` takes the Target and
    the `CodegenOptions` ([passes §7.1](./passes.md#71-converthirtotir)).
    `LinkableModule.link(project)` MUST contribute source and build settings to
    the shared `CMakeProject`; core MUST build the combined project once.
    `LinkableFunction` and whole-program `CodegenContext` MAY be subclassed, but
    MUST NOT be constructed through `Target.resolve` by core orchestration.

## 5. `CpuTarget`

```python
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name: ClassVar[str] = "cpu"
```

- constraints:
  - `name` MUST be `"cpu"`.
  - CPU host Functions MAY coexist with device Functions in one module and are
    exempt from device hardware-fact equality checks.

## 6. Target ownership and compile resolution

- `tilefoundry.target` MUST own the Target abstraction and CPU implementation;
  plugins MUST own their concrete targets. The IR package MUST NOT own Target
  classes.
- `default_target()` MUST return a new `CpuTarget()`. It is the omitted-target
  policy for `PrimFunction` / `@prim_func` and `jit()`, not a string lookup.
- There MUST be no global string-to-Target resolver. A registered name returns
  class identity only through `registered_targets()` and never constructs a
  value.
- A `Target` belongs to a `Module` rather than an authored HIR `Function`.
  Target inheritance and its declaration rules are defined by
  [core-ir `target-inheritance`](./core-ir.md#target-inheritance).
- Analyze MUST obtain the Target from `Module.resolve_target()` and from
  nowhere else. It does not accept a bare `Function`, and it does not resolve
  an undeclared Target to a default: it reports hardware-dependent results, so
  measuring against a device the author never declared is a silent wrong
  answer. In particular it does not read a Target out of `Module.metadata`; the
  `metadata["target"]` the compile pipeline carries is the codegen boundary's
  own record ([passes §6](./passes.md#6-top-level-api)), not a Target source
  for Analyze.
- The compile boundary resolves no default: `build` reports a Module that
  declares no Target ([passes §6](./passes.md#6-top-level-api)) rather than
  compiling it against a guess. `default_target()` is the omitted-target policy
  of the authoring surface — the value a `PrimFunction` constructed without one
  takes — not a Target the compile supplies.
- A TIR `PrimFunction` retains its own `target`: it MUST be the exact Target
  instance resolved from its Module, and no later stage rewrites it. It selects
  the bound emission classes that emit it. A synthesized host entry carries a
  `CpuTarget()`.
- Device Functions MUST be grouped by equal Target values in source order.
  More than one unequal device Target group MUST fail before any generator runs.

## 10. Installed hardware resources

Architecture and Device documents are the canonical authored hardware
database, and the only place a hardware number is written. Each is a complete
document in its own right; a target is the pair composed through a declared
compatibility, never a single combined record.

Each document belongs to the Target class that understands its versioned
schema. The class declares one `HardwareSpec` containing its immutable schema
builders and the package holding its built-in documents. There is no
cross-backend hardware registry.

### 10.1 Document envelope

```toml
[spec]
schema = "example.device/v1"
kind = "device"
id = "example.device"

[compatibility]
architectures = ["example.architecture"]

[facts.memory.main.bandwidth]
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
  - Compiler policy and a program's Topology extents MUST NOT appear in a
    hardware document. They are compiler inputs, not immutable hardware truth:
    a fixed-wave parallel capacity is a policy even when its current value
    equals a device count. An explicit memory level's `owner` is
    different: it MUST name one topology from the Target's hardware vocabulary,
    or the reserved word `target` for an allocation shared by the whole device.
    The document states that ownership directly and MUST NOT encode it as an
    absent value.
  - Only explicit memory levels MUST record an owner. Implicit caches receive no
    program residency and MUST NOT add an owner leaf merely to make their
    capacity scope look like a Target topology.

### 10.2 Registry and resolution

- constraints:
  - A Target class that consumes hardware documents MUST declare one
    `HardwareSpec`. Its package MUST contain that Target's built-in documents,
    and its schema mapping MUST state every document format the Target accepts.
  - A `HardwareSpec` MUST resolve its documents by exact ID. There MUST be no
    cross-backend document table, search path, overlay, or partial document.
    Built-in package documents MAY be scanned lazily, but resolution MUST NOT
    depend on a previously constructed Target instance.
  - A schema name carries a version. Requiring a leaf a previous version did not
    MUST take a new version, because a document written against the old one no
    longer loads and the failure is otherwise a missing fact rather than a
    contract that moved.
  - A Target's schema mapping is fixed by its class. A complete document MAY be
    adopted as another product only when its schema is in that mapping; adding
    another schema is adding a backend, not adding a hardware product.
  - Explicitly registered documents MUST be routed to the one registered Target
    class that declares their schema in its own `HardwareSpec`. An inherited
    reference to another Target's `HardwareSpec` MUST NOT claim that schema.
    Removing such a document MUST discard its resolved cache entry as well as
    its ID, without changing the Target's built-in package data.
  - An adopted device document MUST NOT become available until its declared
    architecture document is present in the same `HardwareSpec`. Explicit
    registration MUST reject that incomplete pair and name the architecture
    document required first.
  - Device document IDs and every provider Target identity share one uniqueness
    boundary. A collision MUST fail before persistent registration and name the
    existing Target value; it MUST NOT choose one source by load order.
  - A schema MAY validate the documents of several products when they state the
    same fact paths, and MUST build one value type from all of them. A product is
    what its document records, so a schema MUST NOT select a type by the identity
    a document declares.
  - A typed schema MUST validate exact fact paths, value types, units, required
    fields, and cross-field invariants, and MUST reject any leaf the document
    carries that the schema does not model, so a misspelled key cannot become
    an unused fact.
  - Units MUST be normalized while constructing the typed value: algorithms see
    canonical integers such as bytes and bytes per second, never source strings
    or unit conversion.
  - A schema MAY model a leaf as optional, which yields the recorded number or
    `None` for a leaf recorded unavailable. The leaf MUST still be declared: a
    document says either what the value is or that there is none, and a missing
    key MUST remain an error rather than becoming an absent value.
  - Resolution MUST retain each document's ID and content digest on the
    composed value, so a compiled artifact can name the exact resources it was
    built against. Editing any recorded value or its evidence MUST change the
    digest.
  - A Target constructor MUST accept a custom document through a filesystem
    path. The document MUST be complete, retain its declared ID and digest, and
    MUST NOT enter that Target's available-ID namespace, so it can neither
    shadow nor replace an available resource.
  - Unknown IDs, unsupported schemas, unmodelled or malformed facts, malformed
    envelopes, duplicate document IDs, and incompatible pairs MUST each raise
    their own actionable diagnostic rather than one shared parse failure.
  - Reporting the resources behind a target MUST name both documents and their
    digests. A target composed from a directly supplied value has no document to
    report and MUST say so rather than name the installed resource it resembles.

## 11. Target Facts projection

A target-aware algorithm declares the immutable aggregate of facts it needs;
the concrete Target's `get_facts` method builds that aggregate. This is the one
boundary between a hardware specification and an algorithm's own view of it.

```python
class Target:
    def get_facts(
        self,
        facts_type: type[FactsT],
        query: object | None = None,
    ) -> FactsT: ...


class TopologyLevelFacts:
    """The program-side and machine-side unit counts for one topology level.

    Attributes:
        name: Topology level being stated.
        max_logical_units: Maximum units a program may declare, or None.
        max_physical_units: Machine positions the level divides over, or None.
        from_target: Whether the target instance supplies extent and program ids.
    """

    name: str
    max_logical_units: int | None
    max_physical_units: int | None
    from_target: bool = False


class TopologyFacts:
    topologies: tuple[TopologyLevelFacts, ...]
    parallel_level: str | None = None

    def level(self, name: str | None) -> TopologyLevelFacts | None: ...
    def parallel(self) -> TopologyLevelFacts | None: ...


class MemoryExecutionFacts:
    topologies: tuple[str, ...]
```

- constraints:
  - `MemoryExecutionFacts.topologies` MUST list the Target's execution levels
    admitted for synchronous Copy and Fill. Each name MUST occur in
    `TopologyFacts.topologies`. An empty list admits no memory execution scope.
    Shared instruction matching MUST specialize the memory mesh wildcard from
    this projection before checking operands.
  - `max_logical_units` is the program-side declaration limit.
    `max_physical_units` is the machine-side count of positions the level
    divides over, independently of hardware residency limits.
  - `parallel_level`, when present, MUST name an entry in `topologies`.
    `parallel()` MUST return that entry. An empty aggregate MUST use `None` and
    return `None` from `parallel()`.
  - A subclass MUST inherit its base Target's projections through normal Python
    inheritance. It MAY override `get_facts` for hardware that differs and
    delegate unknown requests to `super()`.
  - A missing projection MUST fail immediately and MUST NOT fall back to a
    built-in Target or substitute built-in hardware values.
  - A Facts aggregate MUST be a frozen dataclass. Aggregates MUST NOT inherit
    one universal Facts base.
  - `query` is owned by the requesting algorithm. A hardware-only
    projection MUST require it to be absent, while a program-dependent one MAY
    validate its own private query type. There is no common query base or
    mandatory public program-view type.
  - A returned value that is not an instance of the requested Facts type MUST
    fail at the projection boundary, not inside the consuming algorithm.
  - Projection MUST be a read. It MUST NOT analyze IR, build a constraint model,
    solve, export a plan, or mutate the Target, the IR, or runtime
    state. It only converts what the specification already records.
  - There MUST be no public Facts registration step or global Target Facts
    table. A custom Target provider registers only its Target class.
