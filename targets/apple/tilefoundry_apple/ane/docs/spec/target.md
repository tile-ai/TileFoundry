# TileFoundry ANE Spec — Target

The ANE target describes the Apple Neural Engine of an Apple silicon chip.
It has no code generation and no runtime.

## 1. `ane.Architecture`

```python
class Architecture(target.Architecture):
    """An ANE generation: the instructions it runs and the tensor extents an instruction accepts.

    Attributes:
        name: Architecture identity used by compilation.
        capabilities: Instruction mnemonics the generation runs.
        max_channels: Largest channel extent of one instruction.
        max_height: Largest height extent of one instruction.
        max_width: Largest width extent of one instruction.
        max_kernel: Largest kernel extent of one convolution.
    """

    name: str
    capabilities: tuple[str, ...]
    max_channels: int
    max_height: int
    max_width: int
    max_kernel: int
```

- constraints:
  - Every value MUST come from the installed architecture document ([target §10](../../../../../../docs/spec/target.md#10-installed-hardware-resources)). No field has a default.
  - Each capability MUST be an instruction mnemonic. It has the form `ne.<form>` or `pe.<form>`.
  - The architecture `apple.h14` lists six capabilities: `ne.conv`, `ne.bypass`, `pe.pool`, `pe.unary`, `pe.binary` and `pe.reduce`.
  - The extent limits MUST come from the task descriptor field ranges of the generation.
  - A capability lists an instruction form that compiled programs use. It does not claim that the form ran on hardware.

## 2. `ane.Device`

```python
class Device(target.Device):
    """One ANE: its core count and the unified memory it shares with the CPU.

    Attributes:
        name: Device identity.
        core_count: Number of cores the vendor publishes.
        unified_memory_capacity_bytes: Capacity of the memory that the ANE shares with the CPU.
    """

    name: str
    core_count: int
    unified_memory_capacity_bytes: int
```

- constraints:
  - Every value MUST come from the installed device document. No field has a default.
  - The document records the unified memory bandwidth as a vendor value. No Facts projection reads it.
  - The document declares two entries `unavailable`: `memory.l2.capacity` and `compute.throughput`.
    A projection MUST NOT estimate them.
  - The mapping of the cores to the tasks of one program is unknown. The target MUST NOT derive a topology from `core_count`.

## 3. `ane.Target`

```python
class Target(target.Target):
    """ANE target composed from one architecture and one device.

    Attributes:
        name: Registered target name.
        architecture: The ANE generation.
        device: The ANE product.
        architecture_id: Installed architecture document ID, or None.
        device_id: Installed device document ID, or None.
        architecture_digest: Content digest of the architecture document, or None.
        device_digest: Content digest of the device document, or None.
        arch: The architecture name.
    """

    name: ClassVar[str] = "ane"
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

    @classmethod
    def from_installed(cls) -> tuple[Target, ...]: ...
    @property
    def identity(self) -> str: ...
    def get_facts(self, facts_type: type[FactsT], query=None) -> FactsT: ...
```

```python
# example
from tilefoundry_apple import ane

tgt = ane.Target("apple.m2_pro.ane")
```

- constraints:
  - `device` and `architecture` MUST accept an installed document ID, a document path, or a concrete value ([target §10.2](../../../../../../docs/spec/target.md#102-registry-and-resolution)).
  - An omitted `device` MUST select `apple.m2_pro.ane`.
  - An omitted `architecture` MUST be read from the sole compatibility declaration of the device document. For the default device, it is `apple.h14`.
  - A pair selected by ID MUST be checked for declared compatibility.
  - An unknown device ID MUST raise an error that lists the installed IDs.
  - `arch` MUST equal `architecture.name`. `identity` MUST equal `device_id` when one is set.
  - `from_installed()` MUST contain one value for each installed device document.

## 4. Topology

The only topology is `engine`.

| Topology | `max_logical_units` | `max_physical_units` |
|---|---|---|
| `engine` | `1` | `1` |

- constraints:
  - `get_facts(TopologyFacts)` MUST return the one limit above, and `parallel_topology` MUST be `engine`.
  - One engine runs one program.
  - `get_facts(TopologyLimits, name)` MUST return the limit of the named topology.
  - `get_facts(MemoryExecutionFacts)` MUST name `engine` only.
  - A program MUST declare no topology other than `engine`. A program MUST declare the size of `engine` as 1.
  - A core is not a topology. No sourced mapping from cores to tasks exists.

## 5. Facts

| Facts type | Answer |
|---|---|
| `TopologyFacts`, `TopologyLimits`, `MemoryExecutionFacts` | [§4](#4-topology) |
| `MemoryHierarchyFacts` | The two explicit memory levels below. No implicit levels and no relations. |
| `ThroughputFacts`, `PerformanceServiceFacts` | `UnsupportedCapabilityError` |

Instruction selection reads `Architecture.capabilities` directly.

| Memory level | Scope | Owner | Capacity |
|---|---|---|---|
| `gmem` | `device` | `target` | `Device.unified_memory_capacity_bytes` |
| `smem` | `engine` | `engine` | `None` |

- constraints:
  - `gmem` MUST name the unified memory. The CPU shares it, so no topology owns it.
  - `smem` MUST name the on-chip L2 buffer of the engine.
  - The capacity of `smem` MUST stay `None`. No source states the L2 capacity.
  - Throughput and service Facts MUST raise `UnsupportedCapabilityError`. No ANE throughput is measured.
  - Apple publishes 15.8 trillion operations per second without a data type. A projection MUST NOT convert it to a rate.
