# TileFoundry ANE Spec — Target

The ANE target describes the Apple Neural Engine of an Apple silicon chip.
The [codegen](./codegen.md) emits descriptor images and C++ entry points.

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
    """One ANE: its cores, the unified memory it shares with the CPU and its measured fp16 rate.

    Attributes:
        name: Device identity.
        core_count: Number of cores the vendor publishes.
        unified_memory_capacity_bytes: Capacity of the memory that the ANE shares with the CPU.
        f16_throughput: Measured rate of an fp16 convolution, in FLOPs per second.
    """

    name: str
    core_count: int
    unified_memory_capacity_bytes: int
    f16_throughput: int
```

- constraints:
  - Every value MUST come from the installed device document. No field has a default.
  - The document records the unified memory bandwidth as a vendor value. No Facts projection reads it.
  - The document declares `memory.l2.capacity` `unavailable`. A projection MUST NOT estimate it.
  - `f16_throughput` is the wall-clock rate of one 3x3 fp16 convolution on the host, with 2 FLOPs for each
    multiply-accumulate. For `apple.m2_pro.ane` it is 16.71 TFLOP/s. The document states the shape and the method.
    No Facts projection reads it; `get_facts(ThroughputFacts)` stays unsupported.
  - The mapping of the cores to the tasks of one program is unknown. The target MUST NOT derive a topology from `core_count`.

## 3. `ane.Target`

```python
class Target(target.Target):
    """The ANE of one device document. Its architecture is the one that document names.

    Attributes:
        name: Registered target name.
        device_id: Installed device document ID.
        architecture: The ANE generation.
        device: The ANE product.
    """

    name: ClassVar[str] = "ane"
    device_id: str | Path = "apple.m2_pro.ane"

    @property
    def architecture(self) -> Architecture: ...
    @property
    def device(self) -> Device: ...
    @classmethod
    def from_installed(cls) -> tuple[Target, ...]: ...
    @property
    def identity(self) -> str: ...
    def capabilities(self) -> frozenset[str]: ...
    def documents(self) -> tuple[Document, ...]: ...
    def get_facts(self, facts_type: type[FactsT], query=None) -> FactsT: ...
```

```python
# example
from tilefoundry_apple import ane

tgt = ane.Target("apple.m2_pro.ane")
```

- constraints:
  - `device_id` MUST be an installed device document ID or the path of a complete device document ([target §10.2](../../../../../../docs/spec/target.md#102-registry-and-resolution)).
    The constructor has no `architecture` parameter, and it does not take a `Device` or `Architecture` value.
  - An omitted `device_id` MUST select `apple.m2_pro.ane`.
  - The architecture MUST be read from the sole compatibility declaration of the device document. For the default device, it is `apple.h14`.
  - The construction MUST check the declared compatibility of the pair.
  - An unknown device ID MUST raise an error that lists the installed IDs.
  - `identity` MUST be the ID the device document declares. Two values are equal when their `device_id` values are equal.
  - `capabilities()` MUST return the capability names of `architecture`.
    `documents()` MUST return the architecture document, then the device document.
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
