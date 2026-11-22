# TileFoundry Metal Spec — Target

The Metal target describes an Apple GPU for HIR matching, TIR instruction
declarations and lowering to TIR. It has no code generation and no runtime.
The instructions are specified in [tir](./tir.md).

## 1. `metal.Architecture`

```python
class Architecture(target.Architecture):
    """An Apple GPU family: the native functions it provides and its SIMD-group width.

    Attributes:
        name: Architecture identity used by compilation.
        capabilities: Native Metal Shading Language function names.
        simd_width: Number of threads in one SIMD group.
    """

    name: str
    capabilities: tuple[str, ...]
    simd_width: int
```

- constraints:
  - Every value MUST come from the installed architecture document ([target §10](../../../../../../docs/spec/target.md#10-installed-hardware-resources)). No field has a default.
  - Each capability MUST be the name of a native Metal Shading Language function, such as `simd_sum`.
  - A capability MUST NOT name a machine instruction. The architecture is an instruction profile, not a machine ISA.
  - One capability entry covers the `simd_` function and its `quad_` form.

## 2. `metal.Device`

```python
class Device(target.Device):
    """One Apple GPU: threadgroup limits and the unified memory it shares with the CPU.

    Attributes:
        name: Device identity.
        max_threads_per_threadgroup: Device limit on the threads of one threadgroup.
        threadgroup_memory_bytes: Threadgroup memory of one threadgroup.
        unified_memory_capacity_bytes: Capacity of the memory that the GPU shares with the CPU.
    """

    name: str
    max_threads_per_threadgroup: int
    threadgroup_memory_bytes: int
    unified_memory_capacity_bytes: int
```

- constraints:
  - Every value MUST come from the installed device document. No field has a default.
  - The document records the unified memory bandwidth as a vendor value. No Facts projection reads it.
  - The document declares two entries `unavailable`: `compute.resident_threadgroups` and `memory.register_per_thread`.
    A projection MUST NOT estimate them.

## 3. `metal.Target`

```python
class Target(target.Target):
    """The Metal GPU of one device document. Its architecture is the one that document names.

    Attributes:
        name: Registered target name.
        device_id: Installed device document ID.
        architecture: The GPU family.
        device: The GPU product.
    """

    name: ClassVar[str] = "metal"
    device_id: str | Path = "apple.m2_pro.gpu"

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
from tilefoundry_apple import metal

tgt = metal.Target("apple.m2_pro.gpu")
```

- constraints:
  - `device_id` MUST be an installed device document ID or the path of a complete device document ([target §10.2](../../../../../../docs/spec/target.md#102-registry-and-resolution)).
    The constructor has no `architecture` parameter, and it does not take a `Device` or `Architecture` value.
  - An omitted `device_id` MUST select `apple.m2_pro.gpu`.
  - The architecture MUST be read from the sole compatibility declaration of the device document.
  - The construction MUST check the declared compatibility of the pair.
  - An unknown device ID MUST raise an error that lists the installed IDs.
  - `identity` MUST be the ID the device document declares. Two values are equal when their `device_id` values are equal.
  - `capabilities()` MUST return the capability names of `architecture`.
    `documents()` MUST return the architecture document, then the device document.
  - `from_installed()` MUST contain one value for each installed device document.

## 4. Topology

The topologies are `threadgroup` and `thread`, in this order.

| Topology | `max_logical_units` | `max_physical_units` |
|---|---|---|
| `threadgroup` | `None` | `None` |
| `thread` | `Device.max_threads_per_threadgroup` | `None` |

- constraints:
  - `get_facts(TopologyFacts)` MUST return the two limits above, and `parallel_topology` MUST be `threadgroup`.
  - The threadgroup count has no limit. No sourced residency count exists.
  - `get_facts(TopologyLimits, name)` MUST return the limit of the named topology.
  - `get_facts(MemoryExecutionFacts)` MUST name `thread` only.
  - A SIMD group is not a topology. An instruction declares it as a group of `thread` units in its execution mesh ([tir §1](./tir.md#1-common-constraints)).

## 5. Facts

| Facts type | Answer |
|---|---|
| `TopologyFacts`, `TopologyLimits`, `MemoryExecutionFacts` | [§4](#4-topology) |
| `MemoryHierarchyFacts` | The three explicit memory levels below. No implicit levels and no relations. |
| `ThroughputFacts`, `PerformanceServiceFacts` | `UnsupportedCapabilityError` |

Instruction selection reads `Architecture.capabilities` directly ([tir §2](./tir.md#2-ops)).

| Memory level | Scope | Owner | Capacity |
|---|---|---|---|
| `gmem` | `device` | `target` | `Device.unified_memory_capacity_bytes` |
| `smem` | `threadgroup` | `threadgroup` | `Device.threadgroup_memory_bytes` |
| `rmem` | `thread` | `thread` | `None` |

- constraints:
  - Throughput and service Facts MUST raise `UnsupportedCapabilityError`. No Metal throughput is measured.
  - The capacity of `rmem` MUST stay `None`. Apple publishes no register capacity.
