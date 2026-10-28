# TileFoundry CUDA Spec — Codegen

The [core codegen contract](../../../../../../docs/spec/codegen.md) owns grouping,
products, signatures and linking. This spec owns the CUDA implementations.

## 1. Emission and launch boundary

`CudaTarget` binds `EmitContext` to `CudaCodegenContext` and `LinkableModule`
to `CudaLinkableModule`. Both full and skeleton emission are supported;
skeleton emission rejects targets with `device_count`, because its editable
body does not implement multi-device execution. Skeletons retain signatures,
parameter views, specialization dispatch and launchers, replacing statements
with `// TODO: <fn>`.

The device unit owns kernels and C-ABI launchers. A launcher asked for more
than 48 KiB of dynamic shared memory MUST first raise its kernel's
`cudaFuncAttributeMaxDynamicSharedMemorySize` to that size. A launcher reads
launch status immediately and returns the CUDA error text or null, including
when the device refuses the raised limit. Host entries raise on an error; no
exception crosses the C ABI and a refused launch never returns as though the
kernel ran. A host unit MUST NOT reference CUDA or CuTe
types.

CUDA registers the host writer for shared `Launch` at
`(CpuTarget, Role.EMIT, Launch)`. It selects CUDA's CALLEE and CALLER writers
through the registry using the host's local names. `MeshScope` statement
emission and `MeshCoord` scalar emission use the same per-target registry.

## 2. Context and value emission

```python
class CudaCodegenContext(EmitContext):
    """A compile writing CUDA: the shared context plus what only CUDA states."""

    launches: Mapping[int, Geometry]
    dynamic_extents: dict[str, str]

    def bind_extents(self, params) -> None: ...
    def reset_barrier_ids(self) -> None: ...
    def alloc_barrier_id(self) -> int: ...
    def declare_smem_base(self, write: Callable[[CudaCodegenContext], None]) -> None: ...
    def smem_base(self) -> str: ...
    def dtype_to_cpp(self, dtype_name: str) -> str: ...
```

- constraints:
  - `dynamic_extents` is derived from parameter types by `bind_extents`.
    `launches` records settled launch geometry keyed by `id(fn)`.
  - Each kernel declares one byte-addressed dynamic shared-memory base, ahead
    of its first statement, when its body names the base through `smem_base`.
    `declare_smem_base` writes the body first so every scope in the kernel
    reaches that one declaration. Numeric `TensorView` addresses add byte
    offsets before conversion to CuTe shared-memory pointers.
  - Handlers obtain target type spellings and hardware counters through the
    context rather than rediscovering them from IR.

CUDA value emission lowers `TupleGetItem` according to its index form. A constant
index names the selected tuple element directly. A dynamic index requires the
homogeneous tuple established by type inference and emits
`cute::array{a, b, c}[index]`; dimension arithmetic in the index remains runtime
C++ arithmetic.

An IR `Tuple` is structural and has no target-side storage. A `LetStmt` binding
one emits no C++ variable and continues with its body; its authored name (for
example `lhs_stages`) therefore does not appear in generated C++. Each dynamic
`TupleGetItem` use constructs its own `cute::array`, so indexing the same tuple
N times constructs N arrays. Constant selection remains valid for heterogeneous
tuples because it does not materialize an aggregate. The codegen context records
the structural tuple by its fresh SSA `Var` identity so consumers can recover
its elements without target-side storage.

CUDA emits `CopyAsyncBulk` and `LdMatrix` as the uniform runtime calls
`tilefoundry::ops::copy_async_bulk(...)` and
`tilefoundry::ops::ldmatrix(...)`. `CopyAsyncTensor` is nevertheless a public
TIR declaration, but CUDA emission MUST fail explicitly until the host can
construct and pass encoded tensor maps; it MUST NOT substitute a bulk or
thread-issued copy. `CopyReduceAsyncTensor` has the same explicit unsupported
boundary; it MUST NOT substitute a copy that drops the reduction. `StMatrix` likewise MUST fail explicitly until stmatrix
runtime support exists.

## 3. Runtime-owned op dispatch

Codegen emits one uniform runtime call, passing operand `ShardLayout`s and
static participant geometry as template parameters. The runtime selects a
layout-compatible implementation at compile time. Codegen MUST NOT select a
tier or attach tier selection to an op. See
[runtime §2.6](./runtime.md#26-cudaops).

## 4. Linking

`CudaLinkableModule.link` adds `device_<index>.cu`, enables the CUDA language,
and owns architecture selection, compiler flags and packaged runtime includes.
Runtime headers are resolved with `importlib.resources` and included directly;
there is no umbrella header or target-selection macro.

- constraints:
  - Device headers require C++20. Set per-source `-std=c++20` instead of
    `CUDA_STANDARD 20`: older CMake versions reject that dialect and nvcc uses
    the last `-std` option.
  - CUTLASS lookup MUST first use `TILEFOUNDRY_CUTLASS_INCLUDE`. An invalid
    explicit directory MUST fail naming the variable and value. Without the
    variable, search source ancestors for `third_party/cutlass/include`, then
    fail naming the variable and searched paths.
  - CUDA derives `_launch` and `_kernel` from the core's mangled symbol root;
    it MUST NOT rebuild function paths independently.

## 5. HIR skeletons

Skeleton emission MUST reject an HIR device call into a direct child's function
with implicit ConstTensor arguments, naming caller, callee and omitted weights
and requesting explicit weight parameters or orchestration calls.

`tilefoundry_nvidia/cuda/codegen/hir/function.py` owns HIR skeleton emission. An HIR function
enters codegen only as the callee a `Launch` names; Python wraps any other
function and calls the exported entries. Each launched function gets one
editable `__device__` body taking cute parameter views, one `__global__` entry
and one launcher.
Device bodies (including variants) precede all global entries, ordered stably
by increasing reachable-function count so callees precede callers. Prototypes
have empty bodies and variants receive their display-name path segment or
`v<index>` fallback. Global entries wrap parameters and dispatch by the same
range rules as TIR, trapping outside the specialization ranges.

HIR functions with equal effective topology hierarchies share a device unit,
including children that explicitly redeclare the same hierarchy; TIR grouping
is unchanged. HIR same-kernel context rules are owned by
[hir §1.1](../../../../../../docs/spec/hir.md#11-function). An HIR function's parameters are the
arguments its `Launch` binds
([passes §7.3](./passes.md#73-insert_default_host_entry)), and its variants
take the same ones; generated names follow [codegen §4.4](../../../../../../docs/spec/codegen.md#44-signatures).
A synthesized host entry launching HIR uses that HIR function's symbol and
registration name; its own IR name does not add another path segment.
