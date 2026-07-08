# TileFoundry Spec — Target

A `Target` is the back-end a verified, lowered `tir.PrimFunction` is compiled
for. It is a capability descriptor: it names the back-end and carries the
back-end's compile-time parameters and the program topology levels it admits. A
function carries a single `target`, which selects its target-specific lowering
and codegen. Emitting source and linking the artifact are owned by
[codegen](./codegen.md); loading the artifact as a `RuntimeModule` is owned by
[runtime](./runtime.md).

## 1. Role and scope

- **Input** is verified TIR. HIR Ops MUST NOT reach a `Target`.
- A `Target` describes capability; it does not emit source, link, run passes,
  load or launch device code, or own the user-facing entry points
  (`compile` / `build` / `jit`).
- A target is resolved by name: a string reflects into that back-end's default
  target object (`"cuda"` → `CudaTarget()`), and a `Target` object passes
  through unchanged. Codegen groups a module's functions by their target name
  and emits one `LinkableModule` per group
  ([codegen §1](./codegen.md#1-pipeline)).

## 2. `Target`

```text
Target(name: str)
```

- kind: Python class
- fields:
  - name: the stable back-end identifier
- constraints:
  - MUST be the stable back-end identifier used for target resolution and for the
    function-target grouping in codegen.

A `Target` MUST NOT carry the linkable / linked artifact dataclasses; those are
codegen products ([codegen §4](./codegen.md#4-codegen-products)).

## 3. `CudaTarget`

CUDA is the current reference target.

```text
CudaTarget(name: str = "cuda", arch: str = "sm_90", topology_levels: tuple[str, ...] = ("cta", "thread"))
```

- kind: Python class
- fields:
  - name: the back-end identifier, fixed to `"cuda"`
  - arch: the SM architecture the device source is compiled for
  - topology_levels: the program topology level set the target admits
- constraints:
  - MUST be `"cuda"`.
  - MUST name the SM architecture the device source is compiled for.
  - MUST be the program topology level set the target admits: `{cta, thread}`.
  - `warp` / `lane` / `warpgroup` MUST be expressed as axes of a thread mesh
    layout ([shard](./shard.md)), not as program topology levels.
  - A function whose declared program topology levels are not a subset of this set
    MUST raise at lowering. The level set is consumed by the device program
    accessors ([§7](#7-program-shape-and-dynamic-cta)).

## 4. `CpuTarget`

```text
CpuTarget(name: str = "cpu")
```

- kind: Python class
- fields:
  - name: the back-end identifier, fixed to `"cpu"`
- constraints:
  - MUST be `"cpu"`.

The CPU target hosts the entry that marshals arguments and invokes the device
entry through its C-ABI launch shim
([§5](#5-target-driven-emission)).

## 5. Target-driven emission

Emission is split by function `target`; one `LinkableModule` is produced per
target group.

- A `cpu`-target entry function emits the **host translation unit**: a host
  wrapper that marshals `tvm::ffi::Tensor` arguments and invokes the device
  entry through its C-ABI launch shim. For a dispatch prototype the wrapper
  also performs the `DispatchCall` selection (§6).
- A `cuda`-target function emits the **device translation unit**: a
  `__global__` kernel plus its C-ABI launch shim. An operand's layout selects
  how it is viewed inside the kernel — `cute::` for a plain `Layout`,
  `tilefoundry::` shard tensor for a `ShardLayout` (§8) — it does not change the
  function into a `__device__`-parameter function.

## 6. Dispatch and shape-scalar ABI


A `tir.PrimFunction` produced by HIR→TIR lowering for a dispatch prototype
([hir §1.1](./hir.md#11-function)) emits as a host dispatch entry
plus its variant kernels:

- **Dispatch entry** (PrimFunction whose body is a single `tir.DispatchCall`):
  the host module emits a host-only entry — no kernel. It reads the dispatch
  subject from the host-visible tensor shape, evaluates the case predicates in
  source order, and invokes the matching variant's C-ABI launch shim. The
  `fallback` MUST fail the host call with a host-side error.
- **Variant**: a `cuda`-target function (§5) — a `__global__` kernel plus its
  C-ABI launch shim in the CUDA module, carrying the hidden shape-scalar
  parameters below. A variant has no host wrapper of its own; the dispatch entry
  calls its shim.

**Host-visible entry symbol.** The host-visible entry wrapper (the CPU entry) is
emitted under an internal symbol `__tilefoundry_<sanitized>_host`, where
`<sanitized>` is the user-facing name with `$` replaced by `__` (`$` is a GCC
extension, not portable C++). The user-facing name is republished via the
runtime ABI macro:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(<name>, __tilefoundry_<sanitized>_host);
```

**Shape-scalar parameter ABI.** Each `tir.ShapeOf(param, axis)`
([tir §2.2](./tir.md#22-shapeof)) reachable from a PrimFunction body adds a hidden
kernel-scalar parameter named `f"{param.name}_shape_<axis>"` — a rank-0 `i32`
shape-scalar `TensorType` with `storage=None`. The host wrapper extracts the
value from the
corresponding `tvm::ffi::Tensor.shape()[axis]` and forwards it. These hidden
parameters are filtered out of the exported `CallableType.params` so they remain
invisible at the user FFI surface. A user-declared rank-0 `i32` parameter is
*not* filtered — the filter is keyed on the canonical `<name>_shape_<axis>` form
synthesised by lowering.

## 7. Program shape and dynamic CTA

The emitted device source reads its program geometry through two accessors:

- `program_shape<T>()` MUST be the shape composed of a topology level's program
  dims.
- `program_dim<T>()` MUST be the size of one topology level `T`.

For a static CTA count, `program_dim<cta>()` is a compile-time constant and a
constexpr `program_shape<cta>` is emitted. For a launch-provided (dynamic) CTA
count, the emitter MUST NOT emit a constexpr `program_shape<cta>`; device code
MUST read the count through `program_dim<cta>()`, which resolves to the
launch-provided grid extent. The topology level set a target admits is owned by
[§3](#3-cudatarget).

## 8. `ShardLayout` / `ShardTensor` mapping

A `ShardLayout` ([shard §7](./shard.md)) on a TIR tensor type is materialised
through `tilefoundry::make_shard_tensor(buffer, global_layout, shard_layout)`
([runtime](./runtime.md)). A handler that consumes a sharded operand emits the
shard-tensor view; effect Ops then route through `tilefoundry::` runtime helpers
(for example `tilefoundry::copy`) instead of plain `cute::` primitives. A plain
`Layout` operand goes through `cute::` directly. The selection is handler-local,
based on `arg.type.layout`.

Codegen consumes the `ShardLayout` already present on the TIR tensor type; the
cute MMA fragment → `ShardLayout` recipe that produces it is owned by the
lowering pass ([passes.md §7.1](./passes.md#71-hirtotirpass)).
