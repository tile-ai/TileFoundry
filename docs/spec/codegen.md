# TileFoundry Spec — Codegen

Codegen turns verified, lowered `tir.PrimFunction`s into a loadable artifact.
It owns the whole producer side of the build: emitting per-target source,
assembling each target's translation unit, and linking those units into one
host-callable shared library. Loading that artifact and exposing it as a
`RuntimeModule` is owned by [runtime](./runtime.md).

```mermaid
flowchart LR
    TIR["verified <b>tir.PrimFunction</b>s"]
    Emit["per-target <b>Emitter</b>"]
    LM["<b>LinkableModule</b><br/>(per target)"]
    Link["<b>link</b>"]
    Linked["<b>LinkedModule</b><br/>artifact + metadata"]
    RM["<b>RuntimeModule</b><br/>(see runtime)"]

    TIR --> Emit --> LM --> Link --> Linked
    Linked -. runtime load .-> RM
```

## 1. Pipeline

- **Input** is verified TIR. HIR Ops MUST NOT reach codegen.
- A module's functions are grouped by their `target` (`cuda` / `cpu`). Each
  group is emitted by its target's emitter into one `LinkableModule`.
- The link step compiles every `LinkableModule` with its own toolchain and
  links them into one `LinkedModule` — a host-callable shared library plus the
  host-visible metadata the loader needs.
- Codegen does not run passes, does not load or launch device code, and does
  not own the user-facing entry points (`compile` / `build` / `jit`).
- **Host / device boundary.** A host `LinkableModule` MUST NOT reference CUDA or
  CuTe symbols or types. A CUDA `LinkableModule` owns the kernels and their
  C-ABI launch shims. The host module invokes device code only through that
  C-ABI shim.

The target-specific emitter behavior — how a `cpu` vs `cuda` function emits, the
dispatch and shape-scalar ABI, program-shape / dynamic-CTA accessors, and the
`ShardLayout` runtime mapping — is owned by [target](./target.md).

## 2. Emitter


An emitter walks a verified `tir.PrimFunction` and produces source plus the
metadata the link step needs.

### 2.1 Emitter registry

Each target registers an emitter, resolved by target name:

```python
emit = get_emitter("cuda")    # tuple[PrimFunction, ...] -> LinkableModule
```

An emitter MUST consume only TIR and MUST return a `LinkableModule` for its
target. The emitter file layout mirrors the IR file layout
(`codegen/<target>/tir/...` parallels `ir/tir/...`); the mirror rule is owned
by [code-organization](./code-organization.md).

### 2.2 Per-Op handler registry

Within a target, per-Op handlers are registered for the emitter:

```python
@register_codegen_cuda(Copy)
def _(call: Call, ctx: CodegenContext) -> None:
    ctx.emit(f"tilefoundry::copy({ctx.expr(call.args[0])}, {ctx.expr(call.args[1])});")
```

Dispatch is owned by [visitor-registry §6](./visitor-registry.md). A handler
receives the `Call` (the wrapped Op inside `Evaluate`) plus a `CodegenContext`,
and MUST emit through `ctx.emit(...)`; raw `print` / direct file writes are
prohibited.

### 2.3 `CodegenContext`

`CodegenContext` is the per-walk state object passed to handlers. It exposes the
helpers a handler is allowed to use:

- `emit(line: str)` — append a target source line.
- `expr(node: Expr) -> str` — render an `Expr` as a target expression string.
- `dtype_to_cpp(dtype_name: str) -> str` — backend dtype mapping.
- `make_var_name(...)` — allocate fresh target-side identifiers.

A handler MUST NOT reach into the IR for type strings on its own; the context is
the single source of truth. Other helpers MAY be added per target.

### 2.4 Effect Op dispatch

Effect Ops (`Copy`, `Fill`, `Mma`, `tir.nn.*`, ...) appear in Stmt
position as `Evaluate(op, args)` rather than as Stmt subclasses. The
walker matches `Evaluate` and dispatches on `type(callable)` through
the handler registry. Handlers stay small; the runtime function they
call carries the semantic load.

## 3. Runtime-owned op dispatch

Where more than one runtime template implements an op, codegen emits **one
uniform runtime op call**, passing the operand `ShardLayout`s (and any
codegen-static participant geometry) as compile-time template parameters. The
runtime template dispatches on those layouts at compile time; codegen does not
select a tier, compute a per-tier parameter, or carry the selection on the TIR
op. This is the codegen side of the runtime-owned dispatch principle, whose
contract lives in [runtime.md §3](runtime.md#3-runtime-ops). The target-side
emission that produces these calls is owned by
[target §5](./target.md#5-target-driven-emission).

## 4. Codegen products

### 4.1 `LinkableFunction`


One lowered function's pre-link source.

```text
LinkableFunction(name: str, source: str)
```

- kind: Python class
- fields:
  - name: the function / kernel symbol
  - source: that function's emitted text
- constraints:
  - MUST be the function / kernel symbol.
  - MUST be that function's emitted text.

### 4.2 `LinkableModule`

One target's pre-link translation unit.

```text
LinkableModule(target: str, language: str, source: str, functions: tuple[LinkableFunction, ...])
```

- kind: Python class
- fields:
  - target: the function target name (`cuda` / `cpu`)
  - language: the source language (`cu` / `cpp`)
  - source: the assembled translation-unit text
  - functions: the module's constituent `LinkableFunction`s
- constraints:
  - MUST be the function target name (`cuda` / `cpu`).
  - MUST be the source language: `cu` for a CUDA translation unit, `cpp` for a
    host translation unit.
  - MUST be the assembled translation-unit text the link step compiles.
  - MUST list the module's constituent `LinkableFunction`s, in emission order.

A `LinkableModule` is a build artifact, not a runtime object and not a
user-callable.

### 4.3 `LinkedModule`

The link output: a loadable artifact plus the host-visible metadata the loader
needs.

```text
LinkedModule(library_path: Path, source: str, entry: CallableType, launch_config: LaunchConfig, kernels: tuple[KernelInfo, ...])
```

- kind: Python class
- fields:
  - library_path: the produced shared library
  - source: the assembled host + device source
  - entry: the host-visible callable type of the module entry
  - launch_config: the entry's launch geometry (grid / block extents)
  - kernels: the ABI of the module's `__global__` kernels
- constraints:
  - MUST point at the produced shared library.
  - MUST carry the assembled host + device source — the diagnostic source the
    runtime exposes as `RuntimeModule.source` ([runtime](./runtime.md)).
  - MUST be the host-visible callable type of the module entry.
  - MUST carry the entry's launch geometry (grid / block extents).
  - MUST list the ABI of the module's `__global__` kernels.

The `entry` `CallableType`, `launch_config` `LaunchConfig`, and `kernels`
`KernelInfo` are host-visible ABI metadata types owned by
[runtime](./runtime.md); codegen references them on `LinkedModule` and MUST NOT
redefine them.

The link step consumes the per-target `LinkableModule`s, compiles each with its
own toolchain, and links them into one `LinkedModule`. `LinkedModule` is
consumed by the runtime loader ([runtime](./runtime.md)); the concrete compiler
commands are an implementation detail and not part of the contract.
