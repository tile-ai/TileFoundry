# TileFoundry Spec — Codegen

Codegen turns verified, lowered `tir.PrimFunction`s into a loadable artifact.
It owns the whole producer side of the build: emitting per-target source,
assembling each target's translation unit, and linking those units into one
host-callable shared library. Loading that artifact and exposing it as a
`RuntimeModule` is owned by [runtime](./runtime.md).

```mermaid
flowchart LR
    TIR["verified <b>tir.PrimFunction</b>s"]
    Emit["Target-selected <b>CodeGenerator</b>"]
    LM["<b>LinkableModule</b><br/>(per target)"]
    Link["<b>link</b>"]
    Linked["<b>LinkedModule</b><br/>artifact + metadata"]
    RM["<b>RuntimeModule</b><br/>(see runtime)"]

    TIR --> Emit --> LM --> Link --> Linked
    Linked -. runtime load .-> RM
```

## 1. Pipeline

- **Input** is verified TIR. HIR Ops MUST NOT reach codegen.
- A module's functions are grouped by equal Target values in source order. Each
  group is emitted by the exact Target's CodeGenerator into one
  `LinkableModule`.
- The link step compiles every `LinkableModule` with its own toolchain and
  links them into one `LinkedModule` — a host-callable shared library plus the
  host-visible metadata the loader needs.
- Codegen does not run passes, does not load or launch device code, and does
  not own the user-facing entry points (`compile` / `build` / `jit`).
- **Host / device boundary.** A host `LinkableModule` MUST NOT reference CUDA or
  CuTe symbols or types. A CUDA `LinkableModule` owns the kernels and their
  C-ABI launch shims. The host module invokes device code only through that
  C-ABI shim.

The target-specific generator behavior — how a CPU vs CUDA function emits, the
dispatch and shape-scalar ABI, program-shape / dynamic-CTA accessors, and the
`ShardLayout` runtime mapping — is owned by [target](./target.md).

## 2. CodeGenerator

A generator walks verified `tir.PrimFunction`s and produces source plus the
metadata the link step needs.

### 2.1 Target-selected service

```python
class CodeGenerator:
    emit: Callable[
        [Module, tuple[PrimFunction, ...], Target], LinkableModule
    ]


class Target:
    def get_code_generator(self) -> CodeGenerator: ...


def emit_cuda_module(
    module: Module,
    functions: tuple[PrimFunction, ...],
    target: Target,
) -> LinkableModule: ...
```

- constraints:
  - A Target MUST return one immutable CodeGenerator descriptor. A subclass MAY
    inherit its base generator without another registration step.
  - All generators MUST share the `(module, functions, target)` callable shape.
  - Generator selection MUST NOT branch on `Target.name`. There MUST be no
    emitter registry or string-to-emitter lookup.
  - A second unequal CUDA Target group MUST fail before any generator emits;
    multiple device translation units or architectures in one linked artifact
    are unsupported.

A generator MUST consume only TIR and MUST return a `LinkableModule` for its
backend. The emitter file layout mirrors the IR file layout
(`codegen/<target>/tir/...` parallels `ir/tir/...`); the mirror rule is owned
by [code-organization](./code-organization.md).

### 2.2 Per-Op handler registry

```python
def handler(call: Call, ctx: CodegenContext) -> None: ...
```

- constraints:
  - a handler is registered inside a concrete generator with the per-backend
    `register_codegen_*` decorator ([visitor-registry §6](./visitor-registry.md#6-instance-3--codegen_));
    dispatch (matching `Evaluate` and selecting the handler) is owned by
    visitor-registry.

Dispatch is owned by [visitor-registry §6](./visitor-registry.md#6-instance-3--codegen_). A handler
receives the `Call` (the wrapped Op inside `Evaluate`) plus a `CodegenContext`,
and MUST emit through `ctx.emit(...)`; raw `print` / direct file writes are
prohibited.

### 2.3 `CodegenContext`

```python
class CodegenContext:
    """Mutable CUDA source builder passed to registered handlers."""

    def reset_barrier_ids(self) -> None: ...
    def alloc_barrier_id(self) -> int: ...
    def dtype_to_cpp(self, dtype_name: str) -> str: ...
    def register_kernel_param(self, var) -> None: ...
    def is_kernel_param(self, var) -> bool: ...
    def emit(self, line: str) -> None: ...
    def blank(self) -> None: ...
    def indent(self) -> None: ...
    def dedent(self) -> None: ...
    def name_for(self, var) -> str: ...
    def source(self) -> str: ...
    def capture(self, fn) -> str: ...
    def emit_node(self, node) -> None: ...
```

- constraints:
  - This is the concrete CUDA context; it has a zero-argument constructor and
    keeps its output, indentation, symbol table, and counters private.
  - `emit_node` dispatches `Evaluate(op, args)` by the wrapped Op class; all
    other nodes dispatch by their own class.
  - `source` returns accumulated source text, and `capture` temporarily isolates
    only the output buffer while preserving indentation and symbol bindings.
  - The context is the single source of truth for target-side type strings, so
    handlers do not read the IR for them directly.

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
contract lives in [runtime §2.6](./runtime.md#26-cudaops). The target-side
emission that produces these calls is owned by [target](./target.md).

## 4. Codegen products

### 4.1 `LinkableFunction`


One lowered function's pre-link source.

```python
class LinkableFunction:
    """One lowered function's pre-link source.

    Attributes:
        name: attribute; function or kernel symbol.
        source: attribute; emitted function text.
    """

    name: str
    source: str
```

- constraints:
  - No additional constraints.

### 4.2 `LinkableModule`

One target's pre-link translation unit.

```python
class LinkableModule:
    """One target's pre-link translation unit.

    Attributes:
        target: attribute; generator/linker backend label.
        language: attribute; source language.
        source: attribute; assembled translation-unit text.
        functions: attribute; constituent linkable functions in emission order.
    """

    target: str
    language: str
    source: str
    functions: tuple[LinkableFunction, ...] = field(default_factory=tuple)
```

- constraints:
  - `target` MUST be the generator/linker backend label (`"cuda"` or `"cpu"`
    for the built-ins), never an external Target registration name.
  - MUST be the source language: `cu` for a CUDA translation unit, `cpp` for a
    host translation unit.
  - MUST list the module's constituent `LinkableFunction`s, in emission order.

A `LinkableModule` is a build artifact, not a runtime object and not a
user-callable.

### 4.3 `LinkedModule`

The link output: a loadable artifact plus the host-visible metadata the loader
needs.

```python
class LinkedModule:
    """Linked library plus the entry's signature.

    Attributes:
        library_path: attribute; produced shared-library path.
        source: attribute; assembled host and device source.
        entry: attribute; the loaded entry's signature.
    """

    library_path: Path
    source: str
    entry: CallableSignature
```

- constraints:
  - MUST carry the assembled host + device source — the diagnostic source the
    runtime exposes as `RuntimeModule.source` ([runtime](./runtime.md)).

The `entry` signature is [§4.4](#44-signatures)'s `CallableSignature`: codegen
produces it and the runtime loader consumes it.

The link step consumes the per-target `LinkableModule`s, compiles each with its
own toolchain, and links them into one `LinkedModule`. `LinkedModule` is
consumed by the runtime loader ([runtime](./runtime.md)); the concrete compiler
commands are an implementation detail and not part of the contract.

### 4.4 Signatures

A signature is what one IR `Type` becomes once its function is compiled to
C++. The two layers differ: an IR function returns a value, where a C++ entry
writes into trailing parameters, and a call may carry ids the IR never
declared. So codegen states the C++ side as its own family, mirroring
`Type` ([types §7](./types.md#7-callabletype)).

```python
class Signature:
    """One parameter, or one whole call, as C++ spells it."""

class TensorSignature(Signature):
    """A declared tensor parameter."""

    name: str
    type: TensorType                # dtype / shape / storage / layout come from here

class ScalarSignature(Signature):
    """A parameter with no IR type: named and typed in C++ only."""

    name: str
    ctype: str

class TupleSignature(Signature):
    """The C++ side of a `TupleType`."""

    fields: tuple[Signature, ...] = ()

class UnitSignature(Signature):
    """The C++ side of `UnitType`: nothing is passed."""

class CallableSignature(Signature):
    """One calling convention of one IR function."""

    name: str
    params: tuple[Signature, ...] = ()      # what the IR declares, outputs last
    output_count: int = 0                   # trailing count of output parameters
    leading: tuple[Signature, ...] = ()     # hidden parameters ahead of `params`
    trailing: tuple[Signature, ...] = ()    # hidden parameters after `params`
```

- constraints:
  - `params` lists ALL declared parameters (inputs + outputs) in declaration
    order; `output_count` is the trailing count of outputs. `input_count` is
    `len(params) - output_count`; `input_params` / `output_params` are the
    corresponding slices; `all_params` is `leading + params + trailing`, the
    order the C++ declaration lists.
  - a `TensorSignature` reuses the IR type system instead of restating it: a
    dynamic dim is whatever `type.shape` carries (e.g. a `DimVar`), and there
    is no separate dynamic-dim sentinel.
  - a `ScalarSignature` has no IR counterpart on purpose: those parameters do
    not exist in the IR at all, which is what makes them hidden.
  - one compiled function has three C++ conventions — the host entry, the
    launch shim and the device kernel — and they differ only in `leading` and
    `trailing`. They MUST be three `CallableSignature` values; the class MUST
    NOT be subclassed.
  - every identifier codegen generates is spelled by `codegen/names.py` and
    carries the `tilefoundry_` prefix: the three symbols and the parameters
    codegen inserts. A user's own parameter names are never prefixed, and a
    name that reaches C++ through one of these is a plain C identifier -- a
    mangled variant's `$` is not one.
  - the entry the loader is handed ([§4.3](#43-linkedmodule)) is a fourth
    value of the same shape: its `name` is the exported symbol, and each
    `leading` entry is named for the topology level whose id the host must
    supply, because that caller reads the id out of a `Placement`
    ([shard §5](./shard.md#5-mesh)) instead of writing a C++ declaration.
