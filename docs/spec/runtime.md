# TileFoundry Spec — Runtime

This spec owns the Python runtime contract outside the IR compile pipeline
and the packaged CPU runtime header. Target packages own their device runtime
contracts.

## 1. Python Runtime Surface

The sections are one per file, and a file with more than one class states each
under its own heading. Each signature block is read out of that file by
`scripts/runtime_spec_surface.py` and written into its `<!-- generated -->`
region; a file's `__all__` is what it states, and a docstring is the half the
spec says in its own words instead. Section numbers retain gaps where earlier
sections were consolidated under their owning files.

### 1.1 `runtime/module.py`

An ir `Module` (the semantic definition — @func bodies the evaluator runs) and
a `runtime.Module` (the runtime instance — kernel bodies) are twins: same
`name`, same child tree, same `entry`. `@runtime.module` / `@runtime.func`
(`tilefoundry.runtime.decorator`, below) build one twin mechanically from the
other, validated one-to-one at decoration time. The correspondence is held by
comparing the two numerically ([§1.9](#19-runtimemeasurepy)) against bounds the
comparison's caller states; a `runtime.Module` never runs the HIR evaluator.

#### `runtime.Module`

<!-- generated: py-module-Module -->
```python
# src/tilefoundry/runtime/module.py
class Module:
    name: str
    entry: str | None
    children: dict[str, Module | tuple[Module, ...]]
    module: "core.Module | None" = None

    def __init__(
        self,
        name: str,
        entry: str | None = None,
        children: dict[str, Module | tuple[Module, ...]] | None = None,
    ) -> None: ...

    def forward(self, *args): ...

    def __call__(self, *args): ...

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None: ...
```
<!-- /generated -->

- constraints:
  - the base class is authored like a `torch.nn.Module`: subclass it, build
    the child tree in `__init__` (children registered via `children=`), write
    the composition in `forward`. Function bodies are `runtime.Function`
    attributes called from `forward`. `@runtime.module` (below) generates
    this subclass mechanically from a semantic `Module` and is the normal
    authoring path; a direct subclass remains available for special cases
    (e.g. `CompiledModule`, below).
  - `load(resource, *, placement=None)` (base class): recurses into each child
    with `resource.subtree(key)` for a single slot, or
    `resource.subtree(key).subtree(str(i))` for each indexed instance, handing
    every one the same
    *placement*; the base class itself resolves nothing. One process is one
    program throughout the tree it loaded, so a child MUST NOT be given a
    placement of its own.
  - a twin with a sharded weight and no *placement* MUST refuse to read it.
    Nothing else in the system says which of the mesh's positions this process
    holds, and returning the whole tensor would silently give every program
    every other program's data ([§1.10](#110-shardtensor)).
    Weight prefixes follow module paths, matching ir attribute addressing.
    Lifecycle: construct (structure) → `load` (remember source) → call. A
    `runtime.Module` does **not** run `prepare`; it loads straight from the
    directory the semantic side prepared ([§1.3](#13-runtimedecoratorpy)).
  - correspondence contract: the twin's `forward` and the semantic
    `evaluate(LoadedModule, ...)` MUST agree on the same inputs, as judged by
    `measure.check` against stated bounds ([§1.9](#19-runtimemeasurepy)).
  - `module` names the authored `Module` a twin was generated from, so a
    caller holding an implementation can reach what it is judged against. A
    `runtime.Module` that stands for no single authored Module — a compiled entry,
    a hand-written subclass — MUST report `None` rather than something chosen
    for it, and a caller that needs one MUST refuse instead of substituting.
  - `module` is therefore reserved on a twin, as are `bound`, `resource`,
    `placement`, `setup`, `to_local`, `from_local` and `weight_of`, which a
    generated twin keeps its own state and helpers under. An authored `Module`
    MAY name a function, child or method `module`, and a generated twin binds
    each of those as an attribute, which would shadow the field.
    `@runtime.module` MUST reject such a Module when it is decorated, rather than
    generate a twin whose field answers something else.
  - the base class holds no weights or resource.
  - two origins: compiled — `tilefoundry.build` / `compile` / `jit` →
    `LinkedModule` → the loader binds a `CompiledModule` (a `runtime.Module`,
    not a `runtime.Function`) ([§1.4](#14-runtimeloaderpy)); handwritten — a `@runtime.module`
    class (below), loading from a prepared checkpoint directory via a
    `runtime.Resource` ([§1.8](#18-runtimeresourcepy)).

#### `CompiledModule`

<!-- generated: py-module-CompiledModule -->
```python
# src/tilefoundry/runtime/module.py
class CompiledModule(Module):
    def __init__(self, type: codegen.CallableSignature, fn: Callable) -> None: ...

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None: ...

    def program_ids(self) -> tuple: ...

    def forward(self, *args): ...

    def alloc_outputs(self, args) -> tuple: ...
```
<!-- /generated -->

`CompiledModule.forward(*args)` uses the out-param ABI (`type.output_count`
trailing params are outputs):

- **Auto-alloc**: `len(args) == type.input_count` — allocates the outputs on
  the device of the first `torch.Tensor` among the inputs, calls the entry,
  returns result(s). With no `torch.Tensor` input it raises `TypeError`.
- **Pre-alloc**: `len(args) == len(type.params)` — uses the provided output
  tensors and returns them. Any other argument count raises `TypeError`.
- **Return**: single output → bare tensor; multiple outputs → `tuple`.
- **Program ids**: when `type.leading` is non-empty, the ids of those topology
  levels, read from the `Placement` given to `load`, are passed ahead of the
  arguments; with no `Placement` the call raises `ValueError`.

Output shape and dtype come from `CallableSignature.output_params` (set by
codegen from lowered IR, never guessed at runtime). This convention is specific
to `CompiledModule`; other `runtime.Module.forward` implementations are not
bound by it.

### 1.2 `runtime/function.py`

`runtime.Function` is the **base class** for a node's function body; a body
subclasses it and overrides `__call__`. A handwritten torch or another implementation
implementation takes whatever it needs (converted weights, caches) at
construction and returns its value(s) directly. `runtime.Function.type` is the
`CallableSignature` codegen produced for that function
([codegen §4.4](./codegen.md#44-signatures)).

<!-- generated: py-function -->
```python
# src/tilefoundry/runtime/function.py
class Function:
    def __init__(self, type: codegen.CallableSignature) -> None: ...

    def __call__(self, *args): ...
```
<!-- /generated -->

- constraints:
  - the base `__call__` raises; every usable body is a subclass. Agents may
    write any subclass whose `__call__` runs (torch or another implementation / …).
  - the signature is codegen's product and the runtime's input: it is defined
    by [codegen §4.4](./codegen.md#44-signatures) and MUST NOT be redefined
    here.

### 1.3 `runtime/decorator.py`

<!-- generated: py-decorator -->
```python
# src/tilefoundry/runtime/decorator.py
def func(fn: Callable) -> Callable: ...


def module(sem: ir_core.Module) -> Callable[[type], type]: ...
```
<!-- /generated -->

A weight's converter is registered **per weight**, not per module:
`@<compute_fn>.converter("<weight_name>")` decorates a throwaway `def` and
registers it on the base function's `converters`
([parser §3](./parser.md#3-implementation-overview)). Its parameters are
the raw-checkpoint names, annotated like any `@func` parameter; it returns
exactly the one declared `ConstTensor`'s shape / dtype. A weight needing no
transform has no converter. Two converters registered for the same weight
name is an error.

A converter is an HIR body like any other and MAY reach another node's function.

`load` and `prepare` are methods on the IR `Module` owned by
[core-ir §1](./core-ir.md#1-module). Execution of a loaded reading is entered
through `interfaces.evaluator.evaluate`; this section defines that runtime-facing behavior
and the distinct loaded view:

```python
class LoadedModule:
    """Bind one reading of an IR Module to its resource."""

    module: core.Module
    resource: object  # a runtime.Resource
    children: dict[str, LoadedModule | tuple[LoadedModule, ...]]
    target: Target | None
    topologies: tuple[Topology, ...]
    path: str
```

`LoadedModule` is a frozen resource binding: it stores no tensors. Attribute
access resolves a function to another callable `LoadedModule` whose `entry`
names that function, a single child to its reading, an indexed child to its
ordered tuple of readings, and an orchestration method
with `types.MethodType`. Calling a loaded module enters `evaluate` once and
passes only activations; constants come from its resource at first use.

- constraints:
  - A reading covers its own Module's declared weights and holds one reading
    per instance in `Module.children` ([core-ir §1](./core-ir.md#1-module)).
    Calls follow [core-ir §2.1](./core-ir.md#21-call), selecting the corresponding
    loaded child for an instance target. Each loop trip selects its evaluated
    index independently. A runtime index outside `[0, N)` MUST be refused with
    the slot key, index, and N.
  - Loading and preparation MUST carry the effective context under
    [core-ir §1](./core-ir.md#1-module), storing it on each LoadedModule.
    Selecting a named function from a reading MUST re-enter with that stored
    context. Definitions and bodies remain shared; independent readings of the
    same definition MUST NOT borrow each other's resources.
  - `LoadedModule.path` records only the diagnostic instance path: each single
    slot adds its key, each indexed slot adds `key.i`, joined with dots. A missing
    weight diagnostic MUST retain `missing declared weight '<name>'` and include
    the complete path-plus-weight key, e.g. `layers.3.w`. This MUST NOT extend the
    runtime.Resource protocol.
  - `Module.prepare` (semantic side, offline, once): walk every instance,
    children before their parent, using the same key and index segments as load.
    Each instance MUST run its definition's converters independently against its
    own raw subtree. For each declared weight, fetch its converter's parameters by their
    own (raw) names — a one-to-many alias is assembled here via
    `torch.stack` (`prepare`'s only reshaping) — run the converter through
    the evaluator, then strictly validate the result's shape and dtype
    against the weight's declared `ConstTensor` type. A weight with no
    converter is validated the same way against its raw (or stacked) value,
    unchanged. Output: one safetensors shard plus
    `model.safetensors.index.json`, keyed by clean, dot-joined module paths
    (e.g. `layers.0.attention.w_kv`). Plain directory — no content-hash cache /
    manifest. The runtime twin never prepares; both twins `load` straight
    from the directory this writes.
  - `Module.load(resource)`: bind this node to `resource` and recurse into
    each slot under its key and optional index segments, returning a frozen
    `LoadedModule` tree. It MUST NOT write bindings onto the `Module`, which
    stays pure IR: one `Module` may be read any number of times — two
    checkpoints, two devices — and each reading is independent of the others.
    A definition reached from two slots therefore yields one `LoadedModule` per
    instance rather than one binding the last instance wins. This is the semantic-side
    counterpart of the `runtime.Module` twin's own `load` ([§1.1](#11-runtimemodulepy)).
  - The caller supplies `evaluate(..., device=...)`. Every activation and every
    weight used by the reading MUST already be on that device; a mismatch is
    rejected at the activation binding or weight's first-use point, naming the
    offending position or weight. Neither side moves tensors implicitly.
  - state is the caller's: a tensor that must survive across steps (e.g. a KV
    cache) is an ordinary `Tensor` param passed in and returned, and a step MUST
    NOT mutate one it was given. Sharding such a tensor is therefore the same
    mechanism as for any other — its own `TensorType.layout` — rather than a
    second description for an opaque state object.
  - The runtime twin's `forward` runs a registered `forward` orchestration
    method (`Module.methods`) when present, else its entry `@func`. A multi-node
    composition is chained by the caller, one `forward` per node.
    On the runtime twin, a bare step MUST be refused when neither a `forward`
    method nor an entry is present, naming the functions and methods to call
    instead of reporting `entry` as wrong.
    On the semantic side, `evaluate(loaded)` and `loaded(...)` run `entry` when
    one is declared; neither treats a `forward` method as an implicit default.
    A caller selects another HIR function or child through `loaded.<name>` and
    calls it, while a named orchestration method remains ordinary host Python.
  - Python entry into an HIR `Function` through the runtime twin's `forward`
    is the runtime event governed by [hir §1.1](./hir.md#11-function). A
    registered plain Python orchestration method stays on the host; every
    function it calls is a separate Python-to-HIR entry. An HIR-to-HIR `Call`
    remains inside the invocation and MUST NOT be surfaced as another runtime
    launch merely because its callee belongs to another Module.
  - a causal-LM root MAY define `init_caches`,
    `prepare_inputs_for_generation`, and `append_cache` orchestration methods.
    `prepare_inputs_for_generation(input_ids, step, caches, *, device)` receives
    a one-dimensional `torch.Tensor` of token IDs. The model selects the token
    at `step`, reshapes and places it, creates all other activation inputs in
    its own `forward` order, and returns that positional tuple. The caller owns
    the cache and expands only the token-ID tensor; it MUST NOT reconstruct a
    model's positional, rotary, scaling, or state inputs. It passes the active
    token-ID prefix as a view; a method MUST NOT mutate that view or retain it
    across steps. These methods bind on the runtime twin in the same way as
    `forward`.

#### `@runtime.module` / `@runtime.func`


`@runtime.module(sem)` is a class decorator taking the semantic `Module`
instance; it returns a `runtime.Module` subclass whose instances are *sem*'s
runtime twin — same function names, same child tree, same entry.
`@runtime.func` tags a plain method as a kernel body: same call signature as
the semantic `@func` of the same name, weight params included.

```python
# example
@runtime.module(attention_sem)
class Attention:
    @runtime.func                                    # weight params included, in the
    def mla_kv_update(self, hidden, gamma_kv, w_kv,  # semantic @func's own order
                      cos_pos, sin_pos, kv_cache0, cur_pos, s):
        ...  # a real kernel body, e.g. a torch or another implementation implementation

    moe = SomeMoeRuntimeClass  # a @runtime.module class, not an instance
```

- constraints:
  - decoration-time validation is **strictly one-to-one**: the
    `@runtime.func` name set (a tagged method, or a `runtime.Function`
    instance class attribute — a heavy kernel that owns its own compilation
    state, standing in for a `@runtime.func`) MUST equal `sem`'s function
    name set, and the child-attribute name set MUST equal `sem.children`'s
    key set; missing *or* extra either MUST be rejected. Each slot MUST
    match both its single/indexed form and its number of elements, and a mismatch
    diagnostic MUST identify the key and the semantic and runtime counts.
  - Twin child declarations follow the semantic slots
    ([core-ir §1.3](./core-ir.md#13-modulelist)).
    The generated constructor MUST build one runtime instance per semantic
    element, passing `ir=<element definition>` and inherited `topologies=`.
    Child classes MUST accept both constructor keywords. A child's explicit
    topology tuple replaces inheritance; an omitted declaration MUST inherit
    from its parent twin. Sharded input and weight projection MUST use this
    stored effective tuple rather than reading context from the shared definition.
  - weights are filled by name at call time from what `load` bound, so a
    kernel method's caller passes only activations.
  - orchestration methods (`forward` / `init_caches` / …) are reused from the
    semantic `Module.methods` verbatim and are never rewritten on the
    runtime side: inside them, `self.<fn>` / `self.<child>` resolve to the
    runtime twin's own kernels / children. Absent an own `forward`, the generated
    class runs `sem.methods["forward"]` if present, else calls the entry function
    by name.

### 1.4 `runtime/loader.py`

<!-- generated: py-loader -->
```python
# src/tilefoundry/runtime/loader.py
def load_linked_module(linked: "LinkedModule") -> CompiledModule: ...
```
<!-- /generated -->

```
Module (IR) → codegen: per-target LinkableModule… → LinkedModule (.so + metadata)
LinkedModule → load → CompiledModule (fully-loaded, public, callable runtime.Module)
```

`LinkedModule` is a codegen product
([codegen §4.3](./codegen.md#43-linkedmodule)); the loader that turns it into a
`CompiledModule` is owned here. The loader and `LinkedModule` are not public
API; only `CompiledModule` (a `runtime.Module`) is. `load_linked_module` returns
`CompiledModule(type=replace(entry.signature, name=entry.name), fn=entry_callable)`,
where `entry` is the first `LinkedFunction` (codegen sorts the `Module.entry`
host first); `name` / `entry` are its registration name. Its `forward`
implements the out-param calling convention directly
([§1.1](#11-runtimemodulepy)). The compiled path has no `resource` / `weights`
(weights are ordinary entry arguments), so `load` only records the
`Placement` and recurses into no children.

#### Launcher ABI

`tilefoundry.build(mod)` runs codegen, links the artifact (see
[codegen](./codegen.md)), loads it and binds the entry; it MUST raise
`ValueError` when `Module.entry` is not a CPU host `PrimFunction`. Users interact
only with `runtime.Module.__call__` / `runtime.Function.__call__`.

Load contract:

- codegen produces the `LinkedModule` artifact
  ([codegen §4.3](./codegen.md#43-linkedmodule))
- loading uses `tvm_ffi.load_module(...)`
- immediately after loading, entry binding uses
  `tvm_ffi.get_global_func(entry.name)`; `runtime.Module.entry` is this
  dotted registration name
- the loader keeps every library it loads until process exit, because the
  global registration table does not keep the shared library loaded
- host registration allows override: another load replaces the global entry,
  while an existing `CompiledModule` retains its own callable
- a library that lacks the entry's registration is a `RuntimeError` naming the
  library and the entry
- callable arguments are DLPack-compatible tensors; `torch.Tensor` is one
  supported caller-side provider but is not the semantic contract itself

Generated host wrappers register entry functions with TVM FFI:

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::Function::SetGlobal(export_name,
      tvm::ffi::Function::FromTyped(entry_function, export_name), /*override=*/true);
}
```

The exported function accepts flattened input/output tensor arguments. HIR
functions may be written as `Function(params) -> tensor`, but by the runtime
boundary the TIR/codegen surface is explicit input/output parameters.

Launch geometry (grid / block extents) is embedded into the generated host entry
or supplied by an authored `launch(...)`; it is never carried as metadata past
codegen.

A refused launch MUST raise `RuntimeError` from the call and identify the
kernel and device error. Execution faults may surface asynchronously at the
next device synchronization.

### 1.6 `compiler/`

<!-- generated: py-compile-options -->
```python
# src/tilefoundry/compiler/options.py
@dataclass(frozen=True)
class Options:
    target: tilefoundry_target.Target
    passes: passes.Options = passes.Options()
    codegen: codegen.Options = codegen.Options()
    dump: dump.Flags = dump.Flags.ALL
```
<!-- /generated -->

<!-- generated: py-compile -->
```python
# src/tilefoundry/compiler/compilation.py
def normalize_to_module(fn_or_mod: core.Function | core.Module) -> core.Module: ...


def resolve_options(
    operation: str,
    mod: core.Module,
    target: tilefoundry_target.Target | None,
    options: Options | None,
) -> Options: ...


def build(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.Module": ...


def compile(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.Module": ...
```
<!-- /generated -->

```python
# src/tilefoundry/compiler/jit_cache.py: `jit` is the one instance of `Jit`
def jit(
    fn_or_mod, /, *, target: Target | None = None, options: Options | None = None, **kwargs
) -> "runtime.Module": ...


class Jit:
    def __call__(self, fn_or_mod, /, *, target=None, options=None, **kwargs) -> "runtime.Module": ...
    def cache_clear(self) -> None: ...
    def cache_info(self) -> dict: ...  # {"size": N}
```

`tilefoundry.jit(fn_or_mod, *, target=None, options=None)` is the JIT
entry point. It accepts a `core.Function` or `Module`, normalizes to a
`Module`, compiles with cache, and returns a callable `runtime.Module`.

**Input contract**:
- Only TileFoundry IR objects (`Function` / `Module`) are accepted; anything
  else, including a raw Python function, raises `TypeError` — use `@func`
  first. An unexpected keyword raises `TypeError`.
- A stated `target` MUST be a constructed Target instance. Strings
  MUST be refused and MUST NOT select a backend or construct a Target.
- A Module-owned Target is retained as the exact instance; a conflicting
  explicit Target or `Options.target` MUST fail. With no Module Target the
  Target is the explicit one, else `Options.target`, else the host CPU
  (`resolve_options`).
- Topology is declared by the `Module`; a single-function
  `@func(topologies=...)` declares it through the implicit `Module` that
  decorator yields ([parser §2.1](./parser.md#21-syntax)).
- Mesh layout is expressed with lexical `with Mesh(...) as mesh` scopes;
  `jit()` has no `cta_mesh` / `thread_mesh` parameters.

**Pipeline**: `jit()` runs `compile()` ([passes §6](./passes.md#6-top-level-api)).
A bare `Function` input is wrapped into a single-function `Module` that
declares no execution context.

**Cache**: in-process dict cache, held by the `jit` object, keyed by
`sha256(canonical_module_text + "\0" + repr(options))`; the `repr` of the frozen
`compiler.Options` names the Target and every stage's option.
`canonical_module_text` includes functions, declared topology hierarchies, and
`with Mesh` scopes. Independently compiled Modules declare their execution
context ([core-ir §1](./core-ir.md#1-module)). No Python object identity
participates in the key.
`jit.cache_clear()` evicts; `jit.cache_info()` returns `{"size": N}`.

### 1.8 `runtime/resource.py`

Checkpoint aliasing is a base capability of every resource, not a wrapper
class: both implementations below take an `alias={canonical: raw}` table,
resolved by the same lookup order.

<!-- generated: py-resource -->
```python
# src/tilefoundry/runtime/resource.py
@dataclasses.dataclass(frozen=True)
class Absolute:
    name: str


@dataclasses.dataclass(frozen=True)
class Preprocessed:
    name: "str | Absolute"
    read: Convert


def resolved_device(device: str) -> str: ...


class Resource(Protocol):
    def load(self, name: str) -> utils.torch.Tensor: ...

    def load_group(self, name: str) -> "tuple[utils.torch.Tensor, ...] | None": ...

    def subtree(self, seg: str) -> "Resource": ...


class DictResource:
    def __init__(
        self,
        data: Mapping[str, utils.torch.Tensor],
        prefix: str = "",
        alias: AliasMap | None = None,
    ) -> None: ...

    def load(self, name: str) -> utils.torch.Tensor: ...

    def load_group(self, name: str) -> "tuple[utils.torch.Tensor, ...] | None": ...

    def subtree(self, seg: str) -> "DictResource": ...


class SafetensorsResource:
    def __init__(
        self, ckpt_dir: str, prefix: str = "", *, device: str, alias: AliasMap | None = None
    ) -> None: ...

    def share_cache_of(self, parent: "SafetensorsResource") -> None: ...

    def index(self) -> dict[str, str]: ...

    def read_one(self, raw_key: str) -> utils.torch.Tensor: ...

    def load(self, name: str) -> utils.torch.Tensor: ...

    def load_group(self, name: str) -> "tuple[utils.torch.Tensor, ...] | None": ...

    def subtree(self, seg: str) -> "SafetensorsResource": ...


class DrawnResource:
    def __init__(
        self,
        module: core.Module | tuple[core.Module, ...],
        generator: utils.torch.Generator,
        device: str,
        drawn: dict[str, utils.torch.Tensor] | None = None,
        prefix: str = "",
    ) -> None: ...

    def load(self, name: str) -> utils.torch.Tensor: ...

    def load_group(self, name: str) -> None: ...

    def subtree(self, seg: str) -> "DrawnResource": ...


def draw_tensor(
    declared: types.TensorType, generator: utils.torch.Generator, device: str
) -> utils.torch.Tensor: ...
```
<!-- /generated -->

- constraints:
  - `Preprocessed` is a frozen dataclass carrying one raw name and its one-tensor
    read transform.
  - `load(name)` returns the tensor for *name*; raises `KeyError` if absent,
    and MUST raise `TypeError` (naming `load_group`) if *name* resolves to a
    tuple-valued (one-to-many) alias.
  - `load_group(name)` returns the tuple of raw tensors for a one-to-many
    alias entry (e.g. per-expert weight shards, in declared order), or
    `None` when *name* has no tuple-valued alias — the ordinary, one-to-one
    case.
  - `subtree(seg)` returns a view scoped under one more path *segment*
    (`seg` is itself alias-resolved), so a child `runtime.Module` addresses
    its own weights by their bare (unprefixed) name — `runtime.Module.__init__`
    never sees a dotted name.
  - the resource resolves names and reads tensors. It MUST NOT stack, and it
    reshapes only where a `Preprocessed` alias entry states that the checkpoint
    stores that one tensor differently from how the Module declares it — a
    transpose, a slice of a fused tensor, or a dropped axis. Assembling a
    one-to-many group into one tensor stays `prepare`'s job ([§1.3](#13-runtimedecoratorpy)), via
    `torch.stack`, and so does any value that is a function of more than one raw
    tensor. `Preprocessed` MUST resolve to one name: a tuple-valued name is
    rejected at construction, naming the weight converter as the way to express
    it. Precision is not preprocessing: every read tensor keeps the element type
    the checkpoint stores, a declaration requiring another precision uses a
    weight converter, and `load` validates the converted or raw result against
    the declaration ([§1.3](#13-runtimedecoratorpy)).

An `alias` entry renames one path *segment* or *leaf* **within the current
scope**, joined onto the caller's already-accumulated prefix: lookup order is
a path-qualified key (`f"{prefix}{name}"`) first, then a bare `name` entry,
then identity (`f"{prefix}{name}"` unchanged). A bare entry therefore serves
every instance at that level uniformly (e.g. `{"gamma_kv": "kv_norm.weight"}`
fires under every layer); a per-instance name (one real decoder layer, one
per-expert shard group) needs one literal entry per instance instead. A
`Preprocessed` value is a one-to-one leaf: `load` applies its `read` callable
to the raw tensor. A tuple value is the one-to-many group `load_group` reads;
`subtree`'s own segment resolution rejects a tuple-valued or `Preprocessed` hit
(a subtree segment MUST resolve to one relative path).

Aliasing therefore only ever reaches **downward**: a name resolved inside a
scope carries that scope's prefix, so a node cannot address a tensor its
parent owns — and a checkpoint may well store one there, such as a
layer-level norm weight a child consumes. `Absolute(name)` is the escape: an
alias whose value is `Absolute` MUST resolve to `name` as the whole raw key,
with no prefix joined onto it. It stays a leaf-only form — `load_group` reads
it as the one-to-one case and returns `None`, and `subtree` MUST reject it in
the same shape as a tuple-valued hit, because a subtree segment must resolve
to one relative name.

- constraints:
  - `DictResource` — in-memory / test double over a flat, dot-prefixed
    `{"layer0.w": tensor, ...}` mapping; `subtree` only extends the prefix
    each `load` / `load_group` name is joined onto, carrying `alias` down to
    every child view.
  - `SafetensorsResource` — reads a safetensors checkpoint directory; `load` /
    `load_group` open at most one shard handle per shard file (mmap'd via
    `safetensors.safe_open`, shared across `subtree` views) and read only the
    requested tensor(s), placed on *device*. Two directory shapes MUST be
    accepted: N shard files with a `model.safetensors.index.json` whose
    `weight_map` names the shard holding each key, and a single unsharded
    `model.safetensors` with no index, whose own key list is that map — a
    published checkpoint is only sharded once it outgrows the writer's limit,
    so requiring an index would refuse the small ones. A directory with
    neither MUST be reported as such rather than as a missing index.
    A raw key is cached weakly: repeated reads return the same tensor while a
    caller holds it, and the entry may disappear after the last reference is
    collected; scoped views share the index, shard handles, and tensor cache.
  - *device* is resolved on every read, and both the shard handles and the
    tensor cache are keyed by that resolved device alongside the name. A device name
    without an index means the process's currently selected device, which a
    single-process multi-device host can change between reads; a handle keyed by
    file alone would stay pinned to the card that first opened it, and a tensor
    keyed by name alone would hand a reader on one card what another card
    holds.
  - `DrawnResource` draws a declared weight lazily with `draw_tensor`, keeps
    the result strongly, and shares its generator and drawing ledger across
    `subtree` views without reseeding. The strong ledger is a contract: drawing
    the same name again would produce another random tensor, so the two sides
    of a comparison would no longer receive the same value. It has no alias
    groups, so `load_group` always returns `None`.

### 1.9 `runtime/measure.py`

<!-- generated: py-measure -->
```python
# src/tilefoundry/runtime/measure.py
@dataclass(frozen=True)
class Predicate:
    name: ClassVar[str] = ""
    bounds: ClassVar[tuple[str, ...]] = ()
    needs_reference: ClassVar[bool] = True
    discrete: ClassVar[bool] = True
    guidance: ClassVar[str] = ""

    def measure(
        self, candidate: utils.torch.Tensor, reference: utils.torch.Tensor | None
    ) -> "PredicateResult": ...

    def result(
        self, values: Mapping[str, float], passed: bool, note: str | None = None
    ) -> "PredicateResult": ...


@dataclass(frozen=True)
class PredicateResult:
    predicate: Predicate
    values: Mapping[str, float]
    passed: bool
    note: str | None = None


@dataclass(frozen=True)
class AllClose(Predicate):
    atol: float
    rtol: float

    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class RelL2(Predicate):
    max: float

    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class Cosine(Predicate):
    min: float

    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class Equal(Predicate):
    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class Ulp(Predicate):
    max: float

    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class MaxAbs(Predicate):
    max: float

    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class MaxRel(Predicate):
    max: float

    def measure(self, candidate, reference): ...


@dataclass(frozen=True)
class NanInf(Predicate):
    def measure(self, candidate, reference=None): ...


PREDICATES: Mapping[str, type[Predicate]] = MappingProxyType(
    {
        predicate.name: predicate
        for predicate in (AllClose, RelL2, Cosine, Equal, Ulp, MaxAbs, MaxRel, NanInf)
    }
)


@dataclass(frozen=True)
class OutputCheck:
    path: str
    shape: tuple[int, ...]
    dtype: str
    ref_norm: float | None
    results: tuple[PredicateResult, ...]

    @property
    def passed(self) -> bool: ...


@dataclass(frozen=True)
class Report:
    outputs: tuple[OutputCheck, ...]

    @property
    def passed(self) -> bool: ...


def flatten_outputs(x, path: str = "output") -> list[tuple[str, utils.torch.Tensor]]: ...


def check(
    candidate: Callable,
    reference: Callable | None,
    inputs: tuple,
    *,
    expect: Mapping[str, Sequence[Predicate]],
) -> Report: ...
```
<!-- /generated -->

- constraints:
  - `PREDICATES` MUST contain the built-in `allclose`, `rel_l2`, `cosine`,
    `equal`, `ulp`, `max_abs`, `max_rel`, and `nan_inf` predicates by name.
  - `check` runs `candidate(*inputs)`, and `reference(*inputs)` when there is a
    reference, and measures each output against the predicates *expect* states
    for it. Neither *reference* nor *expect* has a default.
  - `inputs` is passed positionally to both callables.
  - a result MUST be a bare tensor or an arbitrarily nested tuple of tensors
    (e.g. `forward`'s `(logits, past_key_values)`); any other leaf raises
    `TypeError`. `check` flattens both results and MUST reject a candidate
    whose flattened structure, shape or dtype differs from the reference's.
    A returned `ShardTensor` is compared as the one program's piece it holds;
    data of more than one program raises `TypeError` ([§1.10](#110-shardtensor)).
  - every produced tensor MUST have exactly one non-empty list of predicates,
    and a path *expect* names that was not produced MUST be rejected. A result
    that flattens to nothing MUST be an error rather than a pass.
  - `reference=None` admits only predicates whose `needs_reference` is false; any
    other MUST be rejected. `ref_norm` is then absent, having nothing to measure.
  - a predicate whose `discrete` is false MUST be rejected on an integer or
    boolean output, naming exact comparison instead.
  - where a measure has no meaning at the values it was given — a relative
    distance against a zero reference, a direction between two zero vectors — the
    result MUST state what was measured instead through its `note`, rather than
    return a number whose scale is an artefact of a clamp.
  - `OutputCheck.passed` and `Report.passed` are read-only properties derived
    as `all` of their parts; neither is accepted as a constructor field, so a
    verdict cannot disagree with the measurements printed beside it.
  - it is not specific to `runtime.Module`: *candidate* / *reference* may be a
    `runtime.Module` bound method, a raw torch callable, or an evaluator
    closure — anything callable on *inputs*.

### 1.10 `ShardTensor`

`Module.prepare` writes one canonical tensor per weight and the HIR keeps the
global `TensorType`, so a checkpoint read hands back every program's data at
once. `ShardTensor` (in `tilefoundry.core`, exported as `tilefoundry.ShardTensor`)
pairs a global type with its data in a global view -- one global logical
tensor -- or a local view -- one piece per `Placement`. The two names are
OneFlow's. The evaluator holds its values in the same class
([evaluator §1](evaluator.md#1-value)).

<!-- generated: py-tensor -->
```python
# src/tilefoundry/core/tensor.py
def local_type(type: types.TensorType) -> types.TensorType: ...


@dataclass(frozen=True)
class ShardTensor:
    type: types.TensorType
    tensor: "utils.torch.Tensor | tuple[tuple[types.Placement, utils.torch.Tensor], ...]"

    @property
    def is_global(self) -> bool: ...

    @property
    def placements(self) -> tuple[types.Placement, ...]: ...

    def to_local(self, placement: "types.Placement | None") -> "utils.torch.Tensor": ...

    def to_global(self) -> "ShardTensor": ...

    def reshard(self, type: types.TensorType) -> "ShardTensor": ...

    def window(self, whole, shard: types.ShardLayout, placement: types.Placement): ...
```
<!-- /generated -->

- constraints:
  - a global view carries no `Placement`: the program is an argument to
    `to_local`, so one owning tensor serves every program a host dispatches.
  - a local view holds one piece per `Placement`, each of `local_type(type)`;
    `to_local` returns the piece its placement holds.
  - a `type` with no `ShardLayout` is every program's whole tensor, and so is
    an axis the mesh only broadcasts over. A global `Partial` value is its
    reduced value, so `to_local` MUST refuse it: no program's own part is left.
  - `to_local` on a global view is the host side of the CUDA runtime's
    `local_tensor`: `mesh_coords` of the placement's program ids, then
    `local_layout_and_offset` ([shard §7.6](shard.md#76-local_layout)) of the
    shard with its layout replaced by the row-major layout of the same modes.
    Host data is logical, while the address stride of a private `Split` may be
    zero, so the address layout cannot select a piece of it.
  - `to_global` reads a local view by the meaning of its type: `Split` pieces
    sit side by side, any `Broadcast` piece serves, and `Partial` pieces combine
    by their reduction. A type with no `ShardLayout`, or `Partial` reductions
    that differ, MUST be refused.
  - a level whose id is unfixed MUST be left undivided
    ([shard §5.1](shard.md#51-placement)).
  - a `runtime.Module`'s public boundary takes and returns `ShardTensor`, while
    a body -- a Python reference or a launched kernel alike -- only ever
    receives the local `torch.Tensor`. A returned tensor is already one
    program's, so it is returned as a local view: this program's piece under the
    declared global type, never narrowed again.
  - a weight narrowed on first use is copied out of the read it came from, so
    the whole canonical tensor is released once every shard it contributes is
    held. A weight the mesh does not divide is kept as the read itself.
  - what remembers a read weight is keyed by name, program ids and the current
    device together. One weight read on two cards is two tensors, and a key of
    the name alone hands a reader on one card what another card holds.

---

## 2. C++ Runtime Surface

CPU headers live under `src/tilefoundry/runtime/include/tilefoundry/runtime/cpu/`.
They are package data, located through `importlib.resources`, and retain their
include-relative paths. Each declaration block is read out of its header by
`scripts/runtime_spec_surface.py` and written into its `<!-- generated -->`
region, so a signature is stated once, where it is compiled. What follows a
region is written by hand: the terms the declarations are spelled in, and the
decisions two reasonable implementations would differ on.

### 2.1 `cpu/`

#### 2.1.1 `cpu/runtime.h`

<!-- generated: cpu-runtime -->
```text
// src/tilefoundry/runtime/include/tilefoundry/runtime/cpu/runtime.h
// This header declares no surface of its own.
```
<!-- /generated -->

**Terms.** The CPU target's launch boundary is raw pointers and shape scalars, so this header declares no surface. Host-side tensor compute is future work.

- constraints:
  - The CPU target crosses the launch boundary as raw pointers and shape scalars, so nothing here is a type a caller names.
