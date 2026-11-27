# TileFoundry Spec — Runtime

This spec owns the Python runtime contract outside the IR compile pipeline
and the packaged CPU runtime header. Target packages own their device runtime
contracts.

## 1. Python Runtime Surface

The sections are one per file, and a file with more than one class states each
under its own heading. The script `scripts/runtime_spec_surface.py` reads each signature block out of that file. It writes the block into its `<!-- generated -->` region. A file's `__all__` is what the spec states. The spec states a docstring in its own words.

### 1.1 `runtime/module.py`

An ir `Module` is the semantic definition: its @func bodies run in the evaluator. A `runtime.Module` is the runtime instance: it holds kernel bodies. The two are twins. They have the same `name`, the same child tree, and the same `entry`. `@runtime.module` / `@runtime.func`
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
    this subclass mechanically from a semantic `Module`. This is the normal
    authoring path. A direct subclass is available for special cases
    (e.g. `CompiledModule`, below).
  - `load(resource, *, placement=None)` (base class) recurses into each child. A single slot uses `resource.subtree(key)`. Each indexed instance uses `resource.subtree(key).subtree(str(i))`. Every child receives the same *placement*. The base class itself resolves nothing. One process is one unit throughout the tree it loads. A child MUST NOT receive a placement of its own.
  - a twin with a sharded weight and no *placement* MUST refuse to read it.
    Nothing else in the system says which of the mesh's units this process
    holds. Returning the whole tensor would silently give every unit
    every other unit's data ([§1.10](#110-shardtensor)).
    Weight prefixes follow module paths, matching ir attribute addressing.
    Lifecycle: construct (structure) → `load` (remember source) → call. A
    `runtime.Module` does **not** run `prepare`; it loads straight from the
    directory the semantic side prepared ([§1.3](#13-runtimedecoratorpy)).
  - correspondence contract: the twin's `forward` and the semantic
    `evaluate(LoadedModule, ...)` MUST agree on the same inputs. `measure.check` judges this against stated bounds ([§1.9](#19-runtimemeasurepy)).
  - `module` names the authored `Module` that generated a twin. A caller that holds an implementation reaches the Module that judges it. A
    `runtime.Module` that stands for no single authored Module MUST report `None` rather than a chosen value. Examples are a compiled entry and a hand-written subclass. A caller that needs one MUST refuse instead of substituting.
  - `module` is therefore reserved on a twin. So are `bound`, `resource`,
    `placement`, `setup`, `to_local`, `from_local` and `weight_of`. A
    generated twin keeps its own state and helpers under these names. An authored `Module`
    MAY name a function, child or method `module`. A generated twin binds
    each of those as an attribute, which would shadow the field.
    `@runtime.module` MUST reject such a Module at decoration, so that no twin has a field that answers something else.
  - the base class holds no weights or resource.
  - a twin has two origins. A compiled twin comes from `tilefoundry.build` / `compile` / `jit` →
    `LinkedModule`. The loader binds a `CompiledModule` (a `runtime.Module`,
    not a `runtime.Function`) ([§1.4](#14-runtimeloaderpy)). A handwritten twin is a `@runtime.module`
    class (below). It loads from a prepared checkpoint directory via a
    `runtime.Resource` ([§1.8](#18-runtimeresourcepy)).

#### `CompiledModule`

<!-- generated: py-module-CompiledModule -->
```python
# src/tilefoundry/runtime/module.py
class CompiledModule(Module):
    library: object
    library_path: Path
    type: codegen.CallableSignature | None
    fn: Callable | None
    weights: dict[str, types.TensorType]
    sharded_weights: set[str]
    operations: dict[str, str]
    resource: Resource | None
    placement: types.Placement | None
    bound: dict[str, object]
    closed: bool

    def __init__(self, library, library_path, description) -> None: ...

    @classmethod
    def from_library(cls, path) -> CompiledModule: ...

    def entry_schema(self) -> tuple[str, ...]: ...

    def export_library(self, path) -> None: ...

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None: ...

    def close(self) -> None: ...

    def operation(self, name: str, *, expected: tuple[str, ...] | None = None): ...

    def unit_ids(self) -> tuple: ...

    def weight_of(self, name: str): ...

    def forward(self, *args): ...

    def alloc_outputs(self, args) -> tuple: ...
```
<!-- /generated -->

- constraints:
  - `name` and `entry` MUST retain the source root and entry names.
    `children` MUST be empty and `module` MUST be `None`.
  - The node MUST install only its entry as a `CompiledFunction` attribute.
    Other functions, children and Python methods MUST NOT become attributes.
  - Entry and child names MUST NOT shadow fields or methods declared by `Module`, `CompiledModule` or the selected subclass.
    Compilation MUST reject conflicts before native compilation. Loading MUST reject conflicts before attribute installation.
  - `from_library` MUST select the node class through the registered Target class and its existing `CompiledModule` binding.
    An unbound provider MUST use `CompiledModule`. An unknown provider MUST report `is not decoded yet`.
    Loading MUST NOT construct a Target or read hardware documents.
  - `from_library` MUST restore the root without IR or in-memory linked signatures.
    The restored node MUST start without resources, bound weights or placement.
  - `export_library` MUST copy only the compiled library.
    Copying onto the source file MUST be a no-op. Another existing destination MUST raise `FileExistsError`.
    Copy failure MUST preserve the original node's call behavior.
  - Default `CompiledModule.load` MUST replace the resource and placement and clear bound weights.
    Static root weights MUST be read lazily, checked for exact shape and dtype, and retained by the node.
    Default node calls MUST inject weights at their native parameter positions without dtype conversion.
    Target subclasses MAY bind weights through their documented native lifecycle.
  - A missing or unloaded weight in the default node MUST raise `KeyError` naming the root and weight.
    A shape or dtype mismatch MUST raise `ValueError`. A failed read MUST clear partial bindings.
    Dynamic or sharded weights MUST report `is not decoded yet`.
  - Public input arguments MUST exclude const weights and retain explicit scratch tensors.
    Calls MUST accept all inputs alone or all inputs followed by all outputs.
    Partial outputs or extra arguments MUST raise `TypeError`.
  - Automatic allocation MUST use declared static output shapes and dtypes and an input tensor's device.
    Dynamic outputs MUST require preallocated tensors. Automatic allocation MUST report `is not decoded yet` for dynamic outputs.
  - A single output MUST return a tensor. Zero or multiple outputs MUST return a tuple.
  - Leading unit ids MUST come from the placement supplied to `load`.
    Missing placement MUST raise `ValueError` when the entry requires unit ids.
  - `close` MUST clear resources, bound weights and the native binding. Repeated closure MUST be safe.
    Call or load after closure MUST raise `RuntimeError`. Closure MUST NOT force library unloading.
  - A root without an entry MUST remain exportable and reloadable.
    Calling it MUST report no entry and list its functions.
  - Missing `operation(name)` MUST identify the library, root and operation and report `is not decoded yet`.

Output shapes and dtypes come from the decoded host signature
([codegen §6](./codegen.md#6-self-describing-library)).

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


class CompiledFunction(Function):
    def __init__(self, owner, name, signature) -> None: ...

    def __call__(self, *args): ...
```
<!-- /generated -->

- constraints:
  - the base `__call__` raises; every usable body is a subclass. Agents may
    write any subclass whose `__call__` runs (torch or another implementation / …).
  - codegen produces the signature, and the runtime takes it as input. [codegen §4.4](./codegen.md#44-signatures) defines it. This spec MUST NOT redefine it.

`CompiledFunction` represents the compiled node's named entry.

- constraints:
  - `__call__` MUST delegate to the owner's current `forward` binding and accept positional arguments.
  - `__name__` MUST equal the entry name. `__qualname__` MUST be `<Module name>.<entry name>`.
  - `__signature__` MUST expose non-const native inputs, including scratch, with tensor type annotations.
    Trailing output parameters MUST be optional positional parameters. `__doc__` MAY be `None`.
  - The named callable MUST retain its owner and library.

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
access resolves each name as follows. A function becomes another callable `LoadedModule` whose `entry`
names that function. A single child becomes its reading. An indexed child becomes its
ordered tuple of readings. An orchestration method binds
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
    own (raw) names. `prepare` assembles a one-to-many alias here via
    `torch.stack`, its only reshaping. Run the converter through
    the evaluator. Then strictly validate the result's shape and dtype
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
    stays pure IR. One `Module` may be read any number of times, for example with two
    checkpoints or two devices. Each reading is independent of the others.
    A definition reached from two slots therefore yields one `LoadedModule` per
    instance rather than one binding the last instance wins. This is the semantic-side
    counterpart of the `runtime.Module` twin's own `load` ([§1.1](#11-runtimemodulepy)).
  - The caller supplies `evaluate(..., device=...)`. Every activation and every
    weight the reading uses MUST already be on that device. The runtime rejects a mismatch at the
    activation binding or the weight's first use. The diagnostic names the
    offending position or weight. Neither side moves tensors implicitly.
  - state belongs to the caller. A tensor that must survive across steps (e.g. a KV
    cache) is an ordinary `Tensor` param, passed in and returned. A step MUST
    NOT mutate a tensor it was given. Such a tensor shards by the same
    mechanism as any other tensor, its own `TensorType.layout`.
  - The runtime twin's `forward` runs a registered `forward` orchestration
    method (`Module.methods`) when present, else its entry `@func`. A multi-node
    composition is chained by the caller, one `forward` per node.
    On the runtime twin, a bare step MUST be refused when neither a `forward`
    method nor an entry is present. The refusal MUST name the functions and methods to call.
    It reports no `entry` as wrong.
    On the semantic side, `evaluate(loaded)` and `loaded(...)` run `entry` when
    one is declared; neither treats a `forward` method as an implicit default.
    A caller selects another HIR function or child through `loaded.<name>` and
    calls it, while a named orchestration method remains ordinary host Python.
  - Python entry into an HIR `Function` through the runtime twin's `forward`
    is the runtime event governed by [hir §1.1](./hir.md#11-function). A
    registered plain Python orchestration method stays on the host; every
    function it calls is a separate Python-to-HIR entry. An HIR-to-HIR `Call`
    remains inside the invocation. It MUST NOT surface as another runtime
    launch merely because its callee belongs to another Module.
  - a causal-LM root MAY define `init_caches`,
    `prepare_inputs_for_generation`, and `append_cache` orchestration methods.
    `prepare_inputs_for_generation(input_ids, step, caches, *, device)` receives
    a one-dimensional `torch.Tensor` of token IDs. The model selects the token
    at `step`, reshapes it and places it. It creates all other activation inputs in
    its own `forward` order. It returns that positional tuple. The caller owns
    the cache. The caller expands only the token-ID tensor. The caller MUST NOT reconstruct a
    model's positional, rotary, scaling, or state inputs. The caller passes the active
    token-ID prefix as a view. A method MUST NOT mutate that view. A method MUST NOT retain it
    across steps. These methods bind on the runtime twin in the same way as
    `forward`.

#### `@runtime.module` / `@runtime.func`


`@runtime.module(sem)` is a class decorator that takes the semantic `Module`
instance. It returns a `runtime.Module` subclass. Its instances are the
runtime twin of *sem*, with the same function names, the same child tree, and the same entry.
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
  - decoration-time validation is **strictly one-to-one**. The
    `@runtime.func` name set MUST equal `sem`'s function
    name set. The name set includes tagged methods and `runtime.Function`
    instance class attributes. Such an attribute is a heavy kernel that owns its own compilation
    state. It stands in for a `@runtime.func`. The child-attribute name set MUST equal `sem.children`'s
    key set. A name missing from either set MUST be rejected. A name extra in either set MUST be rejected. Each slot MUST
    match both its single/indexed form and its number of elements. A mismatch
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
  - the generated class reuses orchestration methods (`forward` / `init_caches` / …) from the
    semantic `Module.methods` verbatim. The runtime side never rewrites them.
    Inside them, `self.<fn>` / `self.<child>` resolve to the
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
API; only `CompiledModule` (a `runtime.Module`) is.
The loader reads and validates the root manifest once
([codegen §6](./codegen.md#6-self-describing-library)).
It resolves the declared symbol from that library and checks typed metadata against the declared ABI.
It decodes the host signature and constructs the selected node from the manifest alone.
Its `forward` implements the out-param calling convention directly ([§1.1](#11-runtimemodulepy)).
The node binds root weights through `load` ([§1.1](#11-runtimemodulepy)).

#### Launcher ABI

`tilefoundry.build(mod)` runs codegen, links the artifact, loads it and binds the entry.
Users call the result through `runtime.Module.__call__`.

- constraints:
  - A present entry MUST be a `PrimFunction` for `build`. An HIR entry MUST report that it requires lowering.
    A non-CPU entry MUST use the root Target. The target factory MUST accept or refuse its entry convention.
  - The selected node MUST check entry metadata against its own `entry_schema`.
    The default schema MUST contain unit ids followed by tensors. A target-owned token MUST NOT become a unit id.
  - The loader MUST call `tvm_ffi.load_module(path, keep_module_alive=True)`.
    TVM FFI MUST retain loaded libraries until process shutdown, including failed loads.
    Returned FFI objects and objects retained by exception tracebacks MAY use deleters in those libraries.
    Manifest conversion MUST copy string subclasses into ordinary Python strings.
  - Manifest and entry lookup MUST use `get_function(symbol, query_imports=False)` on the loaded library.
    Lookup MUST NOT fall back to process-global functions.
  - `runtime.Module.name` MUST use `root.name`. `runtime.Module.entry` MUST use `root.entry`, or `None` when omitted.
  - The typed callable MUST retain its owning library after external module references are released.
    Loading another library MUST NOT replace an existing callable.
  - Missing manifest, local symbol or metadata MUST raise `RuntimeError` naming the library and missing symbol.
    A missing manifest MUST require recompilation.
  - Malformed descriptions and mismatched ABI metadata MUST fail before a compiled object is returned.
    Errors MUST identify the library and invalid field or symbol.
    ABI validation MUST accept both TVM FFI schema encodings: `args` with the return type first,
    and `named_args` with separate `return` and `params` arrays.
    Return type, parameter count and parameter types MUST agree exactly with the node's entry schema.
    A mismatch MUST include the actual schema and expected return and parameter types.
  - Callable arguments MUST be DLPack-compatible tensors.
    `torch.Tensor` is one supported caller-side provider.

The exported function accepts flattened input/output tensor arguments. HIR
functions may be written as `Function(params) -> tensor`, but by the runtime
boundary the TIR/codegen surface is explicit input/output parameters.

Launch geometry (grid / block extents) is embedded in the generated host entry,
or an authored `launch(...)` supplies it. Metadata never carries it past
codegen.

A refused launch MUST raise `RuntimeError` from the call and identify the
kernel and device error. Execution faults may surface asynchronously at the
next device synchronization.

### 1.5 `runtime/placement.py`

<!-- generated: py-placement -->
```python
# src/tilefoundry/runtime/placement.py
def placement_from_env(topology: str = "gpu") -> types.Placement: ...
```
<!-- /generated -->

- constraints:
  - `placement_from_env(topology)` MUST return `Placement({topology: LOCAL_RANK})`.
    A launcher such as `torchrun` sets `LOCAL_RANK` before the script runs.
  - It MUST raise `ValueError` when `LOCAL_RANK` is missing, and when it is not an
    integer. The IR type `Placement` ([shard §5.1](./shard.md#51-placement)) does
    not read the environment.

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
) -> "runtime.CompiledModule": ...


def compile(
    mod: core.Module,
    /,
    *,
    target: tilefoundry_target.Target | None = None,
    options: Options | None = None,
) -> "runtime.CompiledModule": ...
```
<!-- /generated -->

```python
# src/tilefoundry/compiler/jit_cache.py: `jit` is the one instance of `Jit`
def jit(
    fn_or_mod, /, *, target: Target | None = None, options: Options | None = None, **kwargs
) -> "runtime.CompiledModule": ...


class Jit:
    def __call__(self, fn_or_mod, /, *, target=None, options=None, **kwargs) -> "runtime.CompiledModule": ...
    def cache_clear(self) -> None: ...
    def cache_info(self) -> dict: ...  # {"size": N}
```

`tilefoundry.jit(fn_or_mod, *, target=None, options=None)` is the JIT
entry point. It accepts a `core.Function` or `Module`, normalizes to a
`Module`, compiles with cache, and returns a `runtime.CompiledModule`.

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

**Cache**: the `jit` object holds one in-process dictionary.
The key includes entry text, effective topologies, options and the root export description.
The description distinguishes root names, entry names, tensor signatures and weight roles.
An empty root uses only its description and options.
`jit.cache_clear()` evicts; `jit.cache_info()` returns `{"size": N}`.

- constraints:
  - `compile` MUST capture the root declaration before running passes.
    Automatic host names MUST NOT replace the source entry name in the manifest.
  - `build` MUST snapshot the current lowered declaration and MUST NOT run passes.
  - A root without an entry and without children MUST produce a manifest-only library and skip execution passes.
    A root without an entry and with children MUST report `compiled module tree is not decoded yet`.
  - JIT MUST NOT create an automatic disk cache. Export and reload MUST remain explicit actions.
  - JIT MUST preserve the source declaration across pipeline mutations so repeated requests retain their key.

### 1.8 `runtime/resource.py`

Checkpoint aliasing is a base capability of every resource, not a wrapper
class. Both implementations below take an `alias={canonical: raw}` table.
The same lookup order resolves it.

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
  - `load(name)` returns the tensor for *name*. It raises `KeyError` if *name* is absent.
    It MUST raise `TypeError` (naming `load_group`) if *name* resolves to a
    tuple-valued (one-to-many) alias.
  - `load_group(name)` returns the tuple of raw tensors for a one-to-many
    alias entry (e.g. per-expert weight shards, in declared order).
    It returns `None` when *name* has no tuple-valued alias. That is the ordinary, one-to-one
    case.
  - `subtree(seg)` returns a view scoped under one more path *segment*.
    The resource alias-resolves `seg` itself. A child `runtime.Module` therefore addresses
    its own weights by their bare (unprefixed) name. `runtime.Module.__init__`
    never sees a dotted name.
  - the resource resolves names and reads tensors. It MUST NOT stack. It
    reshapes only where a `Preprocessed` alias entry states that the checkpoint
    stores that one tensor differently from how the Module declares it. Examples are a
    transpose, a slice of a fused tensor, or a dropped axis. Assembling a
    one-to-many group into one tensor is the job of `prepare` ([§1.3](#13-runtimedecoratorpy)), via
    `torch.stack`. The same holds for any value that is a function of more than one raw
    tensor. `Preprocessed` MUST resolve to one name. Construction rejects a tuple-valued name
    and names the weight converter as the way to express
    it. Precision is not preprocessing. Every read tensor keeps the element type
    the checkpoint stores. A declaration that requires another precision uses a
    weight converter. `load` validates the converted or raw result against
    the declaration ([§1.3](#13-runtimedecoratorpy)).

An `alias` entry renames one path *segment* or *leaf* **within the current
scope**. The resolved path is the caller's already-accumulated prefix, then the
resolved relative name. The lookup order is
a path-qualified key (`f"{prefix}{name}"`) first, then a bare `name` entry,
then identity (`f"{prefix}{name}"` unchanged). A bare entry therefore serves
every instance at that level uniformly (e.g. `{"gamma_kv": "kv_norm.weight"}`
fires under every layer). A per-instance name needs one literal entry per instance.
Examples are one real decoder layer and one per-expert shard group. A
`Preprocessed` value is a one-to-one leaf: `load` applies its `read` callable
to the raw tensor. A tuple value is the one-to-many group `load_group` reads.
A subtree segment MUST resolve to one relative path.
`subtree`'s own segment resolution rejects a tuple-valued or `Preprocessed` hit.

Aliasing therefore only ever reaches **downward**. A name resolved inside a
scope carries that scope's prefix. A node therefore cannot address a tensor its
parent owns. A checkpoint may store such a tensor there, such as a
layer-level norm weight that a child consumes. `Absolute(name)` is the escape. An
alias whose value is `Absolute` MUST resolve to `name` as the whole raw key.
No prefix joins onto it. It stays a leaf-only form. `load_group` reads
it as the one-to-one case and returns `None`. `subtree` MUST reject it in
the same shape as a tuple-valued hit. The reason is that a subtree segment must resolve
to one relative name.

- constraints:
  - `DictResource` is an in-memory / test double over a flat, dot-prefixed
    `{"layer0.w": tensor, ...}` mapping. `subtree` only extends the prefix
    that each `load` / `load_group` name joins onto. It carries `alias` down to
    every child view.
  - `SafetensorsResource` reads a safetensors checkpoint directory. `load` /
    `load_group` open at most one shard handle per shard file (mmap'd via
    `safetensors.safe_open`, shared across `subtree` views). They read only the
    requested tensor(s), placed on *device*. Two directory shapes MUST be
    accepted. The first has N shard files and a `model.safetensors.index.json`,
    whose `weight_map` names the shard that holds each key. The second has a single unsharded
    `model.safetensors` with no index, and its own key list is that map.
    A directory with
    neither MUST be reported as such rather than as a missing index.
    A raw key is cached weakly. Repeated reads return the same tensor while a
    caller holds it. The entry may disappear after the last reference is
    collected. Scoped views share the index, shard handles, and tensor cache.
  - The resource resolves *device* on every read. The shard handles and the
    tensor cache are both keyed by that resolved device alongside the name. A device name
    without an index means the process's currently selected device. A
    single-process multi-device host can change that device between reads.
  - `DrawnResource` draws a declared weight lazily with `draw_tensor`, keeps
    the result strongly, and shares its generator and drawing ledger across
    `subtree` views without reseeding. The strong ledger is a contract. Drawing
    the same name again would produce another random tensor. The two sides
    of a comparison would then receive different values. It has no alias
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
    (e.g. `forward`'s `(logits, past_key_values)`). Any other leaf raises
    `TypeError`. `check` flattens both results and MUST reject a candidate
    whose flattened structure, shape or dtype differs from the reference's.
    A returned `ShardTensor` is compared as the one unit's piece it holds;
    data of more than one unit raises `TypeError` ([§1.10](#110-shardtensor)).
  - every produced tensor MUST have exactly one non-empty list of predicates.
    A path that *expect* names and the check did not produce MUST be rejected. A result
    that flattens to nothing MUST be an error rather than a pass.
  - `reference=None` admits only predicates whose `needs_reference` is false; any
    other MUST be rejected. `ref_norm` is then absent, having nothing to measure.
  - a predicate whose `discrete` is false MUST be rejected on an integer or
    boolean output, naming exact comparison instead.
  - a measure can have no meaning at the values it was given. Examples are a relative
    distance against a zero reference and a direction between two zero vectors. Then the
    result MUST state what was measured instead through its `note`. It MUST NOT
    return a number whose scale is an artefact of a clamp.
  - `OutputCheck.passed` and `Report.passed` are read-only properties derived
    as `all` of their parts. Neither is accepted as a constructor field. A
    verdict therefore cannot disagree with the measurements printed beside it.
  - the check is not specific to `runtime.Module`. *candidate* / *reference* may be a
    `runtime.Module` bound method, a raw torch callable, or an evaluator
    closure. Anything callable on *inputs* is allowed.

### 1.10 `ShardTensor`

`Module.prepare` writes one canonical tensor per weight and the HIR keeps the
global `TensorType`, so a checkpoint read hands back every unit's data at
once. `ShardTensor` (in `tilefoundry.core`, exported as `tilefoundry.ShardTensor`)
pairs a global type with its data. The data has a global view, one global logical
tensor, or a local view, one piece per `Placement`. The evaluator holds its values in the same class
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
  - a global view carries no `Placement`: the unit is an argument to
    `to_local`, so one owning tensor serves every unit a host dispatches.
  - a local view holds one piece per `Placement`, each of `local_type(type)`;
    `to_local` returns the piece its placement holds.
  - a `type` with no `ShardLayout` is every unit's whole tensor, and so is
    a mesh axis the mesh only broadcasts over. A global `Partial` value is its
    reduced value, so `to_local` MUST refuse it: no unit's own part is left.
  - `to_local` on a global view is the host side of the CUDA runtime's
    `local_tensor`. It applies `mesh_coords` to the placement's unit ids. Then it applies
    `local_layout_and_offset` ([shard §7.6](shard.md#76-local_layout)) to the
    shard, with its layout replaced by the row-major layout of the same modes.
    Host data is logical, while the address stride of a private `Split` may be
    zero, so the address layout cannot select a piece of it.
  - `to_global` reads a local view by the meaning of its type. `Split` pieces
    sit side by side. Any `Broadcast` piece serves. `Partial` pieces combine
    by their reduction. A type with no `ShardLayout`, or `Partial` reductions
    that differ, MUST be refused.
  - a topology whose id is unfixed MUST be left undivided
    ([shard §5.1](shard.md#51-placement)).
  - a `runtime.Module`'s public boundary takes and returns `ShardTensor`, while
    a body -- a Python reference or a launched kernel alike -- only ever
    receives the local `torch.Tensor`. A returned tensor is already one
    unit's. It is returned as a local view, this unit's piece under the
    declared global type. The runtime never narrows it again.
  - a weight narrowed on first use is copied out of the read it came from.
    The whole canonical tensor is therefore released once every shard it contributes is
    held. A weight the mesh does not divide is kept as the read itself.
  - what remembers a read weight is keyed by name, unit ids and the current
    device together. One weight read on two cards is two tensors.

---

## 2. C++ Runtime Surface

CPU headers live under `src/tilefoundry/runtime/include/tilefoundry/runtime/cpu/`.
They are package data, located through `importlib.resources`, and retain their
include-relative paths. The script `scripts/runtime_spec_surface.py` reads each declaration block out of its header.
It writes the block into its `<!-- generated -->`
region. A signature is therefore stated once, where it is compiled. What follows a
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

**Terms.** The CPU target's launch boundary is raw pointers and shape scalars, so this header declares no surface.

- constraints:
  - The CPU target crosses the launch boundary as raw pointers and shape scalars, so nothing here is a type a caller names.
