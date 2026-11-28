# ANE Runtime Spec

Scope: a linked ANE entry, instance-owned weight images and persistent model execution.

## 1. Load, forward, close

```python
class AneCompiledModule(runtime.CompiledModule):
    def __init__(self, type: codegen.CallableSignature, fn: Callable, library_path: Path) -> None: ...
    @classmethod
    def create(cls, tgt, mod: core.Module, options) -> AneCompiledModule: ...
    def load(self, resource: runtime.Resource, *, placement: types.Placement | None = None) -> None: ...
    def close(self) -> None: ...
```

- constraints:
  - `AneCompiledModule` is bound to `ane.Target` as its `runtime.CompiledModule`.
    `tf.build` and `tf.compile` of a lowered ANE entry MUST return it through `create`.
    `create` MUST emit the entry with the launcher, link it with clang++ and bind it.
  - Export and reload MUST use the core library pair
    ([runtime §1.1](../../../../../../docs/spec/runtime.md#11-runtimemodulepy)).
    `AneCompiledModule.from_library(path)` MUST read the weight names from the library.
  - The signature MUST contain only activations and trailing outputs. Weights MUST come from the Resource.
    Entry calls MUST retain the inherited positional calling convention.
  - Each successful load MUST bind its token to the entry.
  - A non-None placement MUST report `is not decoded yet` before replacing the current instance.
  - Load MUST unbind the entry and destroy the previous token before creating a candidate.
    It MUST read CPU fp16 weights and pass contiguous OIHW weights and bias vectors to the generated writers.
  - `NOT_PROVISIONED` MUST raise `ProvisionRequiredError` with owned image bytes and the absolute cache path.
    Every failed candidate MUST be destroyed and leave the node unloaded.
    Other load failures MUST identify the operation and status.
  - The library MUST NOT provision cache files or invoke privileged commands.
    The caller MUST provision externally and retry `load`.
  - Calls before load and call or load after close MUST fail.
    Close MUST clear the token before destruction and be idempotent.
    The caller MUST close a successfully loaded node.

## 2. FFI interface and native implementation

`ane_register` registers each entry and its lifecycle under the entry's dotted name
with `Function::SetGlobal(..., override=true)`, as core registers CPU entries
([runtime §1.4](../../../../../../docs/spec/runtime.md#14-runtimeloaderpy)).

| Registration | Arguments | Result |
|---|---|---|
| `<entry>` | `int64 token`, flattened activation and output tensors | Evaluate the loaded instance |
| `<entry>.create` | None | Positive `int64` token owning a validated skeleton copy |
| `<entry>.weights` | None | Array of the compiled weight names, in writer order |
| `<entry>.set_weight` | `int64 token`, String name, `DLTensor*` tensor | Write the compiled weight |
| `<entry>.load` | `int64 token` | Numeric `AneStatus`, including `NOT_PROVISIONED` |
| `<entry>.cache_path` | `int64 token` | String containing the daemon cache path |
| `<entry>.image` | `int64 token` | Owned Bytes containing the complete image |
| `<entry>.destroy` | `int64 token` | Unload and release the instance; zero is a no-op |

- constraints:
  - A node MUST capture all of its functions right after its library loads.
    A later registration of the same name MUST NOT change an existing node.
  - Each entry's `create` MUST reference its own immutable `AneEntry`.
  - The token table MUST own each `AneProgram` and allocate monotonically increasing positive integers without reuse.
    Unknown, negative or expired tokens MUST be rejected. Only destroy MUST accept zero.
  - Each library owns its token table. A node MUST use its token only with the functions it captured.
  - Native lookup, use and destruction MUST be synchronized so destruction cannot free an instance in use.
    A pointer MUST NOT be reinterpreted as a token. No pointer handle or custom C++ object MUST cross FFI.
  - Invalid weights MUST identify the required unloaded instance, compiled name, CPU fp16 dtype, contiguous OIHW or bias shape and bounded offsets.
  - Invalid activations and outputs MUST identify the parameter, CPU fp16 shape, nonnegative strides and bounded offsets.
  - Other native failures MUST identify the operation and status. Undecoded behavior MUST report `is not decoded yet`.
  - Internal `ane_image` MAY borrow bytes. The FFI operation MUST return an owned copy.
  - Destruction MUST attempt unloading before releasing a loaded model, including exception exits.
    C++ exceptions MUST NOT cross the internal C ABI.

Declarations live in `codegen/include/tilefoundry/runtime/ane/launcher.h`.
The `ane_token_*` helpers validate tokens before calling the internal C ABI.
`ane_create` copies the skeleton. `ane_set_weight` dispatches by compiled weight name.
Loaded instances reject weight mutation.
`ane_cache_path` hashes the complete image and combines it with the signing identifier, OS build and fixed suffix.
Python `Status` retains the numeric `AneStatus` values from the launcher header.
`ane_load` returns `ANE_NOT_PROVISIONED` when the daemon has no matching cache entry.
A successful load retains its model and client until destruction.
`ane_launch` allocates requests, copies host tensors and evaluates the retained model.

## 3. Validation boundary

Device tests MUST compare the assembled native image to the complete validated
`hwx.Image.to_bytes()` result before provisioning. Generated writers are also
compiled and exercised on CPU for single convolution, ReLU convolution and multiple
weight layouts, including complete-image equality through FFI and cache hash checks.
Device tests MUST compare auto-allocated and caller-provided outputs from two
successive calls to the same instance against the evaluator element by element.
The supported transfer and convolution geometries remain those in
[isa §9](./isa.md#9-verified-transfer-validation).
