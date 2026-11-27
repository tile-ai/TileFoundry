# ANE Runtime Spec

Scope: a linked ANE entry, instance-owned weight images and persistent model execution.

## 1. Load, forward, close

`AneCompiledModule(runtime.CompiledModule)` is constructed from a linked library
path, a `codegen.CallableSignature` whose name is the dotted entry export name,
and the ordered resource weight names. Its signature MUST contain only activation
and trailing output parameters. Weights are loaded resources, never call arguments.
The constructor loads the library with `tvm_ffi.load_module(keep_module_alive=True)`
and captures all entry functions with `tvm_ffi.get_global_func`. Registrations allow
override; existing modules retain their captured callables and libraries remain
alive until process exit, following [runtime §1.4](../../../../../../docs/spec/runtime.md#14-runtimeloaderpy).
`tf.build`, `tf.compile`, export and reload integration are outside this interface.

The subclass overrides only `load` and adds `close`. `forward`, `__call__`, output
allocation, argument counts and result unwrapping follow
[runtime §1.1](../../../../../../docs/spec/runtime.md#11-runtimemodulepy).
Both `module(*activations)` and `module(*activations, *outputs)` MUST work.
A non-None placement MUST raise `ValueError` containing `is not decoded yet`
before replacing the current instance.

```text
load(resource, *, placement=None):
    unbind entry; destroy previous instance
    entry.create() -> private skeleton copy
    resource.load(name) -> CPU fp16 snapshot -> entry.set_weight(handle, name, tensor)
    entry.load(handle)
        missing cache -> ProvisionRequiredError(image, path); release instance
        provisioned  -> fn = partial(entry, handle)
forward(*args), inherited:
    allocate outputs or accept trailing outputs -> fn(*args, *outputs)
close():
    unbind entry; unload model; release instance; mark closed
```

The library MUST NOT provision cache files or invoke privileged commands.
`ProvisionRequiredError` carries complete image bytes and the absolute target path.
After external provisioning, the caller invokes `load(resource)` again.
Every load starts a new instance; a failed load leaves the module unloaded.
A call before loading or after close MUST fail; inherited argument validation may
reject a malformed call before reaching the bound entry. Close is idempotent.
The caller MUST close a successfully loaded module. Instance handles and image
copies belong to each Python module independently.

## 2. FFI interface and native implementation

Each compiled entry registers the following TVM FFI functions, using the core
dotted export name as its prefix:

| Registration | Arguments | Result |
|---|---|---|
| `<entry>` | handle, flattened activation and output tensors | Evaluate the loaded instance |
| `<entry>.create` | None | Handle owning a copy of the validated skeleton |
| `<entry>.set_weight` | handle, resource name, tensor | Write the compiled weight |
| `<entry>.load` | handle | Numeric `AneStatus`, including `NOT_PROVISIONED` |
| `<entry>.cache_path` | handle | Absolute daemon cache path |
| `<entry>.image` | handle | Owned bytes of the complete current image |
| `<entry>.destroy` | handle or None | Unload and free the instance |

Python uses these functions exclusively. Tensor arguments cross FFI as DLPack
compatible tensors. Invalid weights MUST name the required unloaded instance,
resource name, CPU fp16 dtype, contiguous OIHW shape and bounded offsets. Invalid
activations and outputs MUST identify the parameter and its required CPU fp16
shape, nonnegative strides and bounded offsets. Other native failures MUST name
the operation and status. An undecoded behavior MUST contain `is not decoded yet`.

The declarations live in `codegen/include/tilefoundry/runtime/ane/launcher.h`.
`AneEntry` is immutable internal metadata. Internal `ane_create` copies its
skeleton; `ane_set_weight` dispatches by resource name. Loaded instances reject
weight mutation. `ane_image` borrows bytes internally; the FFI wrapper copies them.
`ane_cache_path` hashes the complete instance image with SHA-256 and combines that
hash with the executable signing identifier, OS build and fixed daemon key suffix.
Python `Status` declares the numeric `AneStatus` values from the launcher header.

`ane_load` returns `ANE_NOT_PROVISIONED` when the daemon has no matching cache entry.
A successful load retains its model and client until `ane_destroy`; `ane_launch`
performs only request allocation, host copies and evaluation. Entry counts, tensor
pointers, device, dtype, shape and address arithmetic MUST be checked before copying.
Native ownership uses RAII for images, models, CF objects and IOSurfaces. Destruction
MUST attempt unloading before releasing a loaded model, including exception exits.
C++ exceptions MUST NOT cross the internal C ABI; FFI wrappers translate failures.
Python clears its handle before destruction and releases failed load candidates.

## 3. Validation boundary

Device tests MUST compare the assembled native image to the complete validated
`hwx.Image.to_bytes()` result before provisioning. Generated writers are also
compiled and exercised on CPU for single convolution, ReLU convolution and multiple
weight layouts, including complete-image equality through FFI and cache hash checks.
Device tests MUST compare auto-allocated and caller-provided outputs from two
successive calls to the same instance against the evaluator element by element.
The supported transfer and convolution geometries remain those in
[isa §9](./isa.md#9-verified-transfer-validation).
