# ANE Runtime Spec

Scope: a linked ANE entry, instance-owned weight images and persistent model execution.

## 1. Load, forward, close

`AneCompiledModule(runtime.Module)` is constructed from a linked library path and
entry symbol. Its public execution methods are `load(resource, *, placement=None)`,
`forward(*args)` and `close()`. A non-None placement MUST raise `ValueError` containing
`is not decoded yet` before replacing the current instance. `tf.build` integration
is outside this interface.

```text
load(resource, *, placement=None):
    destroy previous instance
    create entry -> private skeleton copy
    resource.load(name) -> CPU fp16 snapshot -> generated weight writer
    ane_load(instance)
        missing cache -> ProvisionRequiredError(image, path); release instance
        provisioned  -> retain loaded model
forward(*activations):
    validate inputs -> allocate outputs -> evaluate retained model
close():
    unload model -> release instance -> mark closed
```

The library MUST NOT provision cache files or invoke privileged commands.
`ProvisionRequiredError` carries complete image bytes and the absolute target path.
After external provisioning, the caller invokes `load(resource)` again.
Every load starts a new instance; a failed load leaves the module unloaded.
Forward before loading and any use after close MUST raise in Python.
Close is idempotent. The caller MUST close a successfully loaded module.

## 2. Native ABI

The declarations live in `codegen/include/tilefoundry/runtime/ane/launcher.h`.
The generated entry returns immutable `AneEntry` metadata. `ane_create` copies its
skeleton; `ane_set_weight` dispatches by resource name. Loaded instances reject
weight mutation. `ane_image` borrows the current bytes until mutation/destruction.
`ane_cache_path` hashes the complete instance image with SHA-256 and combines that
hash with the executable signing identifier, OS build and fixed daemon key suffix.

Python `Status` declares the numeric `AneStatus` values from the launcher header
in one place; callers and tests use its named members.

`ane_load` returns `ANE_NOT_PROVISIONED` when the daemon has no matching cache entry.
A successful load retains its model and client until `ane_destroy`; `ane_launch`
performs only request allocation, host copies and evaluation. Entry counts, tensor
pointers, device, dtype, shape and address arithmetic MUST be checked before copying.
Each instance owns its image and model independently.

Native ownership uses RAII for images, models, CF objects and IOSurfaces. Destruction
MUST attempt unloading before releasing a loaded model, including exception exits.
`ane_destroy` clears the caller's handle and reports `ANE_UNLOAD_FAILED` when unloading
fails. Other C ABI operations report their own failure; C++ exceptions MUST NOT cross
the ABI. Allocation and evaluation errors leave ownership with the caller, who destroys
the handle. Python load failures destroy their candidate instance before propagating.

## 3. Validation boundary

Device tests MUST compare the assembled native image to the complete validated
`hwx.Image.to_bytes()` result before provisioning. Generated writers are also
compiled and exercised on CPU for single convolution, ReLU convolution and multiple
weight layouts, including complete-image equality and cache hash checks.
The supported transfer and convolution geometries remain those in
[isa §9](./isa.md#9-verified-transfer-validation).
