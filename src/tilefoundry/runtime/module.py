"""``RuntimeModule`` and supporting dataclasses.

type returned by ``tilefoundry.build`` / ``tilefoundry.compile``. It is directly
callable via ``rm(a)`` / ``rm(a, out)``.

Internal handoff: codegen links a ``LinkedModule`` (shared library + host-visible
ABI metadata); the runtime loader then constructs a ``RuntimeModule`` with
``RuntimeFunction`` objects wrapping the loaded callables.
"""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.storage import StorageKind

_lazy_dt_map: dict[str, object] | None = None


def _torch_dtype(name: str) -> object:
    global _lazy_dt_map
    if _lazy_dt_map is None:
        # noqa lazy: torch is an optional runtime dep — only required when
        # auto-allocating output tensors.
        import torch  # noqa: PLC0415
        _lazy_dt_map = {
            "f32": torch.float32,
            "f16": torch.float16,
            "bf16": torch.bfloat16,
            "i32": torch.int32,
            "i64": torch.int64,
        }
    if name not in _lazy_dt_map:
        raise TypeError(f"RuntimeFunction: unsupported dtype {name!r}")
    return _lazy_dt_map[name]


@dataclass(frozen=True)
class KernelInfo:
    """ABI of a single ``__global__`` kernel in the generated source."""
    name: str
    param_names: tuple[str, ...]


@dataclass(frozen=True)
class ParamABI:
    """One parameter of the host-visible entry function."""
    name: str
    dtype: str
    shape: tuple[int, ...]
    storage: StorageKind | None


@dataclass(frozen=True)
class CallableType:
    """Host-visible ABI for a function entry.

    ``params`` lists ALL parameters (inputs + outputs) in declaration order.
    ``output_count`` is the trailing count of output parameters.
    """
    name: str
    params: tuple[ParamABI, ...]
    output_count: int = 0

    @property
    def input_count(self) -> int:
        return len(self.params) - self.output_count

    @property
    def input_params(self) -> tuple[ParamABI, ...]:
        return self.params[:self.input_count]

    @property
    def output_params(self) -> tuple[ParamABI, ...]:
        return self.params[self.input_count:]


@dataclass(frozen=True)
class LaunchConfig:
    """CUDA kernel launch config — (grid_dims, block_dims) as dim3-shaped 3-tuples."""
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


@dataclass(frozen=True)
class RuntimeFunction:
    """Callable wrapper for a single compiled function.

    ``__call__(*args)`` supports two modes:
    - auto-alloc: ``fn(a)`` — allocates outputs, calls entry, returns output(s)
    - pre-alloc:  ``fn(a, out)`` — uses provided output tensor(s), returns output(s)
    """
    type: CallableType
    _entry: object  # tvm_ffi callable (internal)

    @property
    def signature(self) -> CallableType:
        """Deprecated alias for ``type`` (kept for backward compat)."""
        return self.type

    def __call__(self, *args):
        if len(args) == len(self.type.params):
            return self._call_pre_alloc(*args)
        elif len(args) == self.type.input_count:
            return self._call_auto_alloc(*args)
        else:
            raise TypeError(
                f"{self.type.name}: expected "
                f"{self.type.input_count} inputs (auto-alloc) or "
                f"{len(self.type.params)} inputs+outputs (pre-alloc), "
                f"got {len(args)}"
            )

    def _call_pre_alloc(self, *args):
        self._entry(*args)
        n_out = self.type.output_count
        outputs = args[-n_out:]
        return outputs[0] if n_out == 1 else outputs

    def _call_auto_alloc(self, *args):
        # noqa lazy: torch is an optional runtime dep (see _torch_dtype).
        import torch  # noqa: PLC0415
        device = None
        for a in args:
            if isinstance(a, torch.Tensor):
                device = a.device
                break
        if device is None:
            raise TypeError(
                f"{self.type.name}: cannot infer device for auto-alloc; "
                f"no torch.Tensor in inputs"
            )
        outs = []
        for p in self.type.output_params:
            dtype = _torch_dtype(p.dtype)
            outs.append(torch.empty(p.shape, dtype=dtype, device=device))
        all_args = tuple(args) + tuple(outs)
        self._entry(*all_args)
        n_out = len(outs)
        return outs[0] if n_out == 1 else tuple(outs)


@dataclass(frozen=True)
class RuntimeModule:
    """Result of ``tilefoundry.build(mod)`` / ``tilefoundry.compile(mod)``.

    Directly callable via ``__call__`` — delegates to the default entry function.
    """
    source: str
    kernels: tuple[KernelInfo, ...]
    launch_config: LaunchConfig
    functions: dict[str, RuntimeFunction]
    entry: str  # name of the default entry in ``functions``

    @property
    def entry_function(self) -> RuntimeFunction:
        return self.functions[self.entry]

    def __call__(self, *args):
        return self.entry_function(*args)

    @property
    def entry_callable(self) -> object:
        """Deprecated: use ``rm(a)`` or ``rm(a, out)`` instead.

        Returns the raw tvm_ffi entry callable for legacy/internal use only.
        """
        return self.entry_function._entry
