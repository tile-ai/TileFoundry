"""ABI layer — ``CallableType`` / ``ParamABI`` / ``KernelInfo`` / ``LaunchConfig``,
and ``RuntimeFunction``, the callable wrapper that binds an implementation
(compiled out-param entry or a plain value-returning callable) to that ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tilefoundry.ir.types.shape_helpers import static_dim_value
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
    """Callable wrapper for a single function implementation.

    ``type.output_count`` selects the calling convention:
    - ``== 0``: *fn* is a value-returning callable (a handwritten torch/triton
      implementation) — ``__call__(*args)`` calls ``fn(*args)`` and returns
      its result directly; no out-param ABI applies.
    - ``> 0``: *fn* is a compiled out-param entry —
      - auto-alloc: ``fn(a)`` — allocates outputs, calls entry, returns output(s)
      - pre-alloc:  ``fn(a, out)`` — uses provided output tensor(s), returns output(s)
    """
    type: CallableType
    fn: Callable

    @property
    def signature(self) -> CallableType:
        """Deprecated alias for ``type`` (kept for backward compat)."""
        return self.type

    def __call__(self, *args):
        if self.type.output_count == 0:
            return self.fn(*args)
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
        self.fn(*args)
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
        self.fn(*all_args)
        n_out = len(outs)
        return outs[0] if n_out == 1 else tuple(outs)


def _abi_dim(dim) -> int:
    static = static_dim_value(dim)
    return static if static is not None else -1


def callable_type_of(fn) -> CallableType:
    """Derive a ``CallableType`` for a HIR ``Function``: one ``ParamABI`` per
    declared parameter (``output_count=0`` — a value-returning callable, per
    the ``RuntimeFunction`` convention above), mirroring the dynamic-dim rule
    ``codegen/cuda/emit.py::_param_abi`` uses for a lowered ``PrimFunction``
    (a static dim stays its ``int`` value; a dynamic dim becomes ``-1``).
    """
    params = tuple(
        ParamABI(
            name=var.name,
            dtype=var.type.dtype.name,
            shape=tuple(_abi_dim(s) for s in var.type.shape),
            storage=var.type.storage,
        )
        for var in fn.params
    )
    return CallableType(name=fn.name, params=params, output_count=0)


__all__ = [
    "CallableType",
    "KernelInfo",
    "LaunchConfig",
    "ParamABI",
    "RuntimeFunction",
    "callable_type_of",
]
