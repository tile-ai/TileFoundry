"""ABI layer — ``CallableType`` / ``ParamABI`` / ``KernelInfo`` / ``LaunchConfig``,
``RuntimeFunction`` (the implementation base class), and ``CompiledFunction``
(the compiled out-param entry implementation the loader binds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.storage import StorageKind


@dataclass(frozen=True)
class KernelInfo:
    """ABI of a single ``__global__`` kernel in the generated source."""
    name: str
    param_names: tuple[str, ...]


@dataclass(frozen=True)
class ParamABI:
    """One parameter of a host-visible entry: name, element type, shape
    (a static dim is its ``int`` value, a dynamic dim is ``-1``), storage."""
    name: str
    dtype: DType
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


class RuntimeFunction:
    """Implementation base class: an ABI ``type`` plus a subclass-overridden
    ``__call__``. A handwritten torch / triton / CUDA implementation subclasses
    this, takes whatever it needs (weights, caches) at construction, and
    returns its value(s) directly from ``__call__``.
    """

    def __init__(self, type: CallableType) -> None:
        self.type = type

    def __call__(self, *args):
        raise NotImplementedError(
            f"RuntimeFunction {self.type.name!r}: subclass must implement __call__()"
        )


class CompiledFunction(RuntimeFunction):
    """A compiled out-param entry (bound by the runtime loader).

    ``type.output_count`` trailing params are outputs:
    - auto-alloc: ``fn(a)`` — allocates outputs, calls the entry, returns them
    - pre-alloc:  ``fn(a, out)`` — uses the provided output tensor(s)
    """

    def __init__(self, type: CallableType, entry: Callable) -> None:
        super().__init__(type)
        self.entry = entry

    def __call__(self, *args):
        if len(args) == len(self.type.params):
            return self._call_pre_alloc(*args)
        elif len(args) == self.type.input_count:
            return self._call_auto_alloc(*args)
        raise TypeError(
            f"{self.type.name}: expected "
            f"{self.type.input_count} inputs (auto-alloc) or "
            f"{len(self.type.params)} inputs+outputs (pre-alloc), "
            f"got {len(args)}"
        )

    def _call_pre_alloc(self, *args):
        self.entry(*args)
        n_out = self.type.output_count
        outputs = args[-n_out:]
        return outputs[0] if n_out == 1 else outputs

    def _call_auto_alloc(self, *args):
        # noqa lazy: torch is an optional runtime dep, needed only for alloc.
        import torch  # noqa: PLC0415

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415

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
            outs.append(torch.empty(p.shape, dtype=to_torch_dtype(p.dtype), device=device))
        self.entry(*args, *outs)
        return outs[0] if len(outs) == 1 else tuple(outs)


def _abi_dim(dim) -> int:
    static = static_dim_value(dim)
    return static if static is not None else -1


def callable_type_of(fn) -> CallableType:
    """Derive a ``CallableType`` for a HIR ``Function``: one ``ParamABI`` per
    declared parameter, ``output_count=0`` (a value-returning implementation),
    mirroring ``codegen/cuda/emit.py::_param_abi``'s dynamic-dim rule (a
    static dim stays its ``int`` value; a dynamic dim becomes ``-1``).
    """
    params = tuple(
        ParamABI(
            name=var.name,
            dtype=var.type.dtype,
            shape=tuple(_abi_dim(s) for s in var.type.shape),
            storage=var.type.storage,
        )
        for var in fn.params
    )
    return CallableType(name=fn.name, params=params, output_count=0)


__all__ = [
    "CallableType",
    "CompiledFunction",
    "KernelInfo",
    "LaunchConfig",
    "ParamABI",
    "RuntimeFunction",
    "callable_type_of",
]
