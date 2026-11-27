"""Load weights through TVM FFI and bind an owned ANE model to a compiled entry."""

import json
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Callable

from tilefoundry import runtime, target, utils
from tilefoundry.ir import types
from tilefoundry_apple.ane import target as ane_target


class Status(IntEnum):
    """AneStatus values from codegen/include/tilefoundry/runtime/ane/launcher.h."""

    OK = 0
    UNAVAILABLE = 1
    NOT_PROVISIONED = 2
    LOAD_FAILED = 3
    BAD_ARGUMENT = 4
    EVALUATE_FAILED = 5
    UNLOAD_FAILED = 6


class ProvisionRequiredError(RuntimeError):
    """An assembled image and the cache file the caller must provision."""

    def __init__(self, image: bytes, path: Path) -> None:
        super().__init__(f"ANE image must be provisioned at {path}")
        self.image = image
        self.path = path


def unavailable(reason: str, *args) -> None:
    """Refuse a call without a loaded instance using the inherited entry ABI."""
    raise RuntimeError(f"ANE module {reason}")


def check(status: int, operation: str) -> None:
    """Turn a native failure into an operation-specific Python error."""
    if status != Status.OK:
        raise RuntimeError(f"{operation} returned ANE status {status}")


@target.bind(ane_target.Target, runtime.CompiledModule)
class AneCompiledModule(runtime.CompiledModule):
    """A root whose native entry binds one library-owned integer token."""

    functions: dict[str, Callable]
    entry_ffi: Callable | None
    program: int

    def __init__(self, library, library_path, description) -> None:
        super().__init__(library, library_path, description)
        self.functions = {
            name: self.operation(name)
            for name in (() if self.type is None else ("create", "set_weight", "load", "cache_path", "image", "destroy"))
        }
        self.entry_ffi = self.fn
        self.program = 0
        if self.type is not None:
            self.fn = partial(unavailable, "must be loaded before forward")

    def entry_schema(self) -> tuple[str, ...]:
        """State the bound token and DLPack tensor ABI of the native entry."""
        if self.type.leading:
            raise RuntimeError(f"library {self.library_path}: {self.name}: ANE unit ids is not decoded yet")
        return ("None", "int", *("DLTensor*" for _ in self.type.params))

    def operation(self, name: str):
        """Bind a library-local lifecycle function and verify its exact typed ABI."""
        function = super().operation(name)
        schemas = {
            "create": ("int",), "set_weight": ("None", "int", "ffi.String", "DLTensor*"),
            "load": ("int", "int"), "cache_path": ("ffi.String", "int"),
            "image": ("ffi.Bytes", "int"), "destroy": ("None", "int"),
        }
        kinds = schemas.get(name)
        if kinds is None:
            raise RuntimeError(f"library {self.library_path}: {self.name}.{name} is not decoded yet")
        expected = {"type": "ffi.Function", "args": [{"type": kind} for kind in kinds]}
        metadata = self.library.get_function_metadata(self.operations[name], query_imports=False)
        if json.loads(metadata["type_schema"]) != expected:
            raise RuntimeError(f"library {self.library_path}: {self.name}.{name}: metadata does not match signature")
        return function

    def load(
        self, resource: runtime.Resource, *, placement: "types.Placement | None" = None
    ) -> None:
        """Assemble private weights and bind a successfully loaded candidate token."""
        if self.closed:
            raise RuntimeError(f"{self.name}: ANE module is closed")
        if placement is not None:
            raise ValueError(f"{self.name}: ANE placement is not decoded yet")
        if self.type is None:
            super().load(resource)
            return
        self.fn = partial(unavailable, "must be loaded before forward")
        previous, self.program = self.program, 0
        self.functions["destroy"](previous)
        candidate = self.functions["create"]()
        try:
            for name in self.weights:
                try:
                    tensor = resource.load(name)
                except KeyError as error:
                    raise KeyError(f"{self.name}: missing weight {name!r}") from error
                if not isinstance(tensor, utils.torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != utils.torch.float16:
                    raise ValueError(f"{self.name}: weight {name} must be a CPU fp16 tensor")
                self.functions["set_weight"](candidate, name, tensor.contiguous())
            status = self.functions["load"](candidate)
            if status == Status.NOT_PROVISIONED:
                raise ProvisionRequiredError(
                    bytes(self.functions["image"](candidate)),
                    Path(self.functions["cache_path"](candidate)),
                )
            check(status, "ane_load")
        except BaseException:
            self.functions["destroy"](candidate)
            raise
        self.program = candidate
        self.fn = partial(self.entry_ffi, candidate)

    def close(self) -> None:
        """Disable calls and release the token once, even if unloading fails."""
        if self.closed:
            return
        previous, self.program = self.program, 0
        super().close()
        if previous:
            self.functions["destroy"](previous)


__all__ = ["AneCompiledModule", "ProvisionRequiredError", "Status"]
