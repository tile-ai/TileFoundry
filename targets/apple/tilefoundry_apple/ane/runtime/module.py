"""Load weights through TVM FFI and bind an owned ANE model to a compiled entry."""

from enum import IntEnum
from functools import partial
from pathlib import Path

from tilefoundry import codegen, runtime, utils
from tilefoundry.ir import types

tvm_ffi = utils.LazyModule("tvm_ffi")


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


class AneCompiledModule(runtime.CompiledModule):
    """A compiled entry whose FFI callable binds one independent model handle."""

    def __init__(
        self, library: str | Path, type: codegen.CallableSignature, weights: tuple[str, ...] = ()
    ) -> None:
        self.library = tvm_ffi.load_module(str(library), keep_module_alive=True)
        try:
            self.functions = {
                suffix: tvm_ffi.get_global_func(type.name + suffix)
                for suffix in (
                    "", ".create", ".set_weight", ".load", ".cache_path", ".image", ".destroy"
                )
            }
        except ValueError as error:
            raise RuntimeError(
                f"ANE library {library} lacks registrations for entry {type.name!r}"
            ) from error
        super().__init__(type, partial(unavailable, "must be loaded before forward"))
        self.weights = weights
        self.program = None
        self.closed = False

    def load(
        self, resource: runtime.Resource, *, placement: "types.Placement | None" = None
    ) -> None:
        """Create a private image, write its weights and bind the provisioned model."""
        if self.closed:
            raise RuntimeError("ANE module is closed")
        if placement is not None:
            raise ValueError("ANE placement is not decoded yet")
        self.fn = partial(unavailable, "must be loaded before forward")
        previous, self.program = self.program, None
        self.functions[".destroy"](previous)
        try:
            self.program = self.functions[".create"]()
            for name in self.weights:
                tensor = resource.load(name)
                if tensor.device.type != "cpu" or tensor.dtype != utils.torch.float16:
                    raise ValueError(f"weight {name} must be a CPU fp16 tensor")
                self.functions[".set_weight"](self.program, name, tensor.contiguous())
            status = self.functions[".load"](self.program)
            if status == Status.NOT_PROVISIONED:
                raise ProvisionRequiredError(
                    bytes(self.functions[".image"](self.program)),
                    Path(self.functions[".cache_path"](self.program)),
                )
            check(status, "ane_load")
            self.fn = partial(self.functions[""], self.program)
        except BaseException:
            candidate, self.program = self.program, None
            self.functions[".destroy"](candidate)
            raise

    def close(self) -> None:
        """Unload and release this module's native instance; repeated calls are safe."""
        self.fn = partial(unavailable, "is closed")
        self.closed = True
        previous, self.program = self.program, None
        self.functions[".destroy"](previous)


__all__ = ["AneCompiledModule", "ProvisionRequiredError", "Status"]
