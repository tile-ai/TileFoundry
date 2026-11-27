"""Build, load and close an ANE entry whose model one integer token owns."""

import tempfile
from dataclasses import replace
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Callable

from tilefoundry import codegen, runtime, target, utils
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, types
from tilefoundry_apple.ane import target as ane_target

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


def entry_name(owner: core.Module, fn: core.PrimFunction) -> str:
    """The dotted name *fn* and its lifecycle functions register under."""
    ref = next(
        ref for ref, reading in valuation.collect_functions(owner).items()
        if reading.function is fn
    )
    return f"{owner.name}.{ref.qualified_name}"


@target.bind(ane_target.Target, runtime.CompiledModule)
class AneCompiledModule(runtime.CompiledModule):
    """A compiled entry whose FFI callable binds one library-owned instance token."""

    def __init__(
        self, type: codegen.CallableSignature, fn: Callable, library_path: Path
    ) -> None:
        super().__init__(type, partial(unavailable, "must be loaded before forward"), library_path)
        self.entry_fn = fn
        try:
            self.functions = {
                name: tvm_ffi.get_global_func(f"{type.name}.{name}")
                for name in (
                    "create", "weights", "set_weight", "load", "cache_path", "image", "destroy"
                )
            }
        except ValueError as error:
            raise RuntimeError(
                f"ANE library {library_path} lacks registrations for entry {type.name!r}"
            ) from error
        self.weights = tuple(self.functions["weights"]())
        self.program = 0
        self.closed = False

    @classmethod
    def create(cls, tgt, mod: core.Module, options) -> "AneCompiledModule":
        """Emit the lowered entry with the launcher, link it with clang++ and bind it."""
        fn = mod.entry_function()
        if not isinstance(fn, core.PrimFunction) or fn.target != tgt:
            raise ValueError(f"tilefoundry.build: Module.entry {mod.entry!r} is not a lowered ANE entry")
        signature = codegen.signature.called_as(fn, codegen.mangle(mod, fn), target=tgt)
        ctx = codegen.EmitContext(codegen.registry, symbols={id(fn): signature}, target=tgt)
        linked = codegen.link_modules(
            (tgt.resolve(codegen.LinkableModule, mod, (fn,), ctx),),
            workdir=tempfile.mkdtemp(prefix=f"tilefoundry_build_{mod.entry}_"),
            lib_name=mod.name,
            functions=(),
            host=tgt,
            host_cxx="clang++",
        )
        return cls.from_library(linked.library_path, replace(signature, name=entry_name(mod, fn)))

    def load(
        self, resource: runtime.Resource, *, placement: "types.Placement | None" = None
    ) -> None:
        """Write the weights into a new instance and bind it once its model loads."""
        if self.closed:
            raise RuntimeError(f"{self.name}: ANE module is closed")
        if placement is not None:
            raise ValueError(f"{self.name}: ANE placement is not decoded yet")
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
                if (
                    not isinstance(tensor, utils.torch.Tensor)
                    or tensor.device.type != "cpu"
                    or tensor.dtype != utils.torch.float16
                ):
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
        self.fn = partial(self.entry_fn, candidate)

    def close(self) -> None:
        """Unbind the entry and release the instance; repeated calls are safe."""
        self.fn = partial(unavailable, "is closed")
        self.closed = True
        previous, self.program = self.program, 0
        self.functions["destroy"](previous)


__all__ = ["AneCompiledModule", "ProvisionRequiredError", "Status", "entry_name"]
