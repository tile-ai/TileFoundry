"""Load weights into an owned ANE image and evaluate its persistent model."""

import ctypes
from enum import IntEnum
from pathlib import Path

from tilefoundry import runtime, utils
from tilefoundry.ir import types

PATH_CAPACITY = 4096
POINTER = ctypes.c_void_p
INT = ctypes.c_int
SIZE = ctypes.c_size_t


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


TensorDesc = type(
    "TensorDesc",
    (ctypes.Structure,),
    {
        "__doc__": "Compiled I/O geometry in C ABI field order.",
        "_fields_": (
            ("dtype", ctypes.c_char_p),
            ("shape", ctypes.c_int64 * 4),
            ("strides", ctypes.c_int64 * 4),
            ("storage_bytes", SIZE),
        ),
    },
)
Weight = type(
    "Weight",
    (ctypes.Structure,),
    {
        "__doc__": "A resource name and its generated native writer.",
        "_fields_": (("name", ctypes.c_char_p), ("write", POINTER)),
    },
)
Entry = type(
    "Entry",
    (ctypes.Structure,),
    {
        "__doc__": "Immutable metadata returned by a generated entry export.",
        "_fields_": (
            ("image", POINTER),
            ("image_len", SIZE),
            ("coefficient_offset", SIZE),
            ("io", ctypes.POINTER(TensorDesc)),
            ("n_inputs", INT),
            ("n_outputs", INT),
            ("weights", ctypes.POINTER(Weight)),
            ("n_weights", INT),
        ),
    },
)


def tensor_pointer(tensor) -> tuple[object, int]:
    """Keep a DLPack capsule alive alongside its embedded tensor address."""
    capsule = utils.torch.utils.dlpack.to_dlpack(tensor)
    get = ctypes.PYFUNCTYPE(POINTER, ctypes.py_object, ctypes.c_char_p)(
        ("PyCapsule_GetPointer", ctypes.pythonapi)
    )
    return capsule, get(capsule, b"dltensor")


def check(status: int, operation: str) -> None:
    """Turn a native failure into an operation-specific Python error."""
    if status != Status.OK:
        raise RuntimeError(f"{operation} returned ANE status {status}")


class AneCompiledModule(runtime.Module):
    """A linked entry with an independent weight image and loaded model handle."""

    def __init__(self, library: str | Path, symbol: str) -> None:
        super().__init__(symbol, entry=symbol)
        self.library = ctypes.CDLL(str(library))
        self.metadata = ctypes.CFUNCTYPE(ctypes.POINTER(Entry))((symbol, self.library))().contents
        self.create = ctypes.CFUNCTYPE(INT, ctypes.POINTER(Entry), ctypes.POINTER(POINTER))(
            ("ane_create", self.library)
        )
        self.set_weight = ctypes.CFUNCTYPE(INT, POINTER, ctypes.c_char_p, POINTER)(
            ("ane_set_weight", self.library)
        )
        self.cache_path = ctypes.CFUNCTYPE(INT, POINTER, POINTER, SIZE)(
            ("ane_cache_path", self.library)
        )
        self.image = ctypes.CFUNCTYPE(INT, POINTER, ctypes.POINTER(POINTER), ctypes.POINTER(SIZE))(
            ("ane_image", self.library)
        )
        self.load_model = ctypes.CFUNCTYPE(INT, POINTER)(("ane_load", self.library))
        self.launch = ctypes.CFUNCTYPE(INT, POINTER, POINTER, INT, POINTER, INT)(
            ("ane_launch", self.library)
        )
        self.destroy = ctypes.CFUNCTYPE(INT, ctypes.POINTER(POINTER))(("ane_destroy", self.library))
        self.program = POINTER()
        self.loaded = False
        self.closed = False

    def load(
        self, resource: runtime.Resource, *, placement: "types.Placement | None" = None
    ) -> None:
        """Create an image, bind its weights, and load the provisioned model."""
        if self.closed:
            raise RuntimeError("ANE module is closed")
        if placement is not None:
            raise ValueError("ANE placement is not decoded yet")
        self.loaded = False
        check(self.destroy(ctypes.byref(self.program)), "ane_destroy")
        try:
            check(
                self.create(ctypes.byref(self.metadata), ctypes.byref(self.program)), "ane_create"
            )
            for index in range(self.metadata.n_weights):
                name = self.metadata.weights[index].name
                tensor = resource.load(name.decode())
                if tensor.device.type != "cpu" or tensor.dtype != utils.torch.float16:
                    raise ValueError(f"weight {name.decode()} must be a CPU fp16 tensor")
                tensor = tensor.contiguous()
                capsule, address = tensor_pointer(tensor)
                check(
                    self.set_weight(self.program, name, address),
                    f"ane_set_weight({name.decode()}): shape must match compiled OIHW",
                )
            status = self.load_model(self.program)
            if status == Status.NOT_PROVISIONED:
                path = ctypes.create_string_buffer(PATH_CAPACITY)
                check(self.cache_path(self.program, path, len(path)), "ane_cache_path")
                data, size = POINTER(), SIZE()
                check(self.image(self.program, ctypes.byref(data), ctypes.byref(size)), "ane_image")
                raise ProvisionRequiredError(
                    ctypes.string_at(data, size.value), Path(path.value.decode())
                )
            check(status, "ane_load")
            self.loaded = True
        except BaseException:
            check(self.destroy(ctypes.byref(self.program)), "ane_destroy")
            raise

    def forward(self, *args):
        """Allocate the compiled outputs and evaluate with CPU tensors."""
        if self.closed or not self.loaded:
            raise RuntimeError(
                "ANE module is closed"
                if self.closed
                else "ANE module must be loaded before forward"
            )
        if len(args) != self.metadata.n_inputs:
            raise ValueError(
                f"ANE forward requires {self.metadata.n_inputs} inputs, got {len(args)}"
            )
        for index, tensor in enumerate(args):
            desc = self.metadata.io[index]
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != utils.torch.float16
                or tuple(tensor.shape) != tuple(desc.shape)
            ):
                raise ValueError(
                    f"input {index} must be a CPU fp16 tensor of shape {tuple(desc.shape)}"
                )
        outputs = tuple(
            utils.torch.empty(
                tuple(self.metadata.io[self.metadata.n_inputs + index].shape),
                dtype=utils.torch.float16,
            )
            for index in range(self.metadata.n_outputs)
        )
        held = [tensor_pointer(tensor) for tensor in (*args, *outputs)]
        pointers = [address for _, address in held]
        inputs = (POINTER * len(args))(*pointers[: len(args)])
        results = (POINTER * len(outputs))(*pointers[len(args) :])
        check(self.launch(self.program, inputs, len(args), results, len(outputs)), "ane_launch")
        return outputs[0] if len(outputs) == 1 else outputs

    def close(self) -> None:
        """Unload and release this module's native instance."""
        self.loaded = False
        self.closed = True
        check(self.destroy(ctypes.byref(self.program)), "ane_destroy")


__all__ = ["AneCompiledModule", "ProvisionRequiredError", "Status", "tensor_pointer"]
