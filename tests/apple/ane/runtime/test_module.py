"""Compiled coefficient writers reproduce validated images without loading the device."""

import ctypes
import hashlib
from pathlib import Path

import pytest

from tests.apple.ane import device
from tests.core import _corpus
from tilefoundry import analysis, codegen, runtime, utils
from tilefoundry.ir import types
from tilefoundry.schedule import finalize
from tilefoundry_apple.ane import runtime as ane_runtime
from tilefoundry_apple.ane.codegen import module as ane_module

ROOT = Path(__file__).parents[1] / "fixtures" / "scheduled"


@pytest.fixture(params=("conv_1x1", "conv_1x1_relu", "multi_weight"))
def compiled(request, tmp_path):
    loaded = _corpus.Case(ROOT / f"{request.param}.py").load()
    checked = analysis.check_program(loaded, loaded.entry_function())
    lowered = finalize(analysis.analyze(*checked, analysis=("memory",)))
    linked = device.build(lowered, tmp_path)
    symbol = codegen.mangle(lowered, lowered.entry_function())
    return lowered, linked, symbol


def native_image(module):
    data, length = ctypes.c_void_p(), ctypes.c_size_t()
    assert (
        module.image(module.program, ctypes.byref(data), ctypes.byref(length))
        == ane_runtime.Status.OK
    )
    return ctypes.string_at(data, length.value)


def test_native_writers_match_complete_image_and_hash(compiled):
    lowered, linked, symbol = compiled
    fn = lowered.entry_function()
    module = ane_runtime.AneCompiledModule(linked.library_path, symbol)
    other = ane_runtime.AneCompiledModule(linked.library_path, symbol)
    generator = utils.torch.Generator().manual_seed(37)
    constants = {}
    try:
        for instance in (module, other):
            assert (
                instance.create(ctypes.byref(instance.metadata), ctypes.byref(instance.program))
                == ane_runtime.Status.OK
            )
        skeleton = native_image(other)
        assert skeleton == ane_module.function_image(fn).to_bytes()
        for param in fn.params:
            if not getattr(param, "is_const", False):
                continue
            tensor = utils.torch.randn(tuple(param.type.shape), generator=generator).half()
            constants[param.name] = tensor.numpy().tobytes()
            capsule, address = ane_runtime.tensor_pointer(tensor)
            assert (
                module.set_weight(module.program, param.name.encode(), address)
                == ane_runtime.Status.OK
            )
        actual = native_image(module)
        assert actual == ane_module.function_image(fn, constants).to_bytes()
        assert actual != skeleton
        assert native_image(other) == skeleton
        path = ctypes.create_string_buffer(4096)
        assert module.cache_path(module.program, path, len(path)) == ane_runtime.Status.OK
        assert hashlib.sha256(actual).hexdigest().upper() in path.value.decode()
        assert module.cache_path(module.program, path, 1) == ane_runtime.Status.BAD_ARGUMENT
        assert (
            module.set_weight(module.program, b"unknown", None) == ane_runtime.Status.BAD_ARGUMENT
        )
        assert (
            module.set_weight(module.program, module.metadata.weights[0].name, None)
            == ane_runtime.Status.BAD_ARGUMENT
        )
        wrong = utils.torch.zeros((32, 32, 1, 1), dtype=utils.torch.float32)
        capsule, address = ane_runtime.tensor_pointer(wrong)
        assert (
            module.set_weight(module.program, module.metadata.weights[0].name, address)
            == ane_runtime.Status.BAD_ARGUMENT
        )
        assert module.launch(module.program, None, -1, None, -1) == ane_runtime.Status.BAD_ARGUMENT
    finally:
        module.close()
        other.close()


def test_python_lifetime_and_invalid_resources(compiled):
    _, linked, symbol = compiled
    module = ane_runtime.AneCompiledModule(linked.library_path, symbol)
    with pytest.raises(RuntimeError, match="must be loaded"):
        module.forward()
    with pytest.raises(ValueError, match="ANE placement is not decoded yet"):
        module.load(runtime.DictResource({}), placement=types.Placement({}))
    assert not module.program.value
    with pytest.raises(KeyError):
        module.load(runtime.DictResource({}), placement=None)
    assert not module.program.value
    with pytest.raises(ValueError, match="CPU fp16"):
        module.load(
            runtime.DictResource({module.metadata.weights[0].name.decode(): utils.torch.zeros(1)})
        )
    assert not module.program.value
    module.close()
    module.close()
    with pytest.raises(RuntimeError, match="closed"):
        module.load(runtime.DictResource({}))
    with pytest.raises(RuntimeError, match="closed"):
        module.forward()


def test_provision_request_releases_candidate_and_allows_retry(compiled):
    lowered, linked, symbol = compiled
    weights = {
        param.name: utils.torch.ones(tuple(param.type.shape), dtype=utils.torch.float16)
        for param in lowered.entry_function().params
        if getattr(param, "is_const", False)
    }
    module = ane_runtime.AneCompiledModule(linked.library_path, symbol)
    module.load_model = lambda program: ane_runtime.Status.NOT_PROVISIONED
    resource = runtime.DictResource(weights)
    try:
        with pytest.raises(ane_runtime.ProvisionRequiredError) as required:
            module.load(resource)
        expected = ane_module.function_image(
            lowered.entry_function(),
            {name: value.numpy().tobytes() for name, value in weights.items()},
        ).to_bytes()
        assert required.value.image == expected
        assert hashlib.sha256(expected).hexdigest().upper() in str(required.value.path)
        assert not module.program.value
        with pytest.raises(RuntimeError, match="must be loaded"):
            module.forward()
        module.load_model = lambda program: ane_runtime.Status.OK
        module.load(resource)
        assert native_image(module) == expected
    finally:
        module.close()


class DLDevice(ctypes.Structure):
    _fields_ = (("kind", ctypes.c_int), ("index", ctypes.c_int))


class DLDataType(ctypes.Structure):
    _fields_ = (("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16))


class DLTensor(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    )


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "data",
        "shape_pointer",
        "shape",
        "ndim",
        "device",
        "device_index",
        "dtype",
        "lanes",
        "negative_stride",
        "overflow_stride",
        "overflow_offset",
    ),
)
@pytest.mark.parametrize("compiled", ("conv_1x1",), indirect=True)
def test_native_tensor_validation_rejects_invalid_host_access(compiled, fault):
    _, linked, symbol = compiled
    module = ane_runtime.AneCompiledModule(linked.library_path, symbol)
    check_tensor = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.POINTER(DLTensor),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_bool,
    )(("ane_check_tensor", module.library))
    data = (ctypes.c_uint16 * 32)()
    shape = (ctypes.c_int64 * 4)(1, 2, 2, 2)
    strides = (ctypes.c_int64 * 4)(32, 16, 4, 1)
    tensor = DLTensor(
        ctypes.addressof(data), DLDevice(1, 0), 4, DLDataType(2, 16, 1), shape, strides, 2
    )
    if fault == "none":
        assert check_tensor(ctypes.byref(tensor), shape, 16, False)
        assert not check_tensor(ctypes.byref(tensor), shape, 16, True)
        assert not check_tensor(None, shape, 16, False)
        return
    if fault == "data":
        tensor.data = None
    elif fault == "shape_pointer":
        tensor.shape = None
    elif fault == "shape":
        tensor.shape = (ctypes.c_int64 * 4)(1, 2, 2, 3)
    elif fault == "ndim":
        tensor.ndim = 3
    elif fault == "device":
        tensor.device.kind = 2
    elif fault == "device_index":
        tensor.device.index = 1
    elif fault == "dtype":
        tensor.dtype.bits = 32
    elif fault == "lanes":
        tensor.dtype.lanes = 2
    elif fault == "negative_stride":
        strides[1] = -1
    elif fault == "overflow_stride":
        strides[1] = 2**63 - 1
    else:
        tensor.byte_offset = 2**64 - 1
    assert not check_tensor(ctypes.byref(tensor), shape, 16, False)
