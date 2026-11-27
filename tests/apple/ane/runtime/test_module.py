"""Compiled coefficient writers reproduce validated images without loading the device."""

import ctypes
import hashlib
from pathlib import Path

import pytest

from tests.apple.ane import device
from tests.core import _corpus
from tilefoundry import analysis, runtime, utils
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
    return lowered, linked


def native_image(module):
    return bytes(module.functions[".image"](module.program))


def test_native_writers_match_complete_image_and_hash(compiled):
    lowered, linked = compiled
    fn = lowered.entry_function()
    module = device.compiled_module(lowered, linked)
    other = device.compiled_module(lowered, linked)
    generator = utils.torch.Generator().manual_seed(37)
    constants = {}
    try:
        for instance in (module, other):
            instance.program = instance.functions[".create"]()
        skeleton = native_image(other)
        assert skeleton == ane_module.function_image(fn).to_bytes()
        for name in module.weights:
            param = next(param for param in fn.params if param.name == name)
            tensor = utils.torch.randn(tuple(param.type.shape), generator=generator).half()
            constants[name] = tensor.numpy().tobytes()
            module.functions[".set_weight"](module.program, name, tensor)
        actual = native_image(module)
        assert actual == ane_module.function_image(fn, constants).to_bytes()
        assert actual != skeleton
        assert native_image(other) == skeleton
        cache_path = module.functions[".cache_path"](module.program)
        assert hashlib.sha256(actual).hexdigest().upper() in cache_path
        tensor = utils.torch.zeros((32, 32, 1, 1), dtype=utils.torch.float16)
        with pytest.raises(ValueError, match="compiled weight"):
            module.functions[".set_weight"](module.program, "unknown", tensor)
        with pytest.raises(ValueError, match="CPU fp16"):
            module.functions[".set_weight"](module.program, module.weights[0], tensor.float())
        with pytest.raises(ValueError, match="OIHW shape"):
            module.functions[".set_weight"](module.program, module.weights[0], tensor[:, :1])
        with pytest.raises(TypeError):
            module.functions[".set_weight"](module.program, module.weights[0], None)
        inputs = activations(module)
        output = module.alloc_outputs(inputs)[0]
        with pytest.raises(ValueError, match=module.type.input_params[0].name + " must be a CPU fp16"):
            module.functions[""](module.program, inputs[0].float(), *inputs[1:], output)
        with pytest.raises(ValueError, match=module.type.output_params[0].name + " must be a CPU fp16"):
            module.functions[""](module.program, *inputs, output.float())
        with pytest.raises(RuntimeError, match="instance must be loaded"):
            module.functions[""](None, *inputs, output)
    finally:
        module.close()
        other.close()


def activations(module):
    return tuple(
        utils.torch.zeros(tuple(param.type.shape), dtype=utils.torch.float16)
        for param in module.type.input_params
    )


def test_python_lifetime_and_invalid_resources(compiled):
    lowered, linked = compiled
    module = device.compiled_module(lowered, linked)
    inputs = activations(module)
    with pytest.raises(RuntimeError, match="must be loaded"):
        module(*inputs)
    with pytest.raises(ValueError, match="ANE placement is not decoded yet"):
        module.load(runtime.DictResource({}), placement=types.Placement({}))
    assert module.program is None
    with pytest.raises(KeyError):
        module.load(runtime.DictResource({}), placement=None)
    assert module.program is None
    with pytest.raises(ValueError, match="CPU fp16"):
        module.load(runtime.DictResource({module.weights[0]: utils.torch.zeros(1)}))
    assert module.program is None
    module.close()
    module.close()
    with pytest.raises(RuntimeError, match="closed"):
        module.load(runtime.DictResource({}))
    with pytest.raises(RuntimeError, match="closed"):
        module(*inputs)


def test_provision_request_releases_candidate_and_allows_retry(compiled):
    lowered, linked = compiled
    weights = {
        param.name: utils.torch.ones(tuple(param.type.shape), dtype=utils.torch.float16)
        for param in lowered.entry_function().params
        if getattr(param, "is_const", False)
    }
    module = device.compiled_module(lowered, linked)
    module.functions[".load"] = lambda program: ane_runtime.Status.NOT_PROVISIONED
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
        assert module.program is None
        with pytest.raises(RuntimeError, match="must be loaded"):
            module(*activations(module))
        module.functions[".load"] = lambda program: ane_runtime.Status.OK
        module.load(resource)
        assert native_image(module) == expected
        assert isinstance(module, runtime.CompiledModule)
        assert module.type.input_count == len(activations(module))
        assert not set(module.weights) & {param.name for param in module.type.params}
        calls = []
        module.functions[""] = lambda program, *args: calls.append((program, args))
        module.load(resource)
        inputs = activations(module)
        first = module(*inputs)
        second = utils.torch.empty_like(first)
        assert module(*inputs, second) is second
        assert calls[0] == (module.program, (*inputs, first))
        assert calls[1] == (module.program, (*inputs, second))
        with pytest.raises(TypeError, match="inputs.*outputs"):
            module()
        module.functions[".load"] = lambda program: ane_runtime.Status.LOAD_FAILED
        with pytest.raises(RuntimeError, match="ane_load returned ANE status 3"):
            module.load(resource)
        assert module.program is None
        with pytest.raises(RuntimeError, match="must be loaded"):
            module(*inputs)
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
    _, linked = compiled
    check_tensor = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.POINTER(DLTensor),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_bool,
    )(("ane_check_tensor", ctypes.CDLL(str(linked.library_path))))
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
