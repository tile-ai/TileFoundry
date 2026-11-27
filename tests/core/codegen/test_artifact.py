"""Exercise manifests, typed metadata and ownership with real CPU shared libraries."""

from __future__ import annotations

import copy
import gc
import json

import pytest
import torch
import tvm_ffi

from tilefoundry import codegen, runtime
from tilefoundry.ir import core, expr, types
from tilefoundry.target import cpu


def _program(scale, shape=(8,)):
    tensor = types.TensorType.from_shape(shape, storage=expr.StorageKind.HOST)
    params = tuple(expr.Var(name=name, type=tensor) for name in ("x", "out"))
    body = core.Sequential(body=(core.Evaluate(callable=core.FunctionRef((), f"scale{scale}"), args=params),))
    functions = tuple(core.PrimFunction(
        name=name, params=params, body=body, target=cpu.Target(), output_count=1,
    ) for name in ("run", "helper"))
    return core.Module(name="CpuRoot", functions=functions, entry="run", target=cpu.Target())


def _emit_scale(entry, stmt, ctx):
    scale = int(stmt.callable.name.removeprefix("scale"))
    return "", [
        'if (x.ndim() != 1 || out.ndim() != 1 || x.shape()[0] != out.shape()[0])',
        '    throw std::runtime_error("shape mismatch");',
        f"for (int64_t i = 0; i < x.shape()[0]; ++i) static_cast<float*>(out.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] * {scale};",
    ]


@pytest.fixture
def cpu_libraries(tmp_path, monkeypatch):
    monkeypatch.setitem(codegen.registry.entries, (cpu.Target, codegen.Role.EMIT, core.FunctionRef), _emit_scale)
    return tuple(codegen.link_program(_program(scale), workdir=tmp_path / str(scale)) for scale in (2, 3))


def test_cpu_libraries_keep_local_exports_and_callable_ownership(cpu_libraries):
    x = torch.arange(8, dtype=torch.float32)
    retained = []
    for scale, linked in zip((2, 3), cpu_libraries, strict=True):
        library = tvm_ffi.load_module(linked.library_path, keep_module_alive=False)
        description = codegen.artifact.validate_description(library.get_function("tilefoundry_manifest", query_imports=False)())
        assert description == {"format": 1, "root": {
            "name": "CpuRoot", "target": "cpu", "entry": "run", "weights": {}, "operations": {},
            "functions": [{"name": "run", "symbol": linked.functions[0].symbol, "signature": {
                "params": [{"name": name, "tensor": {"dtype": "f32", "shape": [8]}, "role": role}
                           for name, role in (("x", "input"), ("out", "output"))],
                "output_count": 1, "unit_ids": [],
            }}],
        }}
        symbol = description["root"]["functions"][0]["symbol"]
        metadata = library.get_function_metadata(symbol, query_imports=False)
        assert json.loads(metadata["type_schema"]) == {"type": "ffi.Function", "args": [
            {"type": "None"}, {"type": "ffi.Tensor"}, {"type": "ffi.Tensor"},
        ]}
        assert library.get_function_doc(symbol) is None
        with pytest.raises(AttributeError):
            library.get_function("tilefoundry_7CpuRoot6helper", query_imports=False)
        compiled = runtime.load_linked_module(linked)
        assert torch.equal(compiled(x), x * scale)
        out = torch.empty_like(x)
        assert torch.equal(compiled(x, out), x * scale)
        retained.append(compiled.fn)
        del compiled, library
    gc.collect()
    for scale, function in zip((2, 3), retained, strict=True):
        out = torch.empty_like(x)
        function(x, out)
        assert torch.equal(out, x * scale)
    assert tvm_ffi.get_global_func(cpu_libraries[0].functions[0].name, allow_missing=True) is None
    assert tvm_ffi.get_global_func(cpu_libraries[0].functions[0].symbol, allow_missing=True) is None


def test_dynamic_cpu_tensor_uses_preallocated_output(tmp_path, monkeypatch):
    monkeypatch.setitem(codegen.registry.entries, (cpu.Target, codegen.Role.EMIT, core.FunctionRef), _emit_scale)
    module = _program(4, (expr.DimVar("N", 1, 16),))
    linked = codegen.link_program(module, workdir=tmp_path)
    compiled = runtime.load_linked_module(linked)
    x = torch.arange(5, dtype=torch.float32)
    out = torch.empty_like(x)
    assert torch.equal(compiled(x, out), x * 4)
    library = tvm_ffi.load_module(linked.library_path, keep_module_alive=False)
    description = codegen.artifact.validate_description(library.get_function("tilefoundry_manifest")())
    assert description["root"]["functions"][0]["signature"]["params"][0]["tensor"]["shape"] == [{"name": "N", "lo": 1, "hi": 16}]


@pytest.fixture(scope="module")
def mutable_library(tmp_path_factory):
    project = codegen.CMakeProject(tmp_path_factory.mktemp("bad_artifacts"), "bad")
    project.add_source("bad.cpp", '''
#define TVM_FFI_DLL_EXPORT_INCLUDE_METADATA 1
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
using Description = tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>;
static Description description;
Description manifest_impl() { return description; }
void set_description(Description value) { description = value; }
void run(tvm::ffi::Tensor x, tvm::ffi::Tensor out) {}
void wrong_arity(tvm::ffi::Tensor x) {}
void wrong_type(int64_t x, tvm::ffi::Tensor out) {}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tilefoundry_manifest, manifest_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_description, set_description);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrong_arity, wrong_arity);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrong_type, wrong_type);
''')
    project.add_source("no_metadata.cpp", '''
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
void no_metadata(tvm::ffi::Tensor x, tvm::ffi::Tensor out) {}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(no_metadata, no_metadata);
''')
    sig = codegen.build_context(_program(2)).symbols.by_function
    native = next(symbol.callable for symbol in sig.values() if symbol.function.name == "run")
    entry = codegen.LinkedFunction(name="CpuRoot.run", symbol="run", signature=native)
    linked = project.build((entry,))
    library = tvm_ffi.load_module(linked.library_path, keep_module_alive=False)
    root = codegen.artifact.describe_root(_program(2))
    root["functions"] = [{"name": "run", "symbol": "run", "signature": codegen.artifact.describe_signature(native, _program(2).entry_function().params)}]
    return linked, library, {"format": 1, "root": root}


@pytest.mark.parametrize(("path", "value", "message"), [
    (("format",), 2, "format"),
    (("format",), "1", "format"),
    (("root", "entry"), "missing", "name"),
    (("root", "availability"), {}, "root"),
    (("root", "functions", 0, "symbol"), "bad.symbol", "symbol"),
    (("root", "operations"), {"load": "run"}, "duplicate symbol"),
    (("root", "functions", 0, "signature", "output_count"), 3, "output_count"),
    (("root", "functions", 0, "signature", "output_count"), 0, "trailing"),
    (("root", "functions", 0, "signature", "params", 1, "name"), "x", "duplicate parameter"),
    (("root", "functions", 0, "signature", "params", 0, "role"), "unknown", "role"),
    (("root", "functions", 0, "signature", "params", 0, "role"), "weight", "weight"),
    (("root", "functions", 0, "signature", "params", 0, "tensor", "dtype"), "unknown", "dtype"),
    (("root", "functions", 0, "signature", "params", 0, "tensor", "shape"), [-1], "shape"),
    (("root", "functions", 0, "signature", "params", 0, "tensor", "shape"), [{"name": "N", "lo": 8, "hi": 1}], "lo <= hi"),
    (("root", "functions", 0, "signature", "unit_ids"), ["gpu", "gpu"], "duplicate topology"),
    (("root", "functions", 0, "symbol"), "wrong_arity", "metadata does not match"),
    (("root", "functions", 0, "symbol"), "wrong_type", "metadata does not match"),
    (("root", "functions", 0, "symbol"), "no_metadata", "missing metadata"),
])
def test_bad_library_is_rejected_before_binding(mutable_library, path, value, message):
    linked, library, description = mutable_library
    malformed = copy.deepcopy(description)
    parent = malformed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    library.get_function("set_description")(tvm_ffi.convert(malformed))
    with pytest.raises(RuntimeError, match=message) as caught:
        runtime.load_linked_module(linked)
    assert str(linked.library_path) in str(caught.value)


def test_missing_local_symbol_does_not_use_global_function(mutable_library):
    linked, library, description = mutable_library
    malformed = copy.deepcopy(description)
    malformed["root"]["functions"][0]["symbol"] = "missing_local_entry"
    library.get_function("set_description")(tvm_ffi.convert(malformed))
    tvm_ffi.register_global_func("missing_local_entry", lambda: None)
    try:
        with pytest.raises(RuntimeError, match="missing symbol 'missing_local_entry'") as caught:
            runtime.load_linked_module(linked)
        assert str(linked.library_path) in str(caught.value)
    finally:
        tvm_ffi.remove_global_func("missing_local_entry")


def test_library_without_manifest_requires_recompile(tmp_path):
    project = codegen.CMakeProject(tmp_path, "old")
    project.add_source("old.cpp", 'extern "C" int old_entry() { return 0; }\n')
    linked = project.build(())
    tvm_ffi.register_global_func("tilefoundry_manifest", lambda: None)
    try:
        with pytest.raises(RuntimeError, match="tilefoundry_manifest.*recompile") as caught:
            runtime.load_linked_module(linked)
        assert str(linked.library_path) in str(caught.value)
    finally:
        tvm_ffi.remove_global_func("tilefoundry_manifest")


@pytest.mark.parametrize("tensor", [
    types.TensorType.from_shape((8,), storage=expr.StorageKind.SMEM),
    types.TensorType((8,), expr.DType.f32, types.Layout((8,), (2,)), expr.StorageKind.HOST),
    types.TensorType.from_shape((expr.DimVar("N", 1, 8) * 2,), storage=expr.StorageKind.HOST),
])
def test_unsupported_tensor_codec_names_parameter(tensor):
    signature = codegen.CallableSignature(name="CpuRoot.run", params=(codegen.TensorSignature("x", tensor),))
    with pytest.raises(ValueError, match=r"CpuRoot.run.params\[0\].*is not decoded yet"):
        codegen.artifact.describe_signature(signature, ())


def test_weight_scratch_and_unit_id_abi(tmp_path):
    tensor = types.TensorType.from_shape((8,), storage=expr.StorageKind.HOST)
    params = tuple(expr.Var(name=name, type=tensor, is_const=name == "weight")
                   for name in ("x", "weight"))
    source = core.PrimFunction(name="run", params=params, body=core.Sequential(body=()), output_count=0)
    module = core.Module(name="Resources", functions=(source,), entry="run", target=cpu.Target())
    native = codegen.CallableSignature(
        name="with_resources",
        params=tuple(codegen.TensorSignature(name, tensor) for name in ("x", "weight", "scratch", "out")),
        output_count=1, leading=(codegen.UnitIdSignature("unit", "gpu"),),
    )
    root = codegen.artifact.describe_root(module)
    sig = codegen.artifact.describe_signature(native, params)
    assert [param["role"] for param in sig["params"]] == ["input", "weight", "scratch", "output"]
    root["functions"] = [{"name": "run", "symbol": "with_resources", "signature": sig}]
    project = codegen.CMakeProject(tmp_path, "resources")
    project.add_source("manifest.cpp", codegen.artifact.render_description({"format": 1, "root": root}))
    project.add_source("entry.cpp", '''
#define TVM_FFI_DLL_EXPORT_INCLUDE_METADATA 1
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
void with_resources(int64_t unit, tvm::ffi::Tensor x, tvm::ffi::Tensor weight,
                    tvm::ffi::Tensor scratch, tvm::ffi::Tensor out) {
    for (int64_t i = 0; i < x.shape()[0]; ++i) {
        static_cast<float*>(scratch.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + static_cast<float*>(weight.data_ptr())[i];
        static_cast<float*>(out.data_ptr())[i] = static_cast<float*>(scratch.data_ptr())[i] * unit;
    }
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(with_resources, with_resources);
''')
    linked = project.build((codegen.LinkedFunction("Resources.run", "with_resources", native),))
    compiled = runtime.load_linked_module(linked)
    x = torch.arange(8, dtype=torch.float32)
    weight = torch.ones_like(x)
    scratch, out = torch.empty_like(x), torch.empty_like(x)
    compiled.fn(2, x, weight, scratch, out)
    assert torch.equal(out, (x + weight) * 2)
