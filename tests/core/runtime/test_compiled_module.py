"""Restore real compiled roots without IR and exercise their resource lifetime."""

from __future__ import annotations

import gc
import inspect
import pydoc
import subprocess
import sys
import weakref
from dataclasses import replace

import pytest
import torch
import tvm_ffi

import tilefoundry
from tilefoundry import codegen, passes, runtime, target
from tilefoundry.ir import core, expr, types
from tilefoundry.target import cpu


def _program(*, name="Persisted", entry="run", outputs=2, dynamic=False):
    shape = (expr.DimVar("N", 1, 16),) if dynamic else (8,)
    tensor = types.TensorType.from_shape(shape, storage=expr.StorageKind.HOST)
    params = tuple(expr.Var(param, type=tensor, is_const=param in ("w0", "w1"))
                   for param in ("x", "w0", "w1", "scratch", *(f"out{index}" for index in range(outputs))))
    function = core.PrimFunction(
        name=entry, params=params, output_count=outputs, target=cpu.Target(),
        body=core.Sequential(body=(core.Evaluate(callable=core.FunctionRef((), None), args=params),)),
    )
    helper = replace(function, name="helper")
    return core.Module(name=name, functions=(function, helper), entry=entry, target=cpu.Target(), methods={"orchestrate": lambda: None})


def _emit_compute(entry, stmt, ctx):
    lines = [
        "for (int64_t i = 0; i < x.shape()[0]; ++i) {",
        "  static_cast<float*>(scratch.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + static_cast<float*>(w0.data_ptr())[i] + static_cast<float*>(w1.data_ptr())[i];",
    ]
    lines += [f"  static_cast<float*>(out{index}.data_ptr())[i] = static_cast<float*>(scratch.data_ptr())[i] * {index + 1};" for index in range(entry.output_count)]
    return "", [*lines, "}"]


class HostNamePass(passes.ModulePass):
    """Rename the native host through the actual compilation pipeline."""

    name = "test-host-name"
    requires = ()

    def run(self, module):
        function = module.entry_function()
        function.relabel(f"{function.name}_host")
        module.replace_functions(module.functions, entry=function.name)
        return module


@pytest.fixture
def compiled(tmp_path, monkeypatch):
    monkeypatch.setitem(codegen.registry.entries, (cpu.Target, codegen.Role.EMIT, core.FunctionRef), _emit_compute)
    monkeypatch.setitem(passes.registry.entries, (cpu.Target, passes.PassPoint.BEFORE_CODEGEN, HostNamePass), HostNamePass)
    return tilefoundry.compile(_program())


def _weights(offset=1):
    return runtime.DictResource({"w0": torch.full((8,), float(offset)), "w1": torch.ones(8)})


def test_export_reload_and_named_entry_follow_current_weights(compiled, tmp_path):
    x, scratch = torch.arange(8, dtype=torch.float32), torch.empty(8)
    assert compiled.name == "Persisted" and compiled.entry == "run"
    assert compiled.module is None and compiled.children == {}
    assert "run" in vars(compiled) and "run" in dir(compiled)
    assert not any(hasattr(compiled, name) for name in ("run_host", "helper", "orchestrate"))
    named = compiled.run
    assert named.__name__ == "run" and named.__qualname__ == "Persisted.run"
    assert named.__doc__ is None
    signature = inspect.signature(named)
    assert tuple(signature.parameters) == ("x", "scratch", "out0", "out1")
    assert signature.parameters["x"].annotation.shape == (8,)
    assert signature.parameters["x"].annotation.dtype == expr.DType.f32
    assert signature.parameters["out0"].default is None
    assert "run" in pydoc.render_doc(named)
    with pytest.raises(KeyError, match="Persisted.*w0.*load"):
        named(x, scratch)
    compiled.load(_weights())
    outputs = named(x, scratch)
    assert all(torch.equal(output, (x + 2) * (index + 1)) for index, output in enumerate(outputs))
    provided = tuple(torch.empty_like(x) for _ in range(2))
    assert all(torch.equal(output, expected) for output, expected in zip(compiled(x, scratch, *provided), outputs, strict=True))
    for args in ((x,), (x, scratch, provided[0]), (x, scratch, *provided, provided[0])):
        with pytest.raises(TypeError, match="inputs.*outputs"):
            compiled(*args)
    resource = _weights(3)
    old_weight = weakref.ref(resource.data["w0"])
    compiled.load(resource)
    del resource
    named(x, scratch)
    compiled.load(_weights(5), placement=types.Placement({"gpu": 4}))
    gc.collect()
    assert old_weight() is None
    assert torch.equal(named(x, scratch)[0], x + 6)
    destination = tmp_path / "persisted.so"
    compiled.export_library(destination)
    compiled.export_library(compiled.library_path)
    with pytest.raises(FileExistsError):
        compiled.export_library(destination)
    dangling = tmp_path / "dangling.so"
    dangling.symlink_to("missing.so")
    with pytest.raises(FileExistsError):
        compiled.export_library(dangling)
    with pytest.raises(OSError):
        compiled.export_library(tmp_path / "absent" / "persisted.so")
    assert torch.equal(named(x, scratch)[0], x + 6)
    restored = runtime.CompiledModule.from_library(destination)
    restored.load(_weights(7))
    borrowed = restored.run
    del restored
    gc.collect()
    assert torch.equal(borrowed(x, scratch)[0], x + 8)
    borrowed.owner.close()
    original = compiled.library_path.parent.parent
    moved = original.with_name(original.name + "_moved")
    original.rename(moved)
    try:
        result = subprocess.run([sys.executable, "-c", '''
import inspect
import torch
from tilefoundry import runtime
import sys
module = runtime.CompiledModule.from_library(sys.argv[1])
x, scratch = torch.arange(8, dtype=torch.float32), torch.empty(8)
assert module.name == "Persisted" and module.entry == "run"
assert tuple(inspect.signature(module.run).parameters) == ("x", "scratch", "out0", "out1")
assert module.placement is None and module.module is None
try:
    module(x, scratch)
except KeyError as error:
    assert "w0" in str(error)
else:
    raise AssertionError("export persisted weights")
module.load(runtime.DictResource({"w0": torch.full((8,), 10.), "w1": torch.ones(8)}))
assert torch.equal(module.run(x, scratch)[0], x + 11)
module.close()
print("fresh process reload passed")
''', str(destination)], cwd=tmp_path, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "fresh process reload passed"
    finally:
        moved.rename(original)
    compiled.close()
    compiled.close()
    for function in (compiled, named):
        with pytest.raises(RuntimeError, match="closed"):
            function(x, scratch)
    with pytest.raises(RuntimeError, match="closed"):
        compiled.load(_weights())
    with pytest.raises(RuntimeError, match="Persisted.*create.*is not decoded yet"):
        compiled.operation("create")


@pytest.mark.parametrize("value", [None, torch.ones(7), torch.ones(8, dtype=torch.float16), torch.ones(8, dtype=torch.complex64)])
def test_bad_resource_does_not_convert_or_keep_partial_weights(compiled, value):
    class Resource:
        def __init__(self):
            self.first_reads = 0
            self.second = value

        def load(self, name):
            if name == "w0":
                self.first_reads += 1
                return torch.ones(8)
            if self.second is None:
                raise KeyError(name)
            return self.second

    resource = Resource()
    compiled.load(resource)
    x, scratch = torch.zeros(8), torch.empty(8)
    with pytest.raises((KeyError, ValueError), match="Persisted.*w1"):
        compiled.run(x, scratch)
    resource.second = torch.full((8,), 2.)
    assert torch.equal(compiled(x, scratch)[0], x + 3)
    assert resource.first_reads == 2


@pytest.mark.parametrize("outputs", (0, 1))
def test_zero_and_single_output_counts(tmp_path, monkeypatch, outputs):
    monkeypatch.setitem(codegen.registry.entries, (cpu.Target, codegen.Role.EMIT, core.FunctionRef), _emit_compute)
    compiled = tilefoundry.build(_program(outputs=outputs))
    compiled.load(_weights())
    x, scratch = torch.zeros(8), torch.empty(8)
    result = compiled(x, scratch)
    if outputs:
        assert torch.equal(result, x + 2)
    else:
        assert result == ()
    compiled.close()


def test_dynamic_outputs_require_preallocation(tmp_path, monkeypatch):
    monkeypatch.setitem(codegen.registry.entries, (cpu.Target, codegen.Role.EMIT, core.FunctionRef), _emit_compute)
    compiled = tilefoundry.build(_program(dynamic=True))
    compiled.load(_weights())
    x, scratch = torch.zeros(8), torch.empty(8)
    with pytest.raises(RuntimeError, match="dynamic shape is not decoded yet"):
        compiled(x, scratch)
    root = core.Module(name="Dynamic", functions=(), entry=None, target=cpu.Target())
    tensor = types.TensorType.from_shape((expr.DimVar("N", 1, 16),), storage=expr.StorageKind.HOST)
    native = codegen.CallableSignature("dynamic", params=tuple(codegen.TensorSignature(name, tensor) for name in ("x", "out")), output_count=1)
    description = codegen.artifact.describe_root(root)
    description.update(entry="run", functions=[{"name": "run", "symbol": "dynamic", "signature": codegen.artifact.describe_signature(native, (expr.Var("x", type=tensor),))}])
    project = codegen.CMakeProject(tmp_path, "dynamic")
    project.add_source("manifest.cpp", codegen.artifact.render_description({"format": 1, "root": description}))
    project.add_source("entry.cpp", '''
#define TVM_FFI_DLL_EXPORT_INCLUDE_METADATA 1
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
void dynamic(tvm::ffi::Tensor x, tvm::ffi::Tensor out) {
    for (int64_t i = 0; i < x.shape()[0]; ++i) static_cast<float*>(out.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i];
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dynamic, dynamic);
''')
    loaded = runtime.CompiledModule.from_library(project.build(()).library_path)
    with pytest.raises(RuntimeError, match="dynamic output allocation is not decoded yet"):
        loaded(x)
    out = torch.empty_like(x)
    assert torch.equal(loaded.run(x, out), x)


def test_empty_root_and_reserved_names(tmp_path):
    empty = core.Module(name="Empty", functions=(), target=cpu.Target())
    for build in (tilefoundry.build, tilefoundry.compile):
        compiled = build(empty)
        assert compiled.entry is None and compiled.type is None
        destination = tmp_path / (build.__name__ + ".so")
        compiled.export_library(destination)
        loaded = runtime.CompiledModule.from_library(destination)
        with pytest.raises(RuntimeError, match=r"no entry.*functions=\[\]"):
            loaded()
    for name in ("forward", "entry", "load", "library", "bound", "close"):
        module = _program(entry=name)
        for build in (tilefoundry.build, tilefoundry.compile):
            with pytest.raises(ValueError, match=rf"Persisted.{name}.*reserved"):
                build(module)
    child = replace(empty, name="Child")
    with pytest.raises(ValueError, match="compiled module tree is not decoded yet"):
        tilefoundry.build(replace(empty, children={"child": child}))
    for name in ("forward", "entry"):
        library = tvm_ffi.load_module(compiled.library_path, keep_module_alive=False)
        description = codegen.artifact.validate_description(library.get_function("tilefoundry_manifest")())
        tensor = {"dtype": "f32", "shape": [8]}
        description["root"].update(entry=name, functions=[{"name": name, "symbol": "missing", "signature": {"params": [{"name": "x", "tensor": tensor, "role": "input"}], "output_count": 0, "unit_ids": []}}])
        project = codegen.CMakeProject(tmp_path / name, name)
        project.add_source("manifest.cpp", codegen.artifact.render_description(description))
        with pytest.raises(RuntimeError, match=rf"Empty.{name}.*reserved"):
            runtime.CompiledModule.from_library(project.build(()).library_path)


def test_provider_binding_and_reserved_runtime_fields(tmp_path, monkeypatch):
    @target.register
    class OfflineTarget(cpu.Target):
        name = "offline_provider"

    class OfflineCompiled(runtime.CompiledModule):
        token: int

        def __init__(self, library, library_path, description):
            super().__init__(library, library_path, description)
            self.token = 0

    monkeypatch.setitem(target.bindings.entries, (OfflineTarget, runtime.CompiledModule), OfflineCompiled)
    empty = core.Module(name="Offline", functions=(), target=OfflineTarget())
    compiled = tilefoundry.build(empty)
    assert isinstance(compiled, OfflineCompiled)
    exported = tmp_path / "offline.so"
    compiled.export_library(exported)

    def refuse_construction(*args, **kwargs):
        raise AssertionError("loading reconstructed a Target")

    monkeypatch.setattr(OfflineTarget, "__init__", refuse_construction)
    loaded = runtime.CompiledModule.from_library(exported)
    assert isinstance(loaded, OfflineCompiled)
    with pytest.raises(RuntimeError, match="Offline.create is not decoded yet") as error:
        loaded.operation("create")
    assert str(exported) in str(error.value)
    function = core.PrimFunction(name="token", params=(), output_count=0, body=core.Sequential(body=()), target=empty.target)
    with pytest.raises(ValueError, match="Offline.token.*reserved"):
        tilefoundry.build(replace(empty, functions=(function,), entry="token"))
    description = {"format": 1, "root": {"name": "Offline", "target": OfflineTarget.name, "entry": "token", "functions": [
        {"name": "token", "symbol": "missing", "signature": {"params": [], "output_count": 0, "unit_ids": []}},
    ], "weights": {}, "operations": {}}}
    project = codegen.CMakeProject(tmp_path / "reserved", "reserved")
    project.add_source("manifest.cpp", codegen.artifact.render_description(description))
    with pytest.raises(RuntimeError, match="Offline.token.*reserved"):
        runtime.CompiledModule.from_library(project.build(()).library_path)
    target.unregister(OfflineTarget)
    with pytest.raises(RuntimeError, match="Offline.*target provider.*is not decoded yet") as error:
        runtime.CompiledModule.from_library(exported)
    assert str(exported) in str(error.value)
