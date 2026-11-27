"""Pin caller-visible ``tilefoundry.jit`` behavior.

End-to-end tests cover compilation. These tests localize rejected inputs and the
observable cache contract: repeated JIT of one program returns the same module,
while clearing the cache ends that identity. Private key structure is not part of
the contract.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tilefoundry import cli, codegen, jit, passes, runtime
from tilefoundry.ir import core, expr, types
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda


class HostNamePass(passes.ModulePass):
    """Exercise JIT reuse across an in-place host rename."""

    name = "test-jit-host-name"
    requires = ()

    def run(self, module):
        function = module.entry_function()
        function.relabel(f"{function.name}_host")
        module.replace_functions(module.functions, entry=function.name)
        return module


def test_jit_rejects_bad_arguments() -> None:
    """Raw Python, authored string Targets, and unexpected kwargs all raise."""

    def raw_fn(a):
        return a

    with pytest.raises(TypeError, match="expected Function or Module"):
        jit(raw_fn, target=cuda.Target("nvidia.h200_sxm"))

    rmsnorm = Path(__file__).parents[1] / "fixtures/placed/rmsnorm.py"
    fn = cli.load_authored_ir(f"{rmsnorm}:RmsnormModule").entry_function()
    with pytest.raises(TypeError, match="Target instance, not a string"):
        jit(fn, target="vulkan")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        jit(fn, target=cuda.Target("nvidia.h200_sxm"), foo=1)


def test_jit_keeps_distinct_root_export_descriptions(monkeypatch):
    """Real CPU libraries distinguish names and weight roles in the JIT cache."""
    tensor = types.TensorType.from_shape((8,), storage=expr.StorageKind.HOST)
    params = tuple(expr.Var(name, type=tensor) for name in ("x", "weight", "out"))
    function = core.PrimFunction(
        name="run", params=params, output_count=1, target=cpu.Target(),
        body=core.Sequential(body=(core.Evaluate(callable=core.FunctionRef((), None), args=params),)),
    )
    root = core.Module(name="JitRoot", functions=(function,), entry="run", target=cpu.Target())

    def emit(entry, stmt, ctx):
        return "", ["for (int64_t i = 0; i < x.shape()[0]; ++i) static_cast<float*>(out.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + static_cast<float*>(weight.data_ptr())[i];"]

    monkeypatch.setitem(codegen.registry.entries, (cpu.Target, codegen.Role.EMIT, core.FunctionRef), emit)
    monkeypatch.setitem(passes.registry.entries, (cpu.Target, passes.PassPoint.BEFORE_CODEGEN, HostNamePass), HostNamePass)
    jit.cache_clear()
    try:
        compiled = jit(root)
        assert jit(root) is compiled
        renamed = jit(replace(root, name="OtherRoot"))
        assert renamed.name == "OtherRoot" and renamed is not compiled
        other_entry = replace(function, name="execute")
        changed_entry = jit(replace(root, functions=(other_entry,), entry="execute"))
        assert changed_entry.entry == "execute" and changed_entry is not compiled
        weighted = replace(function, params=(params[0], replace(params[1], is_const=True), params[2]))
        changed_role = jit(replace(root, functions=(weighted,)))
        changed_role.load(runtime.DictResource({"weight": torch.ones(8)}))
        x = torch.zeros(8)
        assert torch.equal(compiled(x, torch.ones(8)), changed_role(x))
        assert changed_role is not compiled
        empty = core.Module(name="EmptyJit", functions=(), target=cpu.Target())
        assert jit(empty) is jit(empty)
        assert jit(replace(empty, name="OtherEmpty")).name == "OtherEmpty"
    finally:
        jit.cache_clear()
