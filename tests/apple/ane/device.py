"""Run a compiled ANE module on this machine's neural engine.

The engine reads an image only from a cache directory that root owns, so a run
needs a ``sudo`` credential that is already valid; without one ``available`` is
false and a caller stops after code generation.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tilefoundry import codegen, runtime
from tilefoundry.ir import core
from tilefoundry_apple.ane import runtime as ane_runtime


@functools.cache
def available() -> bool:
    """Whether this machine can place an image in the engine's cache without a prompt."""
    if sys.platform != "darwin" or os.environ.get("TILEFOUNDRY_ANE_OFFLINE") == "1":
        return False
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def emit(lowered: core.Module) -> str:
    """Generate the entry's translation unit, which produces and validates its image."""
    fn = lowered.entry_function()
    signature = codegen.signature.called_as(fn, codegen.mangle(lowered, fn), target=fn.target)
    ctx = codegen.EmitContext(codegen.registry, symbols={id(fn): signature}, target=fn.target)
    return fn.target.resolve(codegen.LinkableModule, lowered, (fn,), ctx).source


def run(module, lowered: core.Module, inputs: tuple):
    """Load *module*, provisioning the image when asked, call it on *inputs* and close it.

    The inputs whose parameter is constant are the weights; the rest are call arguments.
    The trailing parameters are the outputs, which have no input.
    """
    bound = tuple(zip(lowered.entry_function().params, inputs, strict=False))
    weights = {param.name: value for param, value in bound if param.is_const}
    arguments = [value for param, value in bound if not param.is_const]
    cached = None
    try:
        try:
            module.load(runtime.DictResource(weights))
        except ane_runtime.ProvisionRequiredError as request:
            cached = request.path
            with tempfile.TemporaryDirectory() as workdir:
                staged = Path(workdir) / "model.hwx"
                staged.write_bytes(request.image)
                subprocess.run(["sudo", "-n", "mkdir", "-p", str(cached.parent)], check=True)
                subprocess.run(["sudo", "-n", "cp", str(staged), str(cached)], check=True)
            module.load(runtime.DictResource(weights))
        return module(*arguments)
    finally:
        module.close()
        if cached is not None:
            subprocess.run(["sudo", "-n", "rm", "-rf", str(cached.parent)], check=True)
