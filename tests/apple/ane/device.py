"""Compile a lowered ANE entry and run it on this machine's neural engine.

The engine reads an image only from a cache directory that root owns, so a run
needs a ``sudo`` credential that is already valid; without one ``available`` is
false and a caller checks the image bytes alone.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tilefoundry
from tilefoundry import runtime, utils
from tilefoundry.ir import core
from tilefoundry_apple.ane import runtime as ane_runtime
from tilefoundry_apple.ane.codegen import module as ane_module


@functools.cache
def available() -> bool:
    """Whether this machine can place an image in the engine's cache without a prompt."""
    if sys.platform != "darwin" or os.environ.get("TILEFOUNDRY_ANE_OFFLINE") == "1":
        return False
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def image_of(lowered: core.Module, constants=None) -> bytes:
    """The container the lowered entry compiles to."""
    return ane_module.function_image(lowered.entry_function(), constants).to_bytes()


def launch(lowered: core.Module, inputs: tuple):
    """Load through the runtime, provision in the test, then evaluate twice and close."""
    fn = lowered.entry_function()
    activation_inputs = tuple(
        value
        for param, value in zip(fn.params, inputs, strict=False)
        if not getattr(param, "is_const", False)
    )
    resource = runtime.DictResource(
        {
            param.name: value
            for param, value in zip(fn.params, inputs, strict=False)
            if getattr(param, "is_const", False)
        }
    )
    with tempfile.TemporaryDirectory() as workdir:
        module = tilefoundry.compile(lowered)
        exported = Path(workdir) / "entry.so"
        module.export_library(exported)
        module.close()
        module = runtime.CompiledModule.from_library(exported)
        constants = {
            param.name: value.numpy().tobytes()
            for param, value in zip(fn.params, inputs, strict=False)
            if getattr(param, "is_const", False)
        }
        expected_image = image_of(lowered, constants)
        native_load = module.functions["load"]

        def load_validated(token):
            assert bytes(module.functions["image"](token)) == expected_image
            return native_load(token)

        module.functions["load"] = load_validated
        cached = None
        try:
            try:
                module.load(resource)
            except ane_runtime.ProvisionRequiredError as request:
                assert request.image == expected_image
                cached = request.path
                staged = Path(workdir) / "model.hwx"
                staged.write_bytes(request.image)
                subprocess.run(["sudo", "-n", "mkdir", "-p", str(cached.parent)], check=True)
                subprocess.run(["sudo", "-n", "cp", str(staged), str(cached)], check=True)
                module.load(resource)
            first = module.forward(*activation_inputs)
            second = utils.torch.empty_like(first)
            assert module(*activation_inputs, second) is second
            assert utils.torch.equal(first, second)
            return first
        finally:
            module.close()
            if cached is not None:
                subprocess.run(["sudo", "-n", "rm", "-rf", str(cached.parent)], check=True)
