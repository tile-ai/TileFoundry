"""Pin caller-visible ``tilefoundry.jit`` behavior.

End-to-end tests cover compilation. These tests localize rejected inputs and the
observable cache contract: repeated JIT of one program returns the same module,
while clearing the cache ends that identity. Private key structure is not part of
the contract.
"""

from pathlib import Path

import pytest

from tilefoundry import jit
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget


def test_jit_rejects_bad_arguments() -> None:
    """Raw Python, authored string Targets, and unexpected kwargs all raise."""

    def raw_fn(a):
        return a

    with pytest.raises(TypeError, match="expected Function or Module"):
        jit(raw_fn, target=CudaTarget("nvidia.h200_sxm"))

    rmsnorm = Path(__file__).parents[1] / "fixtures/placed/rmsnorm.py"
    fn = load_authored_ir(f"{rmsnorm}:RmsnormModule").entry_function()
    with pytest.raises(TypeError, match="Target instance, not a string"):
        jit(fn, target="vulkan")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        jit(fn, target=CudaTarget("nvidia.h200_sxm"), foo=1)
