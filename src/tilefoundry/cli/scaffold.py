"""Build an editable single-device CUDA project from authored IR."""

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.compile import scaffold
from tilefoundry.ir.core.module import module_functions, target_of
from tilefoundry.target import CpuTarget, CudaTarget


def run_scaffold(source: str, out_dir: str) -> int:
    """Validate the output and device boundary before generating project files."""
    destination = Path(out_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"scaffold: output directory {destination} must be empty")
    module = load_authored_ir(source)
    targets = [module.resolve_target()]
    for fn in module_functions(module):
        target = target_of(module, fn)
        if not isinstance(target, CpuTarget):
            targets.append(target)
    if any(
        not isinstance(device, CudaTarget) or device.device_count is not None for device in targets
    ):
        raise ValueError(
            "scaffold supports only single-device CUDA targets without device_count"
        )
    scaffold(module, destination)
    return 0
