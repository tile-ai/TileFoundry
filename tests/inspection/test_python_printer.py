"""Inspection printer: what the emitted source carries besides the program.

The program itself round-trips on every corpus model. What has no such witness is
the material around it — an opt-in type annotation, the binding label the emitted
left-hand side carries, and the target header, whose value must rebuild the
*same* Target when the emitted source is executed.
"""

from dataclasses import replace

from tests.fixtures.demo_ir import build_demo
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.inspection.python_printer import (
    _cuda_target_imports,
    _target_str,
)
from tilefoundry.ir.core import BindingMetadata, Call, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.target.cuda import CudaTarget


def test_inspection_types_are_opt_in_same_line_comments():
    fn, _, _ = build_demo()
    canonical = as_script(fn)
    annotated = as_script(fn, options=PythonPrintOptions(show_types=True))

    assert canonical != annotated
    assert "# Tensor[" in annotated
    # Same line as the statement it types, never a line of its own: a comment the
    # reader has to look up is a comment about a different program.
    assert all(
        line.split("# Tensor[")[0].strip()
        for line in annotated.splitlines()
        if "# Tensor[" in line
    )


def test_binding_metadata_names_the_emitted_binding():
    tensor_type = TensorType.scalar(DType.f32)
    source = Var(name="source", type=tensor_type)
    result = Call(
        target=Binary(kind=BinaryKind.ADD),
        args=(source, source),
        type=tensor_type,
        metadata=(BindingMetadata("result"),),
    )
    function = Function.build(
        name="binding_name",
        params=(source,),
        body=result,
        return_type=tensor_type,
    )

    canonical = as_script(function)

    # The label is the emitted name, and the emitted name is the whole record of
    # it: re-parsing reads it back off the left-hand side.
    assert "result = add(source, source)" in canonical

    unbound = Call(
        target=Binary(kind=BinaryKind.ADD),
        args=(source, source),
        type=tensor_type,
    )
    unbound_function = Function.build(
        name="generated_name",
        params=(source,),
        body=unbound,
        return_type=tensor_type,
    )
    # Nothing to name it after, so the printer numbers it.
    assert "v0 = add(source, source)" in as_script(unbound_function)


class TestPythonPrinterTargetRoundTrip:
    """A printed target must rebuild the same target when the emitted source is
    executed: both the architecture and the device side, whether each was
    selected from the installed namespace or supplied directly."""

    @staticmethod
    def _rebuild(target):
        """Execute just the target portion of the emitted header and value."""
        source = "\n".join(
            [
                "from tilefoundry.target import CpuTarget, CudaTarget",
                *_cuda_target_imports(target),
                f"result = {_target_str(target)}",
            ]
        )
        namespace: dict = {}
        exec(compile(source, "<emitted>", "exec"), namespace)  # noqa: S102
        return namespace["result"]

    def test_an_installed_pair_prints_as_the_device_that_names_it(self):
        installed = CudaTarget("nvidia.h200_sxm")
        assert _target_str(installed) == "CudaTarget('nvidia.h200_sxm')"
        assert _cuda_target_imports(installed) == ()
        assert self._rebuild(installed) == installed

    def test_a_directly_supplied_side_is_not_dropped(self):
        """A device ID alone rebuilds only the pair that device declares, so a
        custom value on either side has to print in full or be lost."""
        installed = CudaTarget("nvidia.h200_sxm")

        custom_arch = CudaTarget(
            "nvidia.h200_sxm",
            architecture=replace(installed.architecture, name="sm_90_custom"),
        )
        rebuilt = self._rebuild(custom_arch)
        assert rebuilt == custom_arch
        assert rebuilt.architecture.name == "sm_90_custom"

        custom_device = CudaTarget(
            device=replace(installed.device, name="h200_custom", sm_count=64),
            architecture="nvidia.sm90",
        )
        rebuilt = self._rebuild(custom_device)
        assert rebuilt == custom_device
        assert rebuilt.device.name == "h200_custom"
        assert rebuilt.device.sm_count == 64

        both = CudaTarget(
            architecture=replace(installed.architecture, name="arch_custom"),
            device=replace(installed.device, name="device_custom", sm_count=8),
        )
        rebuilt = self._rebuild(both)
        assert rebuilt == both
        assert rebuilt.architecture.name == "arch_custom"
        assert rebuilt.device.name == "device_custom"
