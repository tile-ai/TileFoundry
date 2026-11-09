"""Inspection printer: what the emitted source carries besides the program.

The program itself round-trips on every corpus model. What has no such witness is
the material around it — an opt-in type annotation, the binding label the emitted
left-hand side carries, and the target header, whose value must rebuild the
*same* Target when the emitted source is executed.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import cli, inspection, target, utils
from tilefoundry.dsl import *
from tilefoundry.ir import core
from tilefoundry_nvidia.cuda import target as cuda_target

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"
RMSNORM = f"{PLACED / 'rmsnorm.py'}:RmsnormModule"


class ProviderTarget(cuda_target.Target):
    name = "tests.printer.cuda"


@dataclass(frozen=True)
class ExtendedCudaArchitecture(cuda_target.Architecture):
    thirteenth_field: int


class PythonPrinterCase(Case):
    """A program checked by the source the printer emits for it."""


@case
class TypesAreOptInComments(PythonPrinterCase):
    path: Path = PLACED / "rmsnorm.py"

    def guard(self, state, problems):
        fn = state.entry_function()
        canonical = inspection.as_script(fn)
        annotated = inspection.as_script(fn, options=inspection.PythonPrintOptions(show_types=True))

        assert canonical != annotated
        assert "# Tensor[" in annotated

        assert all(
            line.split("# Tensor[")[0].strip()
            for line in annotated.splitlines()
            if "# Tensor[" in line
        )


CASES = special(PythonPrinterCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_python_printer_case(case: PythonPrinterCase) -> None:
    check_case(case)


def _rebuild(rendered: utils.PythonExpr):
    source = "\n".join([*rendered.imports, f"result = {rendered.text}"])
    namespace: dict = {}
    exec(compile(source, "<emitted>", "exec"), namespace)
    return namespace["result"]


def test_installed_target_prints_constructor():
    installed = cuda_target.Target("nvidia.h200_sxm")
    rendered = installed.to_python()

    assert rendered == utils.PythonExpr(
        imports=("from tilefoundry_nvidia import cuda",),
        text='cuda.Target("nvidia.h200_sxm")',
    )
    assert _rebuild(rendered) == installed


def test_hardware_values_keep_added_fields():
    installed = cuda_target.Target("nvidia.h200_sxm")
    architecture = ExtendedCudaArchitecture(
        **{
            field.name: getattr(installed.architecture, field.name)
            for field in fields(installed.architecture)
        },
        thirteenth_field=13,
    )
    tgt = cuda_target.Target(
        architecture=architecture,
        device=replace(installed.device, name="h200_custom", sm_count=64),
    )
    rendered = tgt.to_python()

    assert "thirteenth_field=13" in rendered.text
    assert rendered.imports == (
        "from tests.nvidia.cuda.inspection.test_python_printer import ExtendedCudaArchitecture",
        "from tilefoundry.ir.expr import DType",
        "from tilefoundry_nvidia.cuda import Device, Target",
    )
    rebuilt = _rebuild(rendered)
    assert rebuilt == tgt
    assert rebuilt.architecture.thirteenth_field == 13


def test_external_cuda_target_keeps_provider():
    function = cli.load_authored_ir(RMSNORM).entry_function()
    target.register(ProviderTarget)
    tgt = ProviderTarget("nvidia.h200_sxm")
    module = core.Module("external", (function,), function.name, target=tgt)
    canonical = inspection.as_script(module)

    assert "from tests.nvidia.cuda.inspection.test_python_printer import ProviderTarget" in canonical
    assert "from tilefoundry_nvidia.cuda.target import Target" not in canonical
    rebuilt = reload(canonical, "external")
    assert type(rebuilt.target) is ProviderTarget
    assert rebuilt.target == tgt


def test_printer_emits_provider_expression():
    class UninterpretableTarget(target.Target):
        name = "tests.printer.uninterpretable"

        def to_python(self) -> utils.PythonExpr:
            return utils.PythonExpr(("from nobody import unknown",), "unknown !!!")

    function = cli.load_authored_ir(RMSNORM).entry_function()
    canonical = inspection.as_script(
        core.Module(
            "uninterpretable",
            (function,),
            function.name,
            target=UninterpretableTarget(),
        )
    )

    assert "from nobody import unknown" in canonical
    assert '@module(entry="rmsnorm", target=unknown !!!)' in canonical
