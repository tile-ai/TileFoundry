"""Inspection printer: what the emitted source carries besides the program.

The program itself round-trips on every corpus model. What has no such witness is
the material around it — an opt-in type annotation, the binding label the emitted
left-hand side carries, and the target header, whose value must rebuild the
*same* Target when the emitted source is executed.
"""

from dataclasses import dataclass, fields, replace

from tests._source import import_dsl
from tests.fixtures.demo_ir import build_demo
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.ir.core import BindingMetadata, Call, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.target import Target, register_target
from tilefoundry.target.cuda import CudaArchitecture
from tilefoundry.target.cuda import CudaTarget as BuiltinCudaTarget
from tilefoundry.utils.python_source import PythonExpr


class CudaTarget(BuiltinCudaTarget):
    name = "tests.printer.cuda"


@dataclass(frozen=True)
class ExtendedCudaArchitecture(CudaArchitecture):
    thirteenth_field: int


def test_inspection_types_are_opt_in_same_line_comments():
    fn, _, _ = build_demo()
    canonical = as_script(fn)
    annotated = as_script(fn, options=PythonPrintOptions(show_types=True))

    assert canonical != annotated
    assert "# Tensor[" in annotated

    assert all(
        line.split("# Tensor[")[0].strip() for line in annotated.splitlines() if "# Tensor[" in line
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

    assert "v0 = add(source, source)" in as_script(unbound_function)


def _rebuild(rendered: PythonExpr):
    source = "\n".join([*rendered.imports, f"result = {rendered.text}"])
    namespace: dict = {}
    exec(compile(source, "<emitted>", "exec"), namespace)  # noqa: S102
    return namespace["result"]


def test_an_installed_target_reports_its_public_constructor():
    installed = BuiltinCudaTarget("nvidia.h200_sxm")
    rendered = installed.to_python()

    assert rendered == PythonExpr(
        imports=("from tilefoundry.target import CudaTarget",),
        text='CudaTarget("nvidia.h200_sxm")',
    )
    assert _rebuild(rendered) == installed


def test_direct_hardware_values_keep_added_fields_in_canonical_source():
    installed = BuiltinCudaTarget("nvidia.h200_sxm")
    architecture = ExtendedCudaArchitecture(
        **{
            field.name: getattr(installed.architecture, field.name)
            for field in fields(installed.architecture)
        },
        thirteenth_field=13,
    )
    target = BuiltinCudaTarget(
        architecture=architecture,
        device=replace(installed.device, name="h200_custom", sm_count=64),
    )
    rendered = target.to_python()

    assert "thirteenth_field=13" in rendered.text
    assert rendered.imports == (
        "from tests.inspection.test_python_printer import ExtendedCudaArchitecture",
        "from tilefoundry.ir.types import DType",
        "from tilefoundry.target.cuda import CudaDevice, CudaTarget",
    )
    rebuilt = _rebuild(rendered)
    assert rebuilt == target
    assert rebuilt.architecture.thirteenth_field == 13


def test_external_same_named_cuda_target_keeps_its_provider_in_module_source():
    function, _, _ = build_demo()
    register_target(CudaTarget)
    target = CudaTarget("nvidia.h200_sxm")
    module = Module("external", (function,), function.name, target=target)
    canonical = as_script(module)

    assert "from tests.inspection.test_python_printer import CudaTarget" in canonical
    assert "from tilefoundry.target.cuda import CudaTarget" not in canonical
    rebuilt = import_dsl(canonical)
    assert type(rebuilt.target) is CudaTarget
    assert rebuilt.target == target


def test_printer_emits_provider_expression_without_interpreting_it():
    class UninterpretableTarget(Target):
        name = "tests.printer.uninterpretable"

        def to_python(self) -> PythonExpr:
            return PythonExpr(("from nobody import unknown",), "unknown !!!")

    function, _, _ = build_demo()
    canonical = as_script(
        Module(
            "uninterpretable",
            (function,),
            function.name,
            target=UninterpretableTarget(),
        )
    )

    assert "from nobody import unknown" in canonical
    assert '@module(entry="demo", target=unknown !!!)' in canonical
