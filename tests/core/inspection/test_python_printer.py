"""Inspection printer: what the emitted source carries besides the program.

The program itself round-trips on every corpus model. What has no such witness is
the material around it — an opt-in type annotation, the binding label the emitted
left-hand side carries, and the target header, whose value must rebuild the
*same* Target when the emitted source is executed.
"""

from tests.core._source import import_dsl
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.inspection.analysis_report import _type_text
from tilefoundry.ir.core import BindingMetadata, Call, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary


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

    assert "result = tf.add(source, source)" in canonical

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

    assert "v0 = tf.add(source, source)" in as_script(unbound_function)

    named_later = Call(
        target=Binary(kind=BinaryKind.ADD), args=(unbound, source), type=tensor_type,
        metadata=(BindingMetadata("v0"),),
    )
    reserved_function = Function.build(
        name="reserved_name", params=(source,), body=named_later, return_type=tensor_type,
    )
    reserved = as_script(reserved_function)
    assert "v0_2 = tf.add(source, source)" in reserved
    assert "v0 = tf.add(v0_2, source)" in reserved
    assert as_script(import_dsl(reserved)) == reserved


def test_umat_type_is_printed_and_round_trips() -> None:
    tensor_type = TensorType.scalar(DType.i64, storage=StorageKind.UMAT)
    source = Var(name="source", type=tensor_type)
    function = Function.build(
        name="umat_annotation",
        params=(source,),
        body=source,
        return_type=tensor_type,
    )

    rendered = as_script(function)

    assert 'Tensor[(), "i64", "umat"]' in rendered
    rebuilt = import_dsl(rendered)
    assert rebuilt.params[0].type == tensor_type
    assert rebuilt.return_type == tensor_type


def test_analysis_report_names_umat_storage() -> None:
    assert _type_text(TensorType.umat_scalar()) == "i64[] umat"
