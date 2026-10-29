"""Binding metadata and unmaterialized types in printed programs."""

import pytest

from tests.core._corpus import reload
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.inspection.analysis_report import _type_text
from tilefoundry.ir.core import BindingMetadata, Call, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.target import CpuTarget


def test_binding_metadata_names_binding():
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
    reserved = as_script(
        Module(
            name="Reserved",
            functions=(reserved_function,),
            entry="reserved_name",
            target=CpuTarget(),
        )
    )
    assert "v0_2 = tf.add(source, source)" in reserved
    assert "v0 = tf.add(v0_2, source)" in reserved
    assert as_script(reload(reserved, "Reserved")) == reserved


def test_umat_type_round_trips() -> None:
    tensor_type = TensorType.scalar(DType.i64, storage=StorageKind.UMAT)
    source = Var(name="source", type=tensor_type)
    function = Function.build(
        name="umat_annotation",
        params=(source,),
        body=source,
        return_type=tensor_type,
    )

    rendered = as_script(
        Module(name="Umat", functions=(function,), entry="umat_annotation", target=CpuTarget())
    )

    assert 'Tensor[(), "i64", "umat"]' in rendered
    rebuilt = reload(rendered, "Umat").entry_function()
    assert rebuilt.params[0].type == tensor_type
    assert rebuilt.return_type == tensor_type


def test_analysis_report_names_umat_storage() -> None:
    assert _type_text(TensorType.umat_scalar()) == "i64[] umat"


@pytest.mark.parametrize("kind, alias", [(BinaryKind.MUL, "mul"), (BinaryKind.MIN, "min")])
def test_kinded_alias_round_trips(kind, alias):
    tensor_type = TensorType.scalar(DType.f32)
    source = Var(name="source", type=tensor_type)
    function = Function.build(
        name="alias_round_trip", params=(source,),
        body=Call(target=Binary(kind=kind), args=(source, source), type=tensor_type),
        return_type=tensor_type,
    )
    rendered = as_script(Module(
        name="Aliases", functions=(function,), entry=function.name, target=CpuTarget(),
    ))
    assert f"tf.{alias}(source, source)" in rendered
    rebuilt = reload(rendered, "Aliases").entry_function()
    assert isinstance(rebuilt.body.target, Binary)
    assert rebuilt.body.target.kind is kind
    assert as_script(reload(rendered, "Aliases")) == rendered
