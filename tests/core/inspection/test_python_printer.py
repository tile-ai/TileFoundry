"""Binding metadata and unmaterialized types in printed programs."""

import pytest

from tests.core._corpus import reload
from tilefoundry import inspection
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import math
from tilefoundry.target import cpu


def test_binding_metadata_names_binding():
    tensor_type = TensorType.from_scalar(DType.f32)
    source = expr.Var(name="source", type=tensor_type)
    result = expr.Call(
        target=math.Binary(kind=expr.BinaryKind.ADD),
        args=(source, source),
        type=tensor_type,
        metadata=(expr.BindingMetadata("result"),),
    )
    function = core.Function.from_parts(
        name="binding_name",
        params=(source,),
        body=result,
        return_type=tensor_type,
    )

    canonical = inspection.as_script(function)

    assert "result = tf.add(source, source)" in canonical

    unbound = expr.Call(
        target=math.Binary(kind=expr.BinaryKind.ADD),
        args=(source, source),
        type=tensor_type,
    )
    unbound_function = core.Function.from_parts(
        name="generated_name",
        params=(source,),
        body=unbound,
        return_type=tensor_type,
    )

    assert "v0 = tf.add(source, source)" in inspection.as_script(unbound_function)

    named_later = expr.Call(
        target=math.Binary(kind=expr.BinaryKind.ADD), args=(unbound, source), type=tensor_type,
        metadata=(expr.BindingMetadata("v0"),),
    )
    reserved_function = core.Function.from_parts(
        name="reserved_name", params=(source,), body=named_later, return_type=tensor_type,
    )
    reserved = inspection.as_script(
        core.Module(
            name="Reserved",
            functions=(reserved_function,),
            entry="reserved_name",
            target=cpu.Target(),
        )
    )
    assert "v0_2 = tf.add(source, source)" in reserved
    assert "v0 = tf.add(v0_2, source)" in reserved
    assert inspection.as_script(reload(reserved, "Reserved")) == reserved


def test_umat_type_round_trips() -> None:
    tensor_type = TensorType.from_scalar(DType.i64, storage=StorageKind.UMAT)
    source = expr.Var(name="source", type=tensor_type)
    function = core.Function.from_parts(
        name="umat_annotation",
        params=(source,),
        body=source,
        return_type=tensor_type,
    )

    rendered = inspection.as_script(
        core.Module(name="Umat", functions=(function,), entry="umat_annotation", target=cpu.Target())
    )

    assert 'Tensor[(), "i64", "umat"]' in rendered
    rebuilt = reload(rendered, "Umat").entry_function()
    assert rebuilt.params[0].type == tensor_type
    assert rebuilt.return_type == tensor_type


def test_analysis_report_names_umat_storage() -> None:
    assert inspection.type_text(TensorType.from_umat_scalar()) == "i64[] umat"


@pytest.mark.parametrize("kind, alias", [(expr.BinaryKind.MUL, "mul"), (expr.BinaryKind.MIN, "min")])
def test_kinded_alias_round_trips(kind, alias):
    tensor_type = TensorType.from_scalar(DType.f32)
    source = expr.Var(name="source", type=tensor_type)
    function = core.Function.from_parts(
        name="alias_round_trip", params=(source,),
        body=expr.Call(target=math.Binary(kind=kind), args=(source, source), type=tensor_type),
        return_type=tensor_type,
    )
    rendered = inspection.as_script(core.Module(
        name="Aliases", functions=(function,), entry=function.name, target=cpu.Target(),
    ))
    assert f"tf.{alias}(source, source)" in rendered
    rebuilt = reload(rendered, "Aliases").entry_function()
    assert isinstance(rebuilt.body.target, math.Binary)
    assert rebuilt.body.target.kind is kind
    assert inspection.as_script(reload(rendered, "Aliases")) == rendered
