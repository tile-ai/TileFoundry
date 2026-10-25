"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.parser import ParseError

FIXTURES = Path(__file__).parents[1] / "fixtures"
PARSER = FIXTURES / "parser"
DIAGNOSTICS = FIXTURES / "diagnostics"


def test_a_return_annotation_validates_a_layout_inferred_from_the_body() -> None:
    annotated = load_authored_ir(str(PARSER / "reshard_return_annotation.py"))

    fn = annotated.entry_function()
    assert fn.return_type == fn.body.type
    assert fn.return_type.storage is StorageKind.GMEM
    assert fn.return_type.layout is not None


def test_an_incompatible_return_annotation_reports_annotation_and_body_types() -> None:
    with pytest.raises(ParseError) as raised:
        load_authored_ir(str(DIAGNOSTICS / "parser_return_annotation_storage_mismatch.py"))

    message = str(raised.value)
    assert "return annotation is not compatible with the inferred body type" in message
    assert "annotation TensorType" in message
    assert "body TensorType" in message
    assert "StorageKind.SMEM" in message
    assert "StorageKind.GMEM" in message
    assert "(role 'func')" in message
    assert (
        "tests/nvidia/cuda/fixtures/diagnostics/parser_return_annotation_storage_mismatch.py:"
        in message
    )


def test_a_dispatch_prototype_requires_a_return_annotation() -> None:
    with pytest.raises(ParseError, match="prototype requires a return annotation"):
        load_authored_ir(str(DIAGNOSTICS / "parser_dispatch_prototype_without_return.py"))


def test_a_dispatch_prototype_uses_its_tuple_annotation_as_a_variant_contract() -> None:
    TupleDispatch = load_authored_ir(str(PARSER / "tuple_dispatch_prototype.py"))

    prototype = TupleDispatch.entry_function()
    assert isinstance(prototype.return_type, TupleType)
    assert prototype.return_type == prototype.variants[0].return_type
    assert prototype.type.return_type == prototype.return_type


def test_a_variant_body_must_satisfy_its_dispatch_return_contract() -> None:
    with pytest.raises(ParseError, match="variant body type .*dispatch return contract"):
        load_authored_ir(str(DIAGNOSTICS / "parser_dispatch_variant_return_mismatch.py"))


def test_a_storage_the_target_does_not_have_is_refused() -> None:
    refusal = re.escape(
        "storage tmem is not allowed by hardware context ('gmem', 'smem', 'rmem', 'umat')"
    )
    with pytest.raises(ParseError, match=refusal):
        load_authored_ir(str(DIAGNOSTICS / "parser_param_storage_not_on_target.py"))

    with pytest.raises(ParseError, match=refusal):
        load_authored_ir(str(DIAGNOSTICS / "parser_body_storage_not_on_target.py"))


def test_a_storage_the_target_has_is_accepted() -> None:
    fine = load_authored_ir(str(PARSER / "smem_param_h200.py"))

    assert fine.entry_function().params[0].annotation.storage is StorageKind.SMEM

    blackwell = load_authored_ir(str(PARSER / "tmem_param_b200.py"))

    assert blackwell.entry_function().params[0].annotation.storage is StorageKind.TMEM


def test_a_nested_block_does_not_gain_function_docstring_semantics() -> None:
    with pytest.raises(ParseError):
        load_authored_ir(str(DIAGNOSTICS / "parser_with_body_docstring.py"))
