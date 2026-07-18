"""String dtype / reduce-kind authoring surface (parser.md §2.4).

The DSL surface accepts the string form (`dtype="f32"`, `kind="sum"`); the
parser normalizes it to the IR-canonical descriptor or enum at the call
boundary, and an unknown string raises a clear error. Both authoring forms
print to the same canonical IR.
"""

from __future__ import annotations

import textwrap

import pytest

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.types import DType
from tilefoundry.parser.hir_parser import parse_script


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


_HEADER = """
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *
"""


@pytest.mark.parametrize(
    "name",
    ("f32", "f16", "bf16", "fp8e4m3", "f8e8m0", "f4e2m1", "i32", "i64", "bool"),
)
def test_string_dtype_parses_and_prints_canonically(name: str) -> None:
    src = _HEADER + f"""
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "{name}"]:
    return cast(x, dtype="{name}")
"""
    fn = parse_script(_dedent(src))

    assert fn.body is not None
    assert fn.body.target.dtype is getattr(DType, name)
    assert f'dtype="{name}"' in as_script(fn)


def test_string_and_descriptor_dtype_forms_are_equivalent() -> None:
    string_form = _HEADER + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="bf16")
"""
    descriptor_form = _HEADER + """
from tilefoundry.ir.types import DType
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype=DType.bf16)
"""

    string_ir = parse_script(_dedent(string_form))
    descriptor_ir = parse_script(_dedent(descriptor_form))

    assert string_ir.body is not None
    assert descriptor_ir.body is not None
    assert string_ir.body.target.dtype is DType.bf16
    assert descriptor_ir.body.target.dtype is DType.bf16
    assert as_script(string_ir) == as_script(descriptor_ir)
    assert 'dtype="bf16"' in as_script(string_ir)


def test_string_reduce_kind_parses() -> None:
    src = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="sum")
"""
    fn = parse_script(_dedent(src))
    assert fn.body is not None


def test_string_and_enum_forms_are_equivalent() -> None:
    str_form = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="sum")
"""
    enum_form = _HEADER + """
from tilefoundry.ir.core.kinds import ReduceKind
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
"""
    assert as_script(parse_script(_dedent(str_form))) == as_script(
        parse_script(_dedent(enum_form))
    )


def test_invalid_dtype_string_raises() -> None:
    src = _HEADER + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="float32")
"""
    with pytest.raises(VerifyError, match=r"DType: unknown value 'float32'"):
        parse_script(_dedent(src))


def test_invalid_tensor_annotation_dtype_raises() -> None:
    src = _HEADER + """
@func
def f(x: Tensor[(8,), "float32"]) -> Tensor[(8,), "f32"]:
    return cast(x, dtype="f32")
"""
    with pytest.raises(VerifyError, match=r"DType: unknown value 'float32'"):
        parse_script(_dedent(src))


def test_invalid_reduce_kind_string_raises() -> None:
    src = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="plus")
"""
    with pytest.raises(VerifyError, match=r"ReduceKind: unknown value 'plus'"):
        parse_script(_dedent(src))
