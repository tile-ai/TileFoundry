"""String dtype / reduce-kind authoring surface ([parser §2.4](docs/spec/parser.md#24-pyi-stub-regeneration)).

The DSL surface accepts the string form (`dtype="f32"`, `kind="sum"`); the
parser normalizes it to the IR-canonical descriptor or enum at the call
boundary, and an unknown string raises a clear error. Both authoring forms
print to the same canonical IR.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._source import import_dsl
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.types import DType


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


_HEADER = """
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *
"""


def test_string_and_symbolic_forms_print_to_the_same_canonical_ir() -> None:
    """A string and the descriptor / enum it names are one authoring surface over
    one IR: both normalize at the call boundary and print back as the string, so
    a reader never has to know which spelling the author used."""
    string_dtype = _HEADER + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="bf16")
"""
    descriptor_dtype = _HEADER + """
from tilefoundry.ir.types import DType
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype=DType.bf16)
"""
    string_ir = import_dsl(_dedent(string_dtype))
    descriptor_ir = import_dsl(_dedent(descriptor_dtype))
    assert string_ir.body.target.dtype is DType.bf16
    assert descriptor_ir.body.target.dtype is DType.bf16
    assert as_script(string_ir) == as_script(descriptor_ir)
    assert 'dtype="bf16"' in as_script(string_ir)

    string_kind = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="sum")
"""
    enum_kind = _HEADER + """
from tilefoundry.ir.core.kinds import ReduceKind
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
"""
    assert as_script(import_dsl(_dedent(string_kind))) == as_script(
        import_dsl(_dedent(enum_kind))
    )


_BAD_CALL_DTYPE = _HEADER + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="float32")
"""

_BAD_ANNOTATION_DTYPE = _HEADER + """
@func
def f(x: Tensor[(8,), "float32"]) -> Tensor[(8,), "f32"]:
    return cast(x, dtype="f32")
"""

_BAD_REDUCE_KIND = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="plus")
"""


@pytest.mark.parametrize(
    ("src", "error_type", "message"),
    [
        (_BAD_CALL_DTYPE, VerifyError, r"DType: unknown value 'float32'"),
        (_BAD_ANNOTATION_DTYPE, ValueError, r"DType: unknown value 'float32'"),
        (_BAD_REDUCE_KIND, VerifyError, r"ReduceKind: unknown value 'plus'"),
    ],
    ids=["call-dtype", "annotation-dtype", "reduce-kind"],
)
def test_an_unknown_surface_string_names_the_type_it_failed(
    src: str, error_type: type[Exception], message: str
) -> None:
    """A misspelled string must not fall through as an opaque value: the
    annotation position and the call position both name the type and the value,
    which is the whole benefit of accepting strings at the surface."""
    with pytest.raises(error_type, match=message):
        import_dsl(_dedent(src))
