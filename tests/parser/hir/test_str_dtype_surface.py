"""String dtype and reduce-kind surface.

The DSL surface accepts the string form (`dtype="f32"`, `kind="sum"`); the
parser normalizes it to the IR-canonical descriptor or enum at the call
boundary. Both authoring forms print to the same canonical IR
([parser §2.4](docs/spec/parser.md#24-pyi-stub-regeneration)). The unknown
strings that must raise a clear error are rows in ``error_cases.py``.
"""

from __future__ import annotations

import textwrap

from tests._source import import_dsl
from tilefoundry.inspection import as_script
from tilefoundry.ir.types import DType


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


_HEADER = """
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *
"""


def test_string_and_symbolic_forms_print_to_the_same_canonical_ir() -> None:
    """A string and the descriptor / enum it names are one authoring surface over one IR.

    A string and the descriptor / enum it names are one authoring surface over
    one IR: both normalize at the call boundary and print back as the string, so
    a reader never has to know which spelling the author used.
    """
    string_dtype = (
        _HEADER
        + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="bf16")
"""
    )
    descriptor_dtype = (
        _HEADER
        + """
from tilefoundry.ir.types import DType
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype=DType.bf16)
"""
    )
    string_ir = import_dsl(_dedent(string_dtype))
    descriptor_ir = import_dsl(_dedent(descriptor_dtype))
    assert string_ir.body.target.dtype is DType.bf16
    assert descriptor_ir.body.target.dtype is DType.bf16
    assert as_script(string_ir) == as_script(descriptor_ir)
    assert 'dtype="bf16"' in as_script(string_ir)

    string_kind = (
        _HEADER
        + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="sum")
"""
    )
    enum_kind = (
        _HEADER
        + """
from tilefoundry.ir.core.kinds import ReduceKind
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
"""
    )
    assert as_script(import_dsl(_dedent(string_kind))) == as_script(import_dsl(_dedent(enum_kind)))
