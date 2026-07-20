"""A DSL source value literal parses to an unmaterialized scalar (storage=umat).

A literal such as ``1`` or ``2.0`` carries no committed memory residency; its
``TensorType.storage`` is ``StorageKind.UMAT`` so that, in an op, it abstains
from output storage resolution and the concrete operand anchors the result
regardless of operand order.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.parser.hir_parser import parse_script


@func
def _mul_literal_rhs(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return mul(x, 2.0)


@func
def _mul_literal_lhs(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return mul(2.0, x)


def _literal_arg(call: Call) -> Constant:
    (lit,) = [a for a in call.args if isinstance(a, Constant)]
    return lit


def test_value_literal_is_unmaterialized() -> None:
    """``2.0`` / ``1`` parse to a ``Constant`` whose ``storage`` is ``UMAT``,
    on either side of the op (operand order does not change it)."""
    for fn in (_mul_literal_rhs, _mul_literal_lhs):
        body = fn.body
        assert isinstance(body, Call)
        lit = _literal_arg(body)
        assert lit.type.storage is StorageKind.UMAT, (
            f"literal storage {lit.type.storage} != UMAT in {fn.__name__}"
        )


def test_umat_is_not_an_accepted_surface_storage() -> None:
    """`umat` is compiler-internal: a runtime annotation MUST NOT carry it, so
    the storage surface rejects the string. This keeps an unmaterialized value
    from being smuggled onto a runtime param/return, where it would reach
    codegen without materialization."""
    src = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.dsl.tf import *\n"
        "@func\n"
        "def f(x: Tensor[(8,), 'f32', None, 'umat']) -> Tensor[(8,), 'f32']:\n"
        "    return x\n"
    )
    with pytest.raises(ValueError, match="unknown storage"):
        parse_script(src)
