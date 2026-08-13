"""A DSL source value literal parses to an unmaterialized scalar (storage=umat).

A literal such as ``1`` or ``2.0`` carries no committed memory residency; its
``TensorType.storage`` is ``StorageKind.UMAT`` so that, in an op, it abstains
from output storage resolution and the concrete operand anchors the result
regardless of operand order. Model bodies that multiply by a literal exercise
that resolution; the same kind is also a legal explicit surface annotation
when a caller wants to preserve that unresolved residency.
"""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare bindings used by @func bodies
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.types import DType

_EPS = 1e-6


def test_umat_is_an_accepted_surface_storage() -> None:
    """An explicit ``umat`` annotation preserves unresolved residency."""
    src = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.dsl.tf import *\n"
        "@func\n"
        "def f(x: Tensor[(8,), 'f32', None, 'umat']) -> Tensor[(8,), 'f32']:\n"
        "    return x\n"
    )
    parsed = import_dsl(src)
    assert parsed.params[0].type.storage.name == "UMAT"
    assert parsed.return_type.storage.name == "GMEM"


@func
def _literal_on_bf16(x: Tensor[(1, 8), "bf16"]) -> Tensor[(1, 8), "bf16"]:
    return add(x, 1e-6)  # noqa: F405


@func
def _captured_float_on_bf16(x: Tensor[(1, 8), "bf16"]) -> Tensor[(1, 8), "bf16"]:
    return add(x, _EPS)  # noqa: F405


def test_a_python_float_scalar_takes_the_dtype_of_the_operand_it_meets() -> None:
    """A float is bf16 throughout, spelled inline or captured by name.

    A float is bf16 throughout, spelled inline or captured by name; an integer
    keeps its own dtype and still reports the mismatch.
    """
    for fn in (_literal_on_bf16, _captured_float_on_bf16):
        assert fn.body.args[1].type.dtype == DType.bf16, fn.name
        assert fn.body.type.dtype == DType.bf16, fn.name

    integer = (
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.dsl.tf import *\n"
        "@func\n"
        "def f(x: Tensor[(1, 8), 'f32']) -> Tensor[(1, 8), 'f32']:\n"
        "    return mul(x, 2)\n"
    )
    with pytest.raises(VerifyError, match="dtype mismatch"):
        import_dsl(integer)
