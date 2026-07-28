"""A DSL source value literal parses to an unmaterialized scalar (storage=umat).

A literal such as ``1`` or ``2.0`` carries no committed memory residency; its
``TensorType.storage`` is ``StorageKind.UMAT`` so that, in an op, it abstains
from output storage resolution and the concrete operand anchors the result
regardless of operand order. Model bodies that multiply by a literal exercise
that resolution; what no model can exercise is the surface refusing to name the
internal kind.
"""

from __future__ import annotations

import pytest

from tilefoundry.parser.hir_parser import parse_script


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
