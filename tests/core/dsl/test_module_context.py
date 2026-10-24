"""Module-owned execution context: Target and the ordered Topology hierarchy."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.ir.core import VerifyError


def test_a_raising_class_body_is_not_visible_to_a_later_standalone_func() -> None:
    try:

        @module(topologies=(Topology("leaked", 1),))
        class Failed:
            raise RuntimeError("class body failed")
    except RuntimeError:
        pass

    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):

        @func
        def probe(x: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:
            with Mesh(("leaked",), (1,), ("lane",)) as _mesh:
                return tf.relu(x)


def test_a_failed_body_is_not_inherited_by_a_deeper_module() -> None:
    try:

        @module(topologies=(Topology("leaked", 1),))
        class Failed:
            raise RuntimeError("class body failed")
    except RuntimeError:
        pass

    def define_one_level_deeper():
        @module(entry="probe")
        class Later:
            @func
            def probe(x: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:
                with Mesh(("leaked",), (1,), ("lane",)) as _mesh:
                    return tf.relu(x)

        return Later

    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):
        define_one_level_deeper()
