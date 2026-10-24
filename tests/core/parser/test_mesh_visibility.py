"""Placed values are consumed only where their positions and storage reach."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._source import import_dsl
from tilefoundry.dsl import *
from tilefoundry.ir.core import VerifyError
from tilefoundry.parser import ParseError

_DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"


def _diagnostic(name: str) -> str:
    return (_DIAGNOSTICS / f"{name}.py").read_text()


def test_unscoped_initialization_is_rejected_at_the_authored_call() -> None:
    source = _diagnostic("unscoped_call_loop")
    line = next(i for i, text in enumerate(source.splitlines(), 1) if "out = tf.zeros" in text)
    with pytest.raises(ParseError, match="runs outside every mesh scope") as raised:
        import_dsl(source, "EscapedRebindInLoop")
    assert f"source.py:{line}:" in str(raised.value)
    assert "tf.zeros" in str(raised.value)
    assert "inside with Mesh(...)" in str(raised.value)


@func
def mesh_scope_helper(x: Tensor[(4, 4), "f32"]):
    return x


def add_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = tf.add(x, x)
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def add_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = tf.add(x, x)
    return result


def call_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = mesh_scope_helper(x)
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def call_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = mesh_scope_helper(x)
    return result


def binary_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = x + x
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def binary_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = x + x
    return result


def unary_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = -x
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def unary_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = -x
    return result


def matmul_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = x @ x
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def matmul_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = x @ x
    return result


def slice_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = x[:, :]
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def slice_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = x[:, :]
    return result


def tuple_before_mesh(
    x: Tensor[(4, 4), "f32"],
    pair: tuple[Tensor[(4, 4), "f32"], Tensor[(4, 4), "f32"]],
):
    result = pair[0]
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def tuple_after_mesh(
    x: Tensor[(4, 4), "f32"],
    pair: tuple[Tensor[(4, 4), "f32"], Tensor[(4, 4), "f32"]],
):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = pair[0]
    return result


def scalar_before_mesh(x: Tensor[(4, 4), "f32"]):
    result = 1 + 2
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    return result


def scalar_after_mesh(x: Tensor[(4, 4), "f32"]):
    with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
        _value = x
    result = 1 + 2
    return result


@pytest.mark.parametrize(
    "program",
    (
        add_before_mesh,
        add_after_mesh,
        call_before_mesh,
        call_after_mesh,
        binary_before_mesh,
        binary_after_mesh,
        unary_before_mesh,
        unary_after_mesh,
        matmul_before_mesh,
        matmul_after_mesh,
        slice_before_mesh,
        slice_after_mesh,
        tuple_before_mesh,
        tuple_after_mesh,
        scalar_before_mesh,
        scalar_after_mesh,
    ),
    ids=lambda program: program.__name__,
)
def test_authored_runtime_expressions_require_a_mesh_scope(program) -> None:
    with pytest.raises(ParseError, match="runs outside every mesh scope"):
        func(program, topologies=(Topology("cta", 1),))


def test_a_return_expression_after_a_mesh_requires_a_scope() -> None:
    with pytest.raises(ParseError, match="runs outside every mesh scope"):
        @module(entry="run", topologies=(Topology("cta", 1),))
        class UnscopedReturn:
            @func
            def run(x: Tensor[(4,), "f32"]):
                with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
                    value = x
                return tf.add(value, value)


def test_logical_hir_and_tir_calls_remain_unscoped() -> None:
    @func
    def logical(x: Tensor[(4,), "f32"]):
        return tf.add(x, x)

    @prim_func
    def tir(x: Tensor[(4,), "f32"]):
        _ptr = T.ptr_of(x)
        with Mesh((Topology("thread", 1),), (1,), names=("unit",)) as _mesh:
            _value = T.ptr_of(x)
        _ptr2 = T.ptr_of(x)

    assert logical.body is not None and tir.body is not None


def test_a_layout_finer_than_the_running_scope_is_rejected() -> None:
    with pytest.raises(
        VerifyError,
        match=(
            "input 0 is laid out more finely than the scope it runs in; "
            "write it inside that scope, or reshard it back first"
        ),
    ):
        import_dsl(_diagnostic("value_escapes"), "ValueEscapes")


def test_rmem_does_not_reach_finer_units_than_its_layout() -> None:
    with pytest.raises(
        VerifyError,
        match=(
            "input 0 is laid out more coarsely and kept in rmem, which does not "
            "reach the units this runs on; reshard it to smem or gmem first"
        ),
    ):
        import_dsl(_diagnostic("coarse_rmem"), "CoarseRmem")


def test_reshard_is_the_one_consumer_allowed_to_move_an_invisible_value() -> None:
    @module(
        entry="run",
        topologies=(Topology("cta", 2), Topology("thread", 4)),
    )
    class ReshardEscape:
        @func
        def run(x: Tensor[(8,), "f32"]):
            with Mesh(("thread",), (4,), names=("lane",)) as lanes:
                value = tf.reshard(x, (8 @ lanes.lane,), "rmem")
            with Mesh(("cta",), (2,), names=("block",)) as blocks:
                return tf.reshard(value, (8 @ blocks.block,), "smem")

    assert ReshardEscape.entry_function().body is not None
