"""Parser contracts for explicit variadic input sequences."""

from __future__ import annotations

from typing import get_args, get_origin

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.core import Call, Constant, Tuple, VerifyError
from tilefoundry.ir.core.pattern import Tensor as TensorPattern
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Topology
from tilefoundry.parser import ParseError
from tilefoundry.target import CudaTarget


def test_matmul_layout_literals_are_parser_checked() -> None:
    assert get_args(MatMul.a_layout.annotation) == ("MK", "KM")
    assert get_args(MatMul.b_layout.annotation) == ("NK", "KN")

    @func
    def default_layout(
        a: Tensor[(2, 3), "f32"], b: Tensor[(3, 4), "f32"]
    ) -> Tensor[(2, 4), "f32"]:
        return tf.matmul(a, b)

    assert isinstance(default_layout.body, Call)
    assert isinstance(default_layout.body.target, MatMul)
    assert (default_layout.body.target.a_layout, default_layout.body.target.b_layout) == (
        "MK",
        "KN",
    )

    with pytest.raises(ParseError, match="b_layout must be one of 'NK', 'KN'"):

        @func
        def invalid_layout(
            a: Tensor[(2, 3), "f32"], b: Tensor[(3, 4), "f32"]
        ) -> Tensor[(2, 4), "f32"]:
            return tf.matmul(a, b, b_layout="bad")


def test_variadic_list_and_tuple_literals_flatten_to_call_args() -> None:
    for annotation in (Concat.inputs.annotation, Stack.inputs.annotation):
        assert get_origin(annotation) is tuple
        assert get_args(annotation) == (TensorPattern,)

    @func
    def from_list(
        a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
    ) -> Tensor[(2, 4), "f32"]:
        return tf.concat([a, b], axis=0)

    @func
    def from_tuple(
        a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
    ) -> Tensor[(2, 4), "f32"]:
        return tf.concat((a, b), axis=0)

    for function in (from_list, from_tuple):
        assert isinstance(function.body, Call)
        assert isinstance(function.body.target, Concat)
        assert len(function.body.args) == 2


def test_stack_uses_the_same_variadic_list_contract() -> None:
    @func
    def stack_rows(
        a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]
    ) -> Tensor[(2, 4), "f32"]:
        return tf.stack([a, b], axis=0)

    assert isinstance(stack_rows.body, Call)
    assert isinstance(stack_rows.body.target, Stack)
    assert len(stack_rows.body.args) == 2


def test_an_empty_variadic_list_reaches_the_operation_verifier() -> None:
    with pytest.raises(VerifyError, match="Concat requires at least one input"):

        @func
        def empty(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
            return tf.concat([], axis=0)


def test_a_static_range_list_comprehension_expands_in_source_order() -> None:
    @func
    def rows(x: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.stack([x[index, :] for index in range(2)], axis=0)

    assert isinstance(rows.body, Call)
    assert isinstance(rows.body.target, Stack)
    assert len(rows.body.args) == 2
    indices = []
    for row in rows.body.args:
        assert isinstance(row, Call) and isinstance(row.target, Reshape)
        sliced = row.args[0]
        assert isinstance(sliced, Call) and isinstance(sliced.target, Slice)
        starts = sliced.args[1]
        assert isinstance(starts, Tuple)
        start = starts.elements[0]
        assert isinstance(start, Constant)
        indices.append(start.value)
    assert indices == [0, 1]


def test_variadic_direct_positional_inputs_are_rejected() -> None:
    with pytest.raises(ParseError, match="require exactly one list, tuple"):

        @func
        def direct(
            a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
        ) -> Tensor[(2, 4), "f32"]:
            return tf.concat(a, b, axis=0)

    with pytest.raises(ParseError, match="require exactly one list, tuple"):

        @func
        def positional_attribute(
            a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
        ) -> Tensor[(2, 4), "f32"]:
            return tf.concat([a, b], 0)


def test_variadic_generator_and_starred_inputs_name_the_unsupported_form() -> None:
    with pytest.raises(ParseError, match="do not support generator expressions"):

        @func
        def generator(x: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.stack((x[index] for index in range(2)), axis=0)

    with pytest.raises(ParseError, match="do not support starred expansion"):

        @func
        def starred(
            a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
        ) -> Tensor[(2, 4), "f32"]:
            return tf.concat([a, *[b]], axis=0)

    with pytest.raises(ParseError, match="require an explicit list, tuple"):

        @func
        def expanded(
            a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
        ) -> Tensor[(2, 4), "f32"]:
            parts = (a, b)
            return tf.concat(*parts, axis=0)


def test_variadic_sequence_names_are_not_implicitly_expanded() -> None:
    with pytest.raises(ParseError, match="require an explicit list, tuple"):

        @func
        def named(
            a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
        ) -> Tensor[(2, 4), "f32"]:
            parts = (a, b)
            return tf.concat(parts, axis=0)


@pytest.mark.parametrize(
    "program, message",
    [
        ("multiple", "exactly one generator"),
        ("target", "simple Name target"),
        ("filter", "does not support filters"),
        ("iterator", "range with 1 to 3 static integer arguments"),
        ("bound", "range arguments must be static integers"),
    ],
)
def test_unsupported_list_comprehension_shapes_are_named(program: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        if program == "multiple":

            @func
            def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.concat(
                    [x for first in range(1) for second in range(2)], axis=0
                )

        elif program == "target":

            @func
            def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
                return tf.concat([x for first, second in range(1)], axis=0)

        elif program == "filter":

            @func
            def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
                return tf.concat([x for index in range(1) if index], axis=0)

        elif program == "iterator":

            @func
            def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
                return tf.concat([x for index in (0,)], axis=0)

        else:

            @func
            def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
                return tf.concat([x for index in range(x)], axis=0)


def test_a_python_float_operand_adopts_the_tensor_dtype() -> None:
    @func
    def add_literal(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return x + 1.0

    assert isinstance(add_literal.body, Call)
    constant = next(arg for arg in add_literal.body.args if isinstance(arg, Constant))
    assert constant.type.dtype is DType.bf16


def test_a_negated_python_float_stays_weak() -> None:
    @func
    def scale(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return x * -0.5

    assert isinstance(scale.body, Call)
    constant = next(arg for arg in scale.body.args if isinstance(arg, Constant))
    assert (constant.type.dtype, constant.value) == (DType.bf16, -0.5)


def test_a_python_integer_operand_is_not_adopted() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch \\(bf16 vs i64\\)"):

        @func
        def add_integer(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
            return x + 1


def test_two_tensor_operands_are_never_promoted() -> None:
    with pytest.raises(VerifyError, match="dtype mismatch \\(bf16 vs f32\\)"):

        @func
        def mixed(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "f32"]:
            return x + tf.cast(x, "f32")


def test_the_at_operator_builds_a_matmul() -> None:
    @func
    def infix(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]) -> Tensor[(2, 3), "bf16"]:
        return a @ b

    @func
    def spelled(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]) -> Tensor[(2, 3), "bf16"]:
        return tf.matmul(a, b)

    assert isinstance(infix.body, Call)
    assert isinstance(infix.body.target, MatMul)
    assert infix.body.type == spelled.body.type


def test_an_unresolved_callee_is_named() -> None:
    with pytest.raises(VerifyError, match="unsupported call 'tf.no_such_op'"):

        @func
        def missing(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
            return tf.no_such_op(x, x)


def test_placement_at_and_matmul_at_coexist_in_one_function() -> None:
    """The two spellings of ``@`` are told apart by position, not by shape."""

    @module(
        entry="placed_matmul",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2),),
    )
    class PlacedMatMul:
        @func
        def placed_matmul(x: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
            with Mesh(("cta",), (2,), ("tile",)) as mesh:
                x_local = tf.reshard(x, (2 @ mesh.tile, 4), "rmem")
                b_local = tf.reshard(b, (4, 3), "rmem")
                return x_local @ b_local

    body = PlacedMatMul.entry_function().body
    assert isinstance(body, Call)
    assert isinstance(body.target, MatMul)
    assert any(isinstance(argument.target, Reshard) for argument in body.args)
