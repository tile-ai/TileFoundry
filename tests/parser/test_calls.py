"""Parser contracts for explicit variadic input sequences."""

from __future__ import annotations

from typing import get_args, get_origin

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.core import Call, Constant, Tuple, VerifyError
from tilefoundry.ir.core.pattern import Tensor as TensorPattern
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.parser import ParseError


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
