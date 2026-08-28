"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re

import pytest

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.inspection import as_script
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.parser import ParseError
from tilefoundry.target import CudaTarget


def test_a_lying_return_annotation_is_ignored_not_rejected() -> None:
    @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
    def annotated(
        x: Tensor[(8, 16), "f32"],
    ) -> Tensor[(8, 16), "f32", None, "smem"]:
        return tf.mul(x, x)

    fn = annotated.entry_function()
    assert fn.return_type == fn.body.type
    assert fn.return_type.storage is StorageKind.GMEM


def test_a_dispatch_prototype_still_requires_a_return_annotation() -> None:
    with pytest.raises(ParseError, match="prototype requires a return annotation"):

        @module(
            entry="root",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 1),),
        )
        class NoReturn:
            @func
            def root(x: Tensor[(8, 16), "f32"]):
                pass


def test_a_storage_the_target_does_not_have_is_refused() -> None:
    refusal = re.escape(
        "storage tmem is not allowed by hardware context ('gmem', 'smem', 'rmem', 'umat')"
    )
    with pytest.raises(ParseError, match=refusal):

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def wrong(x: Tensor[(4,), "f32", None, "tmem"]):
            return x

    with pytest.raises(ParseError, match=refusal):

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def wrong_body(x: Tensor[(4,), "f32"]):
            return tf.zeros(Tensor[(4,), "f32", None, "tmem"])


def test_a_storage_the_target_has_is_accepted() -> None:
    @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
    def fine(x: Tensor[(4,), "f32", None, "smem"]):
        return x

    assert fine.entry_function().params[0].annotation.storage is StorageKind.SMEM

    @func(target=CudaTarget("nvidia.b200_sxm"), topologies=(Topology("cta", 1),))
    def blackwell(x: Tensor[(4,), "f32", None, "tmem"]):
        return x

    assert blackwell.entry_function().params[0].annotation.storage is StorageKind.TMEM


def test_a_function_may_have_a_leading_docstring() -> None:
    @func
    def documented(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        """This is documentation, not an HIR statement."""
        return x

    assert "This is documentation" not in as_script(documented)


def test_a_nested_block_does_not_gain_function_docstring_semantics() -> None:
    with pytest.raises(ParseError):

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def nested(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("unit",)) as _mesh:
                """A string in a with body remains an ordinary statement."""
                return x


@pytest.mark.parametrize(
    ("iterator", "expected"),
    (
        ("tile(10)", "tile(extent) is not supported; use range(extent)"),
        ("tile(1, 2, 3)", "tile() takes 2 arguments (extent, step), got 3"),
        ("range(1, 2, 3, 4)", "range() takes 1 to 3 arguments, got 4"),
        ("steps(1, 2)", "loop iterator must be tile(...) or range(...)"),
    ),
)
def test_a_loop_iterator_states_why_its_arity_is_invalid(iterator: str, expected: str) -> None:
    """The shape-exact syntax must not reduce these to a bare match failure.

    Encoding each iterator's arity is what lets the generated grammar show the
    accepted forms, so the reason is stated before the shape rejects.
    """
    with pytest.raises(ParseError, match=re.escape(expected)):
        if iterator == "tile(10)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in tile(10):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out

        elif iterator == "tile(1, 2, 3)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in tile(1, 2, 3):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out

        elif iterator == "range(1, 2, 3, 4)":

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in range(1, 2, 3, 4):
                    out = tf.add(x[row, :], seed)
                return out

        else:

            @func
            def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
                out = tf.add(seed, seed)
                for row in steps(1, 2):  # noqa: F821
                    out = tf.add(x[row, :], seed)
                return out
