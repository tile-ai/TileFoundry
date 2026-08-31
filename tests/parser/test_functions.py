"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tilefoundry import func, module
from tilefoundry.cli.source import load_namespace
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, SourceSpanMetadata, get_metadata
from tilefoundry.ir.core.module import Module, subtree
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TupleType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.parser import ParseError
from tilefoundry.target import CudaTarget

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
_FIXTURE_SOURCES = tuple(
    path for path in sorted(_FIXTURE_ROOT.rglob("*.py")) if path.name != "__init__.py"
)


def _functions_defined_by(namespace: dict[str, object]) -> tuple[Function, ...]:
    found: list[Function] = []
    seen: set[int] = set()

    def add(function: Function) -> None:
        if id(function) not in seen:
            seen.add(id(function))
            found.append(function)

    for value in namespace.values():
        if isinstance(value, Function):
            add(value)
        elif isinstance(value, Module):
            for module_value in subtree(value):
                for function in module_value.functions:
                    if isinstance(function, Function):
                        add(function)
    return tuple(found)


def test_a_return_annotation_validates_a_layout_inferred_from_the_body() -> None:
    @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2),))
    def annotated(
        x: Tensor[(8, 16), "f32"],
    ) -> Tensor[(8, 16), "f32"]:
        with Mesh(("cta",), layout=(2,), names=("lane",)) as cta:
            return tf.reshard(x, (8 @ cta.lane, 16), "gmem")

    fn = annotated.entry_function()
    assert fn.return_type == fn.body.type
    assert fn.return_type.storage is StorageKind.GMEM
    assert fn.return_type.layout is not None


def test_an_incompatible_return_annotation_reports_annotation_and_body_types() -> None:
    with pytest.raises(ParseError) as raised:

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def annotated(
            x: Tensor[(8, 16), "f32"],
        ) -> Tensor[(8, 16), "f32", None, "smem"]:
            return tf.mul(x, x)

    message = str(raised.value)
    assert "return annotation is not compatible with the inferred body type" in message
    assert "annotation TensorType" in message
    assert "body TensorType" in message
    assert "StorageKind.SMEM" in message
    assert "StorageKind.GMEM" in message
    assert "(role 'func')" in message
    assert "tests/parser/test_functions.py:" in message


def test_a_dispatch_prototype_requires_a_return_annotation() -> None:
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


def test_a_dispatch_prototype_uses_its_tuple_annotation_as_a_variant_contract() -> None:
    size = DimVar("prototype_size", 1, 9)

    @module(
        entry="root",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1),),
    )
    class TupleDispatch:
        @func
        def root(
            x: Tensor[(size,), "f32"],
        ) -> tuple[Tensor[(size,), "f32"], Tensor[(size,), "f32"]]:
            pass

        @root.specialize(DimVarRangePat("prototype_size", 1, 9))
        def both(
            x: Tensor[(size,), "f32"],
        ) -> tuple[Tensor[(size,), "f32"], Tensor[(size,), "f32"]]:
            return x, x

    prototype = TupleDispatch.entry_function()
    assert isinstance(prototype.return_type, TupleType)
    assert prototype.return_type == prototype.variants[0].return_type
    assert prototype.type.return_type == prototype.return_type


def test_a_variant_body_must_satisfy_its_dispatch_return_contract() -> None:
    size = DimVar("prototype_mismatch_size", 1, 9)

    with pytest.raises(ParseError, match="variant body type .*dispatch return contract"):

        @module(
            entry="root",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 1),),
        )
        class IncompatibleVariant:
            @func
            def root(
                x: Tensor[(size,), "f32"],
            ) -> tuple[Tensor[(size,), "f32"], Tensor[(size,), "f32"]]:
                pass

            @root.specialize(DimVarRangePat("prototype_mismatch_size", 1, 9))
            def scalar(x: Tensor[(size,), "f32"]):
                return x


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
    "source",
    _FIXTURE_SOURCES,
    ids=lambda source: str(source.relative_to(_FIXTURE_ROOT)),
)
def test_every_parsed_call_knows_where_it_came_from(source: Path) -> None:
    """Every parser-authored Call reachable from fixture source has a source span."""
    namespace, _ = load_namespace(str(source))
    for function in _functions_defined_by(namespace):
        for expr in collect_exprs(function.body):
            if isinstance(expr, Call):
                assert get_metadata(expr, SourceSpanMetadata) is not None


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
