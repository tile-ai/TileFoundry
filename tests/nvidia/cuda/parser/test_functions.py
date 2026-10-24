"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.core.parser._fixtures import check_parsed_call_sources
from tilefoundry.dsl import *
from tilefoundry.parser import ParseError
from tilefoundry_nvidia import CudaTarget

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
_FIXTURES_THAT_REFUSE_TO_PARSE = {_FIXTURE_ROOT / "schedule" / "plain" / "gemm_relu_gemm_smem_staged.py"}
_FIXTURE_SOURCES = tuple(
    path
    for path in sorted(_FIXTURE_ROOT.rglob("*.py"))
    if path.name != "__init__.py" and path not in _FIXTURES_THAT_REFUSE_TO_PARSE
)


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
    assert "tests/nvidia/cuda/parser/test_functions.py:" in message


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
    size = DimVar("prototype_size", 1, 8)

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

        @root.specialize(RangePattern("prototype_size", 1, 8))
        def both(
            x: Tensor[(size,), "f32"],
        ) -> tuple[Tensor[(size,), "f32"], Tensor[(size,), "f32"]]:
            return x, x

    prototype = TupleDispatch.entry_function()
    assert isinstance(prototype.return_type, TupleType)
    assert prototype.return_type == prototype.variants[0].return_type
    assert prototype.type.return_type == prototype.return_type


def test_a_variant_body_must_satisfy_its_dispatch_return_contract() -> None:
    size = DimVar("prototype_mismatch_size", 1, 8)

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

            @root.specialize(RangePattern("prototype_mismatch_size", 1, 8))
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


def test_a_nested_block_does_not_gain_function_docstring_semantics() -> None:
    with pytest.raises(ParseError):

        @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
        def nested(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("unit",)) as _mesh:
                """A string in a with body remains an ordinary statement."""
                return x


@pytest.mark.parametrize("source", _FIXTURE_SOURCES, ids=lambda source: str(source.relative_to(_FIXTURE_ROOT)))
def test_every_parsed_call_knows_where_it_came_from(source):
    check_parsed_call_sources(source)
