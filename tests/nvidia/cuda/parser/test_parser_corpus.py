"""NVIDIA inspection, canonical TIR, and named tree inputs share one parser runner."""

from pathlib import Path

import pytest

from tests.core._corpus import cases
from tests.core.parser._parser_case import ParserCase, check_parser_case, unique_parser_cases
from tests.nvidia.cuda.parser._parser_hooks import (
    _child_contexts,
    _composed_tree,
    _derived_geometry,
    _dispatch_tree,
    _empty_child_entry,
    _flash_slices,
    _nan_fill,
    _paired_calls,
    _prefill_variants,
    _region_escapes,
    _rms_structure,
    _root_context,
    _scaled_common,
    _stacked_calls,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
EXPECTED = {
    "inspection/type_printer_sugar": dict(
        golden=FIXTURES / "inspection/type_printer_sugar.printed.txt"
    ),
    "inspection/device": dict(prints=("T.sync(thread)", "T.sync(thread[:])")),
    "inspection/launched": dict(prints=("cluster=(1, 1, 1), dynamic_smem=512)",)),
    "inspection/nan_value_fill": dict(prints=("math.nan",), hooks=(_nan_fill,)),
    "tir/async_sync": dict(canonical=True),
    "tir/e4m3_gemm": dict(canonical=True),
    "tir/mma": dict(canonical=True),
    "tir/split_stage": dict(canonical=True),
    "tir/square": dict(canonical=True),
    "tir/sync": dict(canonical=True),
}
NAMED = (
    ParserCase(
        path=FIXTURES / "logical/module_context.py",
        name="ContextTree",
        hooks=(
            _root_context,
            _child_contexts,
            _empty_child_entry,
        ),
    ),
    ParserCase(path=FIXTURES / "logical/gqa_static.py", name="static_online_attend"),
    ParserCase(
        path=FIXTURES / "placed/derived_prefill.py",
        name="DerivedPrefill",
        hooks=(_derived_geometry,),
    ),
    ParserCase(
        path=FIXTURES / "placed/flash_split_k_decode.py",
        name="FlashSplitKDecode",
        hooks=(_flash_slices,),
    ),
    ParserCase(
        path=FIXTURES / "placed/prefill_decode_attention.py",
        name="PrefillDecodeAttention",
        hooks=(_prefill_variants,),
    ),
    ParserCase(
        path=FIXTURES / "placed/rmsnorm.py",
        name="RmsnormModule",
        hooks=(_rms_structure,),
    ),
    ParserCase(
        path=FIXTURES / "placed/region_boundaries.py",
        name="RegionBoundaries",
        hooks=(_region_escapes,),
    ),
    ParserCase(
        path=FIXTURES / "shapes/scaled_modules.py",
        name="PairedScaledParent",
        hooks=(
            _scaled_common,
            _paired_calls,
        ),
    ),
    ParserCase(
        path=FIXTURES / "shapes/scaled_modules.py",
        name="StackedScaledParent",
        hooks=(
            _scaled_common,
            _stacked_calls,
        ),
    ),
    ParserCase(path=FIXTURES / "scaffold/tir_dispatch.py", name="Root"),
    ParserCase(
        path=FIXTURES / "scaffold/tir_launches.py",
        name="Root",
        prints=("launch(child.run,",),
    ),
    ParserCase(
        path=FIXTURES / "inspection/composed.py",
        name="_Composed",
        hooks=(_composed_tree,),
    ),
    ParserCase(
        path=FIXTURES / "inspection/dispatching.py",
        name="_Dispatching",
        hooks=(_dispatch_tree,),
    ),
)
CASES = (
    tuple(
        case
        for case in cases(
            FIXTURES / "inspection", FIXTURES / "tir", case_type=ParserCase, expected=EXPECTED
        )
        if case.path.stem not in ("composed", "dispatching")
    )
    + NAMED
)
CASES = unique_parser_cases(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_parser_case(case: ParserCase) -> None:
    check_parser_case(case)
