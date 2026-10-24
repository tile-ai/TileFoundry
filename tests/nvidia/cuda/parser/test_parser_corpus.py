"""NVIDIA inspection, canonical TIR, and named tree inputs share one parser runner."""

from pathlib import Path

import pytest

from tests.core._corpus import cases
from tests.core.parser._parser_case import ParserCase, check_parser_case
from tests.nvidia.cuda.parser._parser_hooks import (
    _child_contexts,
    _composed_tree,
    _derived_geometry,
    _dispatch_tree,
    _empty_child_entry,
    _flash_slices,
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
        corpus=FIXTURES / "logical",
        name="ContextTree",
        hooks=(
            _root_context,
            _child_contexts,
            _empty_child_entry,
        ),
    ),
    ParserCase(
        path=FIXTURES / "logical/gqa_static.py",
        corpus=FIXTURES / "logical",
        name="static_online_attend",
    ),
    ParserCase(
        path=FIXTURES / "logical/hir_composition.py", corpus=FIXTURES / "logical", name="Expert"
    ),
    ParserCase(
        path=FIXTURES / "placed/derived_prefill.py",
        corpus=FIXTURES / "placed",
        name="DerivedPrefill",
        hooks=(_derived_geometry,),
    ),
    ParserCase(
        path=FIXTURES / "placed/flash_split_k_decode.py",
        corpus=FIXTURES / "placed",
        name="FlashSplitKDecode",
        hooks=(_flash_slices,),
    ),
    ParserCase(
        path=FIXTURES / "placed/prefill_decode_attention.py",
        corpus=FIXTURES / "placed",
        name="PrefillDecodeAttention",
        hooks=(_prefill_variants,),
    ),
    ParserCase(
        path=FIXTURES / "placed/rmsnorm.py",
        corpus=FIXTURES / "placed",
        name="RmsnormModule",
        hooks=(_rms_structure,),
    ),
    ParserCase(
        path=FIXTURES / "placed/region_boundaries.py",
        corpus=FIXTURES / "placed",
        name="RegionBoundaries",
        hooks=(_region_escapes,),
    ),
    ParserCase(
        path=FIXTURES / "shapes/scaled_modules.py",
        corpus=FIXTURES / "shapes",
        name="PairedScaledParent",
        hooks=(
            _scaled_common,
            _paired_calls,
        ),
    ),
    ParserCase(
        path=FIXTURES / "shapes/scaled_modules.py",
        corpus=FIXTURES / "shapes",
        name="StackedScaledParent",
        hooks=(
            _scaled_common,
            _stacked_calls,
        ),
    ),
    ParserCase(
        path=FIXTURES / "scaffold/tir_dispatch.py", corpus=FIXTURES / "scaffold", name="Root"
    ),
    ParserCase(
        path=FIXTURES / "scaffold/tir_launches.py",
        corpus=FIXTURES / "scaffold",
        name="Root",
        prints=("launch(child.run,",),
    ),
    ParserCase(
        path=FIXTURES / "inspection/composed.py",
        corpus=FIXTURES / "inspection",
        name="_Composed",
        hooks=(_composed_tree,),
    ),
    ParserCase(
        path=FIXTURES / "inspection/dispatching.py",
        corpus=FIXTURES / "inspection",
        name="_Dispatching",
        hooks=(_dispatch_tree,),
    ),
)
INSPECTION = tuple(
    case
    for case in cases(
        FIXTURES / "inspection",
        case_type=ParserCase,
        expected={key: value for key, value in EXPECTED.items() if key.startswith("inspection/")},
    )
    if case.path.stem not in ("composed", "dispatching")
)
CASES = (
    INSPECTION
    + cases(
        FIXTURES / "tir",
        case_type=ParserCase,
        expected={key: value for key, value in EXPECTED.items() if key.startswith("tir/")},
    )
    + NAMED
)
if len({case.id for case in CASES}) != len(CASES):
    raise ValueError("duplicate NVIDIA parser case ids")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_parser_case(case: ParserCase) -> None:
    check_parser_case(case)
