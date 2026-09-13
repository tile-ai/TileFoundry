from pathlib import Path

from tests.fixtures.hir.tensor_types import TensorTypes
from tilefoundry.analysis import analyze
from tilefoundry.analysis.metadata import BufferFootprint, LoopFootprintMetadata
from tilefoundry.inspection import as_script
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.inspection.values import render_comment


def test_loop_footprint_comment_aggregates_by_level() -> None:
    record = LoopFootprintMetadata(
        footprints=(
            BufferFootprint("a", "gmem", 8, 16, 24),
            BufferFootprint("b", "gmem", 4, 8, 12),
            BufferFootprint("c", "rmem", 2, 2, 2),
        ),
        known=True,
    )
    rendered = render_comment(record)
    assert rendered == "loop-footprint footprints=gmem:12,rmem:2 status=complete"
    detailed = render_comment(record, opt_in=frozenset({"details"}))
    assert "details=a@gmem:8/16/24,b@gmem:4/8/12,c@rmem:2/2/2" in detailed


def test_loop_nest_typed_and_analyze_goldens() -> None:
    golden_dir = Path(__file__).with_name("golden")
    fn = TensorTypes.entry_function()
    result = analyze(TensorTypes, fn, analysis=("compute-cost", "memory"))
    typed = as_script(result.function)
    analyzed = render_analysis(result).annotated
    assert typed == (golden_dir / "tensor_types.loop_nest.golden").read_text()
    assert analyzed == (golden_dir / "tensor_types.loop_nest.analyze.golden").read_text()

    def strip_comments(source: str) -> str:
        return "\n".join(line.split("  # ", 1)[0].rstrip() for line in source.splitlines())

    assert strip_comments(analyzed) == strip_comments(typed)
