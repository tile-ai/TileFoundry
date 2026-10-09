"""A lookup reads one element per coordinate, wherever its data sends it."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.analysis import analyze
from tilefoundry.analysis.report import report_data
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.ir.types.layout import Layout
from tilefoundry.target import CudaTarget

ROWS, WIDTH, PICKED = 1024, 64, 4


@module(entry="gather", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class Gather:
    @func
    def gather(
        table: Tensor[(ROWS, WIDTH), "bf16"], index: Tensor[(PICKED,), "i32"]
    ) -> Tensor[(PICKED, WIDTH), "bf16"]:
        return tf.index_select(table, index, dim=0)


@module(entry="gather", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class GatherView:
    @func
    def gather(
        table: Tensor[(1, ROWS, WIDTH), "bf16"], index: Tensor[(PICKED,), "i32"]
    ) -> Tensor[(PICKED, WIDTH), "bf16"]:
        rows = tf.reshape(table, new_shape=(ROWS, WIDTH))
        return tf.index_select(rows, index, dim=0)


def _memory(module_) -> dict:
    result = analyze(module_, module_.entry_function(), analysis=("memory",))
    return report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )["function_records"]["memory"]


def test_a_gather_reads_the_rows_it_selects_not_the_table() -> None:
    """Its traffic is the selected rows; where they may come from is still the table."""
    memory = _memory(Gather)

    rows = PICKED * WIDTH * 2
    assert memory["traffic"]["storage"]["gmem"]["logical"] == {
        "read": rows + PICKED * 4,
        "write": rows,
    }
    assert memory["footprint"]["buffers"]["table"]["gmem"]["logical"] == ROWS * WIDTH * 2


def test_a_gather_from_a_view_lays_out_the_rows_it_selects() -> None:
    selected = GatherView.entry_function().body

    assert selected.type.shape == (PICKED, WIDTH)
    assert selected.type.layout == Layout((PICKED, WIDTH), (WIDTH, 1))
