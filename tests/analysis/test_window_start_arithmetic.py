"""Where a window starts is index arithmetic, which costs no work and moves nothing."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.analysis import analyze
from tilefoundry.analysis.report import report_data
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import Layout
from tilefoundry.target import CudaTarget

SCALAR = Layout((), ())


@module(
    entry="walk",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class LoopIndexWindow:
    @func
    def walk(index: Tensor[(64,), "i32"]) -> Tensor[(), "i32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                total = tf.schedule(
                    (tf.reshape(index[0:1], new_shape=()),), op=T.copy(rmem_layout=SCALAR)
                )
                for kp in tile(64, 8):
                    base = kp + 0
                    for r in range(8):
                        total = total + tf.schedule(
                            (tf.reshape(index[base + r : base + r + 1], new_shape=()),),
                            op=T.copy(rmem_layout=SCALAR),
                        )
                return total


def test_a_window_at_a_loop_offset_costs_only_what_it_reads() -> None:
    """64 loaded indices summed and `kp + 0` once per tile: 72 integer additions.

    The start `base + r` of each window adds none.
    """
    result = analyze(LoopIndexWindow, LoopIndexWindow.entry_function(), analysis=("compute-cost",))
    cost = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )["function_records"]["compute-cost"]

    assert cost["other_ops"]["integer"]["logical"] == 64 + 8
