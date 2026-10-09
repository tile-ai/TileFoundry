"""Lower a row gather: a loaded index starts each window, each row lands in its tile row."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.inspection.python_printer import as_script
from tilefoundry.ir.types import ComposedLayout, Layout, Swizzle
from tilefoundry.schedule import finalize
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.verify import verify_prim_function

ROWS, WIDTH, PICKED = 1024, 64, 8
SW128 = Swizzle(3, 4, 3)
ROW_SMEM = ComposedLayout(SW128, 0, Layout((1, WIDTH), (WIDTH, 1)))
TILE_SMEM = ComposedLayout(SW128, 0, Layout((PICKED, WIDTH), (WIDTH, 1)))
SCALAR = Layout((), ())


@module(
    entry="gather",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class GatherRows:
    @func
    def gather(
        table: Tensor[(ROWS, WIDTH), "bf16"], index: Tensor[(PICKED,), "i32"]
    ) -> Tensor[(PICKED, WIDTH), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _warp:
                tile = tf.zeros(Tensor[(PICKED, WIDTH), "bf16", TILE_SMEM, "smem"])
                for r in range(PICKED):
                    start = tf.cast(
                        tf.schedule(
                            (tf.reshape(index[r : r + 1], new_shape=()),),
                            op=T.copy(rmem_layout=SCALAR),
                        ),
                        "i64",
                    )
                    row = tf.schedule(
                        (table[start : start + 1, :],), op=T.copy_async(smem_layout=ROW_SMEM)
                    )
                    tile = tf.insert_slice(tile, row, (r, 0))
                return tf.schedule((tile,), op=T.copy_async_tensor())


def test_a_loaded_index_starts_the_window_it_reads() -> None:
    """The index is loaded and widened once per row, and the window reads that register."""
    function = finalize(GatherRows)
    verify_prim_function(function)
    text = as_script(function)

    assert "T.schedule(" not in text
    assert "T.cast(value_3, value_2, dtype='i64')" in text
    assert "T.ptr_of(table[value_2:value_2 + 1, 0:0 + 64])" in text


def test_a_scheduled_row_lands_in_its_row_of_the_staged_tile() -> None:
    """The cp.async writes the tile row directly; nothing passes through the output."""
    text = as_script(finalize(GatherRows))

    assert "T.ptr_of(tile[r:r + 1, 0:0 + 64])" in text
    assert "T.copy_async(tile_3, window)" in text
    assert "T.copy(out, tile)" not in text
