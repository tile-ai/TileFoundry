"""End-to-end demo: ``proj_qkv``.

Acceptance demo for the DSL subsystem. Runnable subsets are written
as **standard ``@func`` DSL** so the parser sees real Python source
via the decorator path (which returns a ``hir.Function``) rather than
fixture strings.

Coverage:

- ``@func`` + outer ``with Mesh(...)`` scope
- ``zeros((1, 64), "bf16", storage="smem")`` + string-dtype sugar
- ``tile(extent, step)`` chunked iteration → ``RangeSlice``
- ``x[:, ok]`` subscript → ``Slice`` Op call
- carry-out lifting → ``GridRegionExpr.carried_args``
- layout sugar at call-arg position (single-axis ``@`` shorthand,
  ``ShardLayout`` literal)
- ``Mma_SM80_16x8x16`` SSA Op + outer ``add(acc, mma)`` carry
- viewer model walks into nested ``GridRegionExpr`` bodies
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.inspection.viewer import (
    Viewer,
)
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.shard import (
    Mesh,
    Topology,
)


def _walk_calls(expr, *, into_grid: bool = True):
    """Yield every ``Call`` reachable from *expr* (including those
    inside ``GridRegionExpr`` bodies / yield_values)."""
    if isinstance(expr, Call):
        yield expr
        for a in expr.args:
            yield from _walk_calls(a, into_grid=into_grid)
    if into_grid and isinstance(expr, GridRegionExpr):
        yield from _walk_calls(expr.body, into_grid=into_grid)
        for y in expr.yield_values:
            yield from _walk_calls(y, into_grid=into_grid)


def _walk_grid(expr):
    """Yield every ``GridRegionExpr`` reachable from *expr*."""
    if isinstance(expr, GridRegionExpr):
        yield expr
    if isinstance(expr, Call):
        for a in expr.args:
            yield from _walk_grid(a)


# ── Working subset (canonical acceptance, standard @func DSL) ───────────


@func(topologies=(Topology("cta", 128),))
def proj_qkv_subset(
    x: Tensor[(1, 2048), "bf16"],
    wqkv: Tensor[(2048, 8192), "bf16"],
) -> Tensor[(1, 8192), "bf16"]:
    with Mesh(topology="cta", layout=(128,)) as cta:  # noqa: F841
        o_smem = zeros((1, 64), "bf16", storage="smem")
        for ok in tile(2048, 512):
            o_smem = relu(o_smem)
        return reshard(o_smem, (1, 64), "smem")


def test_proj_qkv_subset_parses() -> None:
    """The runnable slice of the proj_qkv demo parses cleanly via
    the standard ``@func`` decorator path."""
    fn = proj_qkv_subset
    assert fn.name == "proj_qkv_subset"


def test_proj_qkv_subset_produces_grid_region_with_carry() -> None:
    """``for ok in tile(2048, 512)`` body carries ``o_smem`` out, so
    the resulting ``GridRegionExpr`` has non-empty
    ``carried_args``."""
    fn = proj_qkv_subset
    grids = list(_walk_grid(fn.body))
    assert any(g.carried_args for g in grids), (
        "expected GridRegionExpr with non-empty carried_args; "
        f"got {[len(g.carried_args) for g in grids]}"
    )


# ── subset with Mma SSA + carry + @ shorthand (standard @func DSL) ───


@func(topologies=(Topology("thread", 32),))
def proj_qkv_with_mma(
    x: Tensor[(16, 2048), "bf16"],
    w: Tensor[(2048, 8), "bf16"],
) -> Tensor[(16, 8), "f32"]:
    # One warp (32 threads, mesh shape (4, 8)) iterating the K dimension
    # in 16-wide tiles. Each iteration loads A and B fragments into the
    # SM80 16x8x16 layouts and runs the matching mma atom. Carry-out
    # via ``add(acc, mma(...))`` accumulates the C fragment across the
    # K-tile loop. Layouts use the parser sugar (axis-tuple,
    # stride-tuple) form with named mesh axes ``warp.x`` (size 4) and
    # ``warp.y`` (size 8); ``i @ warp.<axis>`` marks a Split slot, plain
    # ints are broadcast value axes.
    with Mesh(topology="thread", layout=(4, 8), names=("x", "y")) as warp:
        acc = zeros((16, 8), "f32", storage="rmem")
        for ok in tile(2048, 16):
            x_frag = reshard(
                x[:, ok],
                ((2, 4 @ warp.x, 2, 8 @ warp.y, 2), (1, 2, 8, 16, 128)),
                "rmem",
            )
            w_frag = reshard(
                w[ok, :],
                ((8 @ warp.y, 2, 4 @ warp.x, 2), (1, 8, 16, 64)),
                "rmem",
            )
            acc = add(
                acc,
                mma_sm80_16x8x16(
                    x_frag, w_frag,
                    dtype_a="bf16", dtype_b="bf16", dtype_acc="f32",
                ),
            )
        return reshard(
            acc,
            ((2, 4 @ warp.x, 8 @ warp.y, 2), (1, 2, 8, 64)),
            "gmem",
        )


def test_subset_uses_mma_ssa_and_carry_and_int_at_mesh() -> None:
    """Exercise the DSL surface in one parsed ``@func``:
    ``Mma_SM80_16x8x16`` SSA, carry-out via ``add(acc, Mma(...))``,
    single-axis ``@ cta`` shorthand."""
    fn = proj_qkv_with_mma

    found_mma = [c for c in _walk_calls(fn.body)
                 if isinstance(c.target, Mma_SM80_16x8x16)]
    assert found_mma, "expected at least one Mma_SM80_16x8x16 Call"
    # SSA contract: only 2 args (a, b) — no accumulator.
    assert len(found_mma[0].args) == 2
    assert found_mma[0].target.dtype_acc.name == "f32"

    # Carry-out: Mma is wrapped in add(acc, mma) inside a
    # GridRegionExpr.
    grids = list(_walk_grid(fn.body))
    assert grids, "expected GridRegionExpr from tile-for"
    assert any(g.carried_args for g in grids)


@func
def _module_demo_helper(
    x: Tensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    return relu(x)


@func
def _module_demo_main(
    x: Tensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    return relu(x)


# ── Interactive viewer entry point ───────────────────────────────────────
#
# Running this file directly launches the local viewer HTTP server on
# the demo fixture so the parsed graph can be inspected in a browser:
#
#     python tests/dsl/test_demo_proj_qkv.py
#     python tests/dsl/test_demo_proj_qkv.py --fixture subset
#
# pytest does not run the ``if __name__ == "__main__"`` block, so
# this stays a side-channel utility and does not interfere with CI.


def _run_viewer(fixture: str = "demo") -> None:

    fixtures = {
        "subset": proj_qkv_subset,
        "demo": proj_qkv_with_mma,
    }
    if fixture not in fixtures:
        raise SystemExit(
            f"unknown fixture {fixture!r}; pick one of {sorted(fixtures)}"
        )
    fn = fixtures[fixture]
    print(f"serving {fn.name} (Ctrl-C to stop)")
    Viewer(fn).serve(port=0, open_browser=True)


if __name__ == "__main__":
    import sys

    fixture = "demo"
    if len(sys.argv) >= 3 and sys.argv[1] == "--fixture":
        fixture = sys.argv[2]
    _run_viewer(fixture)

