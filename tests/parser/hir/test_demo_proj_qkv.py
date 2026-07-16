"""Parser assertions for the reusable ``proj_qkv`` model definitions."""

from __future__ import annotations

from tests.models.demo.proj_qkv import proj_qkv_subset, proj_qkv_with_mma
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.inspection.viewer import Viewer
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16
from tilefoundry.ir.hir.grid_region import GridRegionExpr


def _walk_calls(expr, *, into_grid: bool = True):
    """Yield every ``Call`` reachable from *expr*."""
    if isinstance(expr, Call):
        yield expr
        for argument in expr.args:
            yield from _walk_calls(argument, into_grid=into_grid)
    if into_grid and isinstance(expr, GridRegionExpr):
        yield from _walk_calls(expr.body, into_grid=into_grid)
        for value in expr.yield_values:
            yield from _walk_calls(value, into_grid=into_grid)


def _walk_grid(expr):
    """Yield every ``GridRegionExpr`` reachable from *expr*."""
    if isinstance(expr, GridRegionExpr):
        yield expr
    if isinstance(expr, Call):
        for argument in expr.args:
            yield from _walk_grid(argument)


def test_proj_qkv_subset_parses() -> None:
    """The reusable subset parses through the standard decorator path."""
    assert proj_qkv_subset.name == "proj_qkv_subset"


def test_proj_qkv_subset_produces_grid_region_with_carry() -> None:
    """The tile loop carries ``o_smem`` out of its body."""
    grids = list(_walk_grid(proj_qkv_subset.body))
    assert any(grid.carried_args for grid in grids), (
        "expected GridRegionExpr with non-empty carried_args; "
        f"got {[len(grid.carried_args) for grid in grids]}"
    )


def test_subset_uses_mma_ssa_and_carry_and_int_at_mesh() -> None:
    """The reusable MMA model exercises SSA, carry-out, and ``@`` sugar."""
    found_mma = [
        call
        for call in _walk_calls(proj_qkv_with_mma.body)
        if isinstance(call.target, Mma_SM80_16x8x16)
    ]
    assert found_mma, "expected at least one Mma_SM80_16x8x16 Call"
    assert len(found_mma[0].args) == 2
    assert found_mma[0].target.dtype_acc.name == "f32"

    grids = list(_walk_grid(proj_qkv_with_mma.body))
    assert grids, "expected GridRegionExpr from tile-for"
    assert any(grid.carried_args for grid in grids)


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
