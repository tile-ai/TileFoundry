"""Persistent GEMM schedules cover the output once and remain analyzable."""

from __future__ import annotations

import isl
import pytest
import torch

from tests.fixtures.placed import persistent_gemm_flat as flat
from tests.fixtures.placed import persistent_gemm_tiled as tiled
from tilefoundry.analysis import analyze
from tilefoundry.analysis.scope import (
    Access,
    AccessPrecision,
    Scope,
    build_scopes,
    walk_scopes,
)
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.utils.isl_utils import cardinality

_FAMILIES = ("compute-cost", "memory", "roofline", "performance")


def _loop_scopes(module) -> dict[str, Scope]:
    root = build_scopes(module, module.entry_function())
    return {
        scope.owner.induction_var.name: scope
        for scope in walk_scopes(root)
        if isinstance(scope.owner, LoopRegion)
    }


def _store(scope: Scope) -> tuple[object, Access]:
    for call, accesses in scope.outputs["narrow"].values():
        if isinstance(call.target, InsertSlice):
            assert len(accesses) == 1
            return call, accesses[0]
    raise AssertionError("loop has no InsertSlice output")


def _free_domain_count(scope: Scope) -> int | None:
    domain = scope.domain
    if count := domain.dim(isl.dim_type.PARAM):
        domain = domain.project_out(isl.dim_type.PARAM, 0, count)
    return cardinality(domain)


def _at_unit(image: isl.set, coordinates: tuple[int, ...]) -> isl.set:
    assert image.dim(isl.dim_type.PARAM) == len(coordinates)
    for axis, coordinate in enumerate(coordinates):
        image = image.fix_si(isl.dim_type.PARAM, axis, coordinate)
    return image


@pytest.mark.parametrize(
    "module",
    (tiled.PersistentGemmTiled, flat.PersistentGemmFlat),
    ids=("tiled", "flat"),
)
def test_persistent_gemm_answers_every_analysis_family(module) -> None:
    result = analyze(module, module.entry_function(), analysis=_FAMILIES, level="cta")

    assert set(result.executed) == set(_FAMILIES)


def test_tiled_schedule_is_exact_and_partitions_units() -> None:
    scopes = _loop_scopes(tiled.PersistentGemmTiled)
    assert scopes["mi"].trips() == tiled.CHUNK_M // tiled.BM
    assert scopes["ni"].trips() == tiled.CHUNK_N // tiled.BN
    assert scopes["ki"].trips() == tiled.K // tiled.BK
    assert _free_domain_count(scopes["mi"]) == tiled.M // tiled.BM
    assert _free_domain_count(scopes["ni"]) == (tiled.M // tiled.BM) * (
        tiled.N // tiled.BN
    )

    store, access = _store(scopes["ni"])
    assert tuple(store.args[1].type.shape) == (tiled.BM, tiled.BN)
    assert access.precision is AccessPrecision.EXACT
    written = access.relation.range()
    assert _at_unit(written, (0, 0)).is_disjoint(_at_unit(written, (1, 0)))


def test_flat_schedule_records_its_quasiaffine_limit() -> None:
    """Floor-div/mod offsets widen because affine.py cannot bind those operands."""
    scopes = _loop_scopes(flat.PersistentGemmFlat)
    assert scopes["t"].trips() == flat.NUM_TILES // flat.NBLOCKS
    assert scopes["ki"].trips() == flat.K // flat.BK
    assert _free_domain_count(scopes["t"]) == flat.NUM_TILES

    store, access = _store(scopes["t"])
    assert tuple(store.args[1].type.shape) == (flat.BM, flat.BN)
    assert access.precision is AccessPrecision.WIDENED
    written = access.relation.range()
    assert _at_unit(written, (0,)).is_disjoint(_at_unit(written, (1,)))


def _reference_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(172)
    return (
        torch.randn(tiled.M, tiled.K, dtype=torch.bfloat16),
        torch.randn(tiled.K, tiled.N, dtype=torch.bfloat16),
    )


def _tiled_schedule(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result = torch.empty(tiled.M, tiled.N, dtype=torch.float32)
    for x in range(tiled.BX):
        for y in range(tiled.BY):
            for mi in range(x * tiled.CHUNK_M, (x + 1) * tiled.CHUNK_M, tiled.BM):
                for ni in range(y * tiled.CHUNK_N, (y + 1) * tiled.CHUNK_N, tiled.BN):
                    acc = torch.zeros(tiled.BM, tiled.BN, dtype=torch.float32)
                    for ki in range(0, tiled.K, tiled.BK):
                        acc += a[mi : mi + tiled.BM, ki : ki + tiled.BK].float() @ b[
                            ki : ki + tiled.BK, ni : ni + tiled.BN
                        ].float()
                    result[mi : mi + tiled.BM, ni : ni + tiled.BN] = acc
    return result


def _flat_schedule(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result = torch.empty(flat.M, flat.N, dtype=torch.float32)
    for unit in range(flat.NBLOCKS):
        for tile_index in range(unit, flat.NUM_TILES, flat.NBLOCKS):
            mi = (tile_index // flat.GRID_N) * flat.BM
            ni = (tile_index % flat.GRID_N) * flat.BN
            acc = torch.zeros(flat.BM, flat.BN, dtype=torch.float32)
            for ki in range(0, flat.K, flat.BK):
                acc += a[mi : mi + flat.BM, ki : ki + flat.BK].float() @ b[
                    ki : ki + flat.BK, ni : ni + flat.BN
                ].float()
            result[mi : mi + flat.BM, ni : ni + flat.BN] = acc
    return result


def test_persistent_schedules_match_nonpersistent_gemm() -> None:
    a, b = _reference_inputs()
    expected = a.float() @ b.float()

    torch.testing.assert_close(_tiled_schedule(a, b), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_flat_schedule(a, b), expected, rtol=1e-5, atol=1e-5)
