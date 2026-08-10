"""Pin ``extract`` behavior for the registered ``RMSNorm`` relation.

Its domain contains batch axes only; the reduced last axis is existential in
read/write maps, including the weight read. ``extract._local_type`` resolves
sharding before the relation sees the type. ``test_analysis_invariants.py``
pins the corresponding ``SoftMax`` shape.
"""

from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.analysis.poly import _local_type
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- rms_norm resolved dynamically
from tilefoundry.ir.types import make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, Mesh, Split, Topology

_MESH = Mesh((Topology("gpu", 2),), Layout((2,), (1,)), names=("a",))


@func
def rmsnorm_only(x: Tensor[(2, 64), "f32"], weight: Tensor[(64,), "f32"]) -> Tensor[(2, 64), "f32"]:
    y = rms_norm(x, weight)
    return y


def test_extract_rmsnorm_single_statement():
    """``y = rms_norm(x, weight)`` extracts to one statement.

    ``y = rms_norm(x, weight)`` extracts to one statement: domain =
    the batch axis only, reads/writes both range over the whole
    (existentially-quantified) row -- exactly like ``SoftMax``'s shape.
    """
    tg = extract(rmsnorm_only)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 1
    assert tg.units[0].name == "RN"
    assert type(tg.units[0].op.target).__name__ == "RMSNorm"

    print("\n=== rmsnorm: domain ===")
    print(tg.domain)
    print("=== rmsnorm: reads ===")
    print(tg.reads)
    print("=== rmsnorm: writes ===")
    print(tg.writes)

    assert tg.domain.is_equal(isl.union_set("{ RN[i] : 0 <= i < 2 }"))
    expected_reads = (
        isl.union_map("{}")
        .union(isl.map("{ RN[i] -> x[i, j] : 0 <= i < 2 and 0 <= j < 64 }"))
        .union(isl.map("{ RN[i] -> weight[j] : 0 <= i < 2 and 0 <= j < 64 }"))
    )
    expected_writes = isl.union_map("{ RN[i] -> y[i, j] : 0 <= i < 2 and 0 <= j < 64 }")
    assert tg.reads.is_equal(expected_reads)
    assert tg.writes.is_equal(expected_writes)

    assert tg.deps.is_empty()


def test_local_type_divides_the_split_axis_and_keeps_tensor_rank():
    """A ``Split`` axis contributes its per-shard extent.

    Canonical layouts may outrank tensors: split ``(8, 16)`` becomes layout
    ``(2, 4, 16)``. Localization must use ``split_target_axes`` to divide the
    tensor axis while preserving rank, or reader and writer enter different isl
    spaces and lose dependencies. The test also covers a trailing split and the
    identity behavior for an unsharded type.
    """
    x = make_shard_tensor_type((8, 16), mesh=_MESH, attrs=(Split(0),))
    assert len(x.layout.layout.shape) == 3

    local = _local_type(x)

    assert local.shape == (4, 16)
    assert len(local.shape) == len(x.shape)

    trailing = make_shard_tensor_type((8, 16), mesh=_MESH, attrs=(Split(1),))
    assert _local_type(trailing).shape == (8, 8)

    plain = make_tensor_type((8, 16))
    assert _local_type(plain) is plain
