"""``extract`` coverage for ``RMSNorm`` now that it carries a registered
forward ``type_relation`` (``rms_norm.py``'s ``_rms_norm_type_relation``,
modelled on ``SoftMax``'s) instead of a poly-private fallback: the domain is
the batch axes only (``x.shape[:-1]``) and the reduced (last) axis is an
existential range dim on the read/write maps, with ``weight`` read over that
same range. This is the reduction relation every fused row-wise op is built to
the shape of; ``SoftMax``'s own is pinned in ``test_analysis_invariants.py``.

Sharding is resolved one level up, in ``extract``'s ``_local_type``, so the
relation itself never sees a layout; the localization tests below cover that
helper directly.
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.analysis.poly import _local_type
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- rms_norm resolved dynamically
from tilefoundry.ir.types import make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Mesh, Split, Topology

_MESH = Mesh(Topology("gpu", 2), (2,), names=("a",))


@func
def rmsnorm_only(
    x: Tensor[(2, 64), "f32"], weight: Tensor[(64,), "f32"]
) -> Tensor[(2, 64), "f32"]:
    y = rms_norm(x, weight)
    return y


def test_extract_rmsnorm_single_statement():
    """``y = rms_norm(x, weight)`` extracts to one statement: domain =
    the batch axis only, reads/writes both range over the whole
    (existentially-quantified) row -- exactly like ``SoftMax``'s shape."""
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
    # Single statement, nothing else reads/writes x/weight/y: no dependence.
    assert tg.deps.is_empty()


def test_local_type_divides_the_split_axis_and_keeps_tensor_rank():
    """A ``Split`` axis contributes its per-shard extent, and the result keeps
    the tensor's own rank.

    The layout a real sharding path produces (``make_shard_tensor_type`` ->
    ``canonical_shard_layout``) factors the split axis into several layout
    positions, so ``layout.shape`` can outrank the tensor -- ``(8, 16)`` with
    ``Split(0)`` over a 2-way mesh becomes ``(2, 4, 16)``. Localizing through
    the layout would hand a rank-3 access map to a rank-2 buffer, which puts
    reader and writer in different isl spaces and drops the dependence between
    them; ``split_target_axes`` names the *tensor* axis each mesh axis splits
    instead, which is what keeps the rank.

    A non-leading split axis and an unsharded type are asserted beside it because
    the helper has to answer all three: the axis the mesh names is divided, every
    other axis is left whole, and a type with no layout at all comes back as the
    very same object rather than a rebuilt copy of it.
    """
    x = make_shard_tensor_type((8, 16), mesh=_MESH, attrs=(Split(0),))
    assert len(x.layout.layout.shape) == 3  # the factored layout, not a typo

    local = _local_type(x)

    assert local.shape == (4, 16)
    assert len(local.shape) == len(x.shape)

    trailing = make_shard_tensor_type((8, 16), mesh=_MESH, attrs=(Split(1),))
    assert _local_type(trailing).shape == (8, 8)

    plain = make_tensor_type((8, 16))
    assert _local_type(plain) is plain
