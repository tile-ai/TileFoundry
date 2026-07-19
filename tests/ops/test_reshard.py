"""``Reshard`` typeinfer: layout / storage / sharding on one op.

- ``layout`` and ``storage`` are optional; omitting either preserves the
  input's value, and a storage change requires an explicit ``layout=``.
- Logical-shape changes follow the new ``ShardLayout``'s shape.
- Stride materialization is direction-based: low->high (e.g. reg->gmem) uses a
  shared C-order over the canonical shape; high->low (gmem->reg) uses the
  per-instance form (Split axes -> 0, others C-order, size-1 -> 0); same storage
  matches the form already on ``src.layout``; explicit (verbose) strides are
  preserved verbatim in every direction.
"""

from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import make_tensor_type
from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.ir.types.storage import StorageKind


def _shard_layout(shape) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=tuple([1] * len(shape))),
        attrs=(),
        mesh=make_mesh((128,), topology=Topology("cta", 128)),
    )


# Layouts whose identity passes through unchanged.
_SL_PRESERVES_SHAPE = _shard_layout((1, 8, 192))
_SL_LAYOUT_ONLY = _shard_layout((1, 1536))

_MESH_FRAG = Mesh(
    topology=Topology("thread", 32),
    layout=Layout(shape=(4, 8), strides=(1, 4)),
    names=("x", "y"),
)
_FRAG_LAYOUT = ShardLayout(
    layout=Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64)),
    attrs=(Split(axis=1), Split(axis=2)),
    mesh=_MESH_FRAG,
)
_FRAG_VERBOSE = ShardLayout(
    layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)),
    attrs=(Split(1), Split(3)),
    mesh=_MESH_FRAG,
)

# Direction-of-materialization meshes / sugar (strides=None) inputs.
_MESH_H2L = Mesh(topology=Topology("thread", 4), layout=Layout(shape=(4,), strides=(1,)), names=("t",))
_SL_H2L = ShardLayout(layout=Layout(shape=(2, 4, 128), strides=None), attrs=(Split(1),), mesh=_MESH_H2L)

_MESH_L2H = Mesh(topology=Topology("thread", 4), layout=Layout(shape=(4,), strides=(1,)), names=("w",))
_REG_L2H_LAYOUT = ShardLayout(layout=Layout(shape=(4, 64), strides=(0, 1)), attrs=(Split(0),), mesh=_MESH_L2H)
_SL_L2H = ShardLayout(layout=Layout(shape=(4, 64), strides=None), attrs=(Split(0),), mesh=_MESH_L2H)

_MESH_SAME_PLAIN = Mesh(topology=Topology("cta", 8), layout=Layout(shape=(8,), strides=(1,)), names=("c",))
_SL_SAME_PLAIN = ShardLayout(layout=Layout(shape=(8, 64), strides=None), attrs=(Split(0),), mesh=_MESH_SAME_PLAIN)

_MESH_SAME_PI = Mesh(topology=Topology("thread", 4), layout=Layout(shape=(4,), strides=(1,)), names=("t",))
_SRC_SAME_PI_LAYOUT = ShardLayout(layout=Layout(shape=(4, 16), strides=(0, 1)), attrs=(Split(0),), mesh=_MESH_SAME_PI)
_DST_SAME_PI_SUGAR = ShardLayout(layout=Layout(shape=(4, 16), strides=None), attrs=(Split(0),), mesh=_MESH_SAME_PI)


def _materialized(shape, strides, attrs, mesh):
    return ShardLayout(layout=Layout(shape=shape, strides=strides), attrs=attrs, mesh=mesh)


# Dynamic (DimVar) non-split bare axis: admissible only in the shared-engine
# materialization form; the per-instance (high->low) form rejects it because a
# per-shard register/shared buffer cannot be sized by a non-split dynamic axis.
_S_DYN = DimVar("seq_len", 1, 4)
_MESH_DYN = Mesh(topology=Topology("cta", 8), layout=Layout(shape=(8,), strides=(1,)))
_SL_DYN_BARE = ShardLayout(
    layout=Layout(shape=(1, _S_DYN, 32, 128), strides=None),
    attrs=(Split(axis=2),),
    mesh=_MESH_DYN,
)
# Shared-engine C-order strides: only the stride of the axis above the dynamic
# one becomes a symbolic dim-expr; inner strides stay plain ints.
_DYN_OUTER_STRIDE = simplify_dim(DimMul, (32 * 128, _S_DYN))


CASES = [
    TypeInferCase(
        "preserves_logical_shape",
        Reshard(layout=_SL_PRESERVES_SHAPE, storage=rmem),
        (make_tensor_type((1, 1536)),),
        make_tensor_type((1, 1536), storage=rmem, layout=_SL_PRESERVES_SHAPE),
    ),
    # Reshard targets a concrete residency; an unmaterialized destination is
    # rejected (umat is not a place a value can be resharded to).
    TypeInferCase(
        "destination_umat_rejected",
        Reshard(layout=_SL_PRESERVES_SHAPE, storage=StorageKind.UMAT),
        (make_tensor_type((1, 1536)),),
        ExpectedError(match="unmaterialized"),
    ),
    TypeInferCase(
        "storage_unchanged_layout_none_is_noop",
        Reshard(),
        (make_tensor_type((1, 1536)),),
        make_tensor_type((1, 1536)),
    ),
    TypeInferCase(
        "layout_only_preserves_input_storage",
        Reshard(layout=_SL_LAYOUT_ONLY),
        (make_tensor_type((1, 1536)),),
        make_tensor_type((1, 1536), layout=_SL_LAYOUT_ONLY),
    ),
    TypeInferCase(
        "to_reg_preserves_explicit_non_default_strides",
        Reshard(layout=_FRAG_LAYOUT, storage=rmem),
        (make_tensor_type((16, 8)),),
        make_tensor_type((16, 8), storage=rmem, layout=_FRAG_LAYOUT),
    ),
    TypeInferCase(
        "high_to_low_sugar_materializes_per_instance",
        Reshard(layout=_SL_H2L, storage=rmem),
        (make_tensor_type((2, 4, 128)),),
        make_tensor_type((2, 4, 128), storage=rmem,
             layout=_materialized((2, 4, 128), (128, 0, 1), (Split(1),), _MESH_H2L)),
    ),
    TypeInferCase(
        "low_to_high_sugar_materializes_shared",
        Reshard(layout=_SL_L2H, storage=gmem),
        (make_tensor_type((4, 64), storage=rmem, layout=_REG_L2H_LAYOUT),),
        make_tensor_type((4, 64), storage=gmem,
             layout=_materialized((4, 64), (64, 1), (Split(0),), _MESH_L2H)),
    ),
    TypeInferCase(
        "same_storage_sugar_plain_src_falls_back_to_shared",
        Reshard(layout=_SL_SAME_PLAIN, storage=None),
        (make_tensor_type((8, 64)),),
        make_tensor_type((8, 64), storage=gmem,
             layout=_materialized((8, 64), (64, 1), (Split(0),), _MESH_SAME_PLAIN)),
    ),
    TypeInferCase(
        "same_storage_sugar_matches_src_per_instance_form",
        Reshard(layout=_DST_SAME_PI_SUGAR, storage=None),
        (make_tensor_type((4, 16), storage=rmem, layout=_SRC_SAME_PI_LAYOUT),),
        make_tensor_type((4, 16), storage=rmem,
             layout=_materialized((4, 16), (0, 1), (Split(0),), _MESH_SAME_PI)),
    ),
    TypeInferCase(
        "explicit_fragment_strides_preserved_verbatim",
        Reshard(layout=_FRAG_VERBOSE, storage=rmem),
        (make_tensor_type((16, 8)),),
        make_tensor_type((16, 8), storage=rmem, layout=_FRAG_VERBOSE),
    ),
    TypeInferCase(
        "storage_change_without_layout_errors",
        Reshard(storage=rmem),
        (make_tensor_type((1, 1536)),),
        ExpectedError(match="storage change requires"),
    ),
    # Dynamic non-split bare axis: same-storage plain source -> shared-engine,
    # which materializes a symbolic outer stride and keeps inner strides static.
    TypeInferCase(
        "same_storage_dynamic_bare_axis_shared_engine",
        Reshard(layout=_SL_DYN_BARE, storage=None),
        (make_tensor_type((1, _S_DYN, 32, 128)),),
        make_tensor_type((1, _S_DYN, 32, 128), storage=gmem,
             layout=_materialized((1, _S_DYN, 32, 128),
                                  (_DYN_OUTER_STRIDE, 32 * 128, 128, 1),
                                  (Split(2),), _MESH_DYN)),
    ),
    # ... but the per-instance (high->low) form cannot size a static per-shard
    # buffer with a non-split dynamic axis -> deliberate, tested error.
    TypeInferCase(
        "high_to_low_dynamic_bare_axis_rejected",
        Reshard(layout=_SL_DYN_BARE, storage=rmem),
        (make_tensor_type((1, _S_DYN, 32, 128)),),
        ExpectedError(match="not static after sharding", exc=ValueError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reshard_typeinfer(case):
    run_typeinfer_case(case)


@pytest.mark.parametrize("op", [Reshard()], ids=["identity"])
def test_reshard_evaluate(op):
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    run_eval_case(EvalCase("", op, (x,), x, atol=0.0, rtol=0.0))
