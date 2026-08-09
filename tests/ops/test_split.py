"""Split result layouts describe each smaller result."""

import math

from tests.ops.typeinfer_utils import infer_call
from tilefoundry.ir.hir.tensor.split import Split
from tilefoundry.ir.types import make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split as SplitAttr
from tilefoundry.ir.types.shard.shard_layout import shard_layout_local_shape


def test_split_rebuilds_plain_and_sharded_result_layouts():
    plain = make_tensor_type(
        (16, 8), layout=Layout(shape=(16, 8), strides=(8, 1))
    )
    plain_parts = infer_call(Split(axis=0, num_splits=4), plain).fields
    assert all(
        part.layout == Layout(shape=(4, 8), strides=(8, 1))
        for part in plain_parts
    )

    sharded = make_shard_tensor_type(
        (16, 8), mesh=make_mesh((4,)), attrs=(SplitAttr(0),)
    )
    sharded_parts = infer_call(Split(axis=1, num_splits=2), sharded).fields
    assert all(isinstance(part.layout, ShardLayout) for part in sharded_parts)
    assert all(part.layout.attrs == (SplitAttr(0),) for part in sharded_parts)
    assert all(
        math.prod(shard_layout_local_shape(part.layout))
        == math.prod(part.shape) // 4
        for part in sharded_parts
    )
