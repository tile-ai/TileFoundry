"""Parser rules for runtime matmul and placement uses of ``@``."""

from __future__ import annotations

from tests._source import import_dsl
from tilefoundry.ir.constraints import LayoutConstraint, ScheduleConstraintMetadata
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types.shard import Split

_HEADER = """from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf

"""


def _function(expression: str) -> Function:
    function = import_dsl(
        _HEADER
        + f"""@func
def matmul(x: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
    return {expression}
"""
    )
    assert isinstance(function, Function)
    return function


def test_runtime_matmul_matches_the_tf_op() -> None:
    infix = _function("x @ b")
    explicit = _function("tf.matmul(x, b)")

    assert isinstance(infix.body, Call)
    assert isinstance(explicit.body, Call)
    assert isinstance(infix.body.target, MatMul)
    assert isinstance(explicit.body.target, MatMul)
    assert infix.body.type == explicit.body.type
    assert tuple(arg.type for arg in infix.body.args) == tuple(
        arg.type for arg in explicit.body.args
    )


def test_placement_matmul_sugar_remains_a_layout() -> None:
    authored_module = import_dsl(
        _HEADER
        + """from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

@module(
    entry="placed_matmul",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class PlacedMatMul:
    @func
    def placed_matmul(x: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
        with Mesh(("cta",), (2,), ("tile",)) as mesh:
            x_local = tf.reshard(x, (2 @ mesh.tile, 4), "rmem")
            b_local = tf.reshard(b, (4, 3), "rmem")
            return x_local @ b_local
"""
    )
    function = authored_module.entry_function()

    assert isinstance(function.body, Call)
    assert isinstance(function.body.target, MatMul)
    x_local = function.body.args[0]
    assert isinstance(x_local, Call)
    assert isinstance(x_local.target, Reshard)
    assert x_local.target.layout.layout.shape == (2, 4)
    assert x_local.target.layout.attrs == (Split(axis=0),)
    assert x_local.target.layout.mesh.names == ("tile",)


def test_annotation_matmul_sugar_remains_a_constraint_binding() -> None:
    authored_module = import_dsl(
        _HEADER
        + """from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CudaTarget

N = 8
_MESH = Mesh((Topology("m.a", 8),), Layout((8,), (1,)))

@module(
    entry="constrained",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("m.a", 8),),
)
class Constraint:
    @func
    def constrained(x: Tensor[(N, 16), "bf16"]):
        y: where(layout=(N @ m.a, 16), mesh=_MESH, storage="gmem") = tf.add(x, x)
        return y
"""
    )
    function = authored_module.entry_function()

    metadata = next(
        item for item in function.body.metadata if isinstance(item, ScheduleConstraintMetadata)
    )
    layout = next(
        constraint
        for constraint in metadata.constraints
        if isinstance(constraint, LayoutConstraint)
    )
    assert layout.layout.shape == (8, 16)
    assert layout.bindings == (("m.a", Split(axis=0)),)
