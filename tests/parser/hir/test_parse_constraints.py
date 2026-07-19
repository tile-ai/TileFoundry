from __future__ import annotations

import pytest

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Tuple, VerifyError
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import Broadcast, Partial, Split
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.schedule.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    constraint_metadata,
)


def _source(body: str) -> str:
    return f'''from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

cta_mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))

@func
def candidate(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
{body}
'''


def _walk(expr):
    if isinstance(expr, Call):
        yield expr
        for arg in expr.args:
            yield from _walk(arg)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk(element)


def _constraint_signature(metadata: ScheduleConstraintMetadata):
    result = []
    for item in metadata.constraints:
        if isinstance(item, LayoutConstraint):
            result.append((type(item), item.layout, item.bindings))
        elif isinstance(item, MeshConstraint):
            result.append((type(item), item.mesh))
        elif isinstance(item, StorageConstraint):
            result.append((type(item), item.storage))
    return tuple(result)


def test_layout_mesh_storage_constraints_parse_verify_and_round_trip() -> None:
    fn = parse_script(
        _source(
            '''    y: where(layout=(_, 16 @ cta), mesh=cta_mesh, storage="gmem") = tf.add(x, x)
    return y'''
        )
    )
    verify_function(fn)

    metadata = constraint_metadata(fn.body)
    assert isinstance(metadata, ScheduleConstraintMetadata)
    assert len(metadata.constraints) == 3
    layout, mesh, storage = metadata.constraints
    assert isinstance(layout, LayoutConstraint)
    assert layout.layout.shape[0] is not None
    assert repr(layout.layout.shape[0]) == "_"
    assert layout.layout.shape[1] == 16
    assert layout.bindings == (("cta", Split(1)),)
    assert isinstance(mesh, MeshConstraint)
    assert isinstance(storage, StorageConstraint)

    printed = as_script(fn)
    assert 'where(layout=(_, 16 @ cta), mesh=Mesh(' in printed
    assert 'storage="gmem"' in printed
    reparsed = parse_script(printed)
    verify_function(reparsed)
    assert _constraint_signature(constraint_metadata(reparsed.body)) == _constraint_signature(
        metadata
    )


def test_broadcast_and_partial_bindings_reuse_existing_shard_attrs() -> None:
    broadcast = parse_script(
        _source(
            '''    y: where(layout=((_, 16), {cta @ B()})) = tf.add(x, x)
    return y'''
        )
    )
    partial = parse_script(
        _source(
            '''    y: where(layout=((_, 16), {cta @ P("sum")})) = tf.add(x, x)
    return y'''
        )
    )
    broadcast_layout = constraint_metadata(broadcast.body).constraints[0]
    partial_layout = constraint_metadata(partial.body).constraints[0]
    assert isinstance(broadcast_layout, LayoutConstraint)
    assert isinstance(partial_layout, LayoutConstraint)
    assert broadcast_layout.bindings == (("cta", Broadcast()),)
    assert partial_layout.bindings == (("cta", Partial("sum")),)
    assert 'layout=((_, 16), {cta @ B()})' in as_script(broadcast)
    assert 'layout=((_, 16), {cta @ P("sum")})' in as_script(partial)


def test_parameter_and_bound_tuple_get_item_are_valid_subjects() -> None:
    parameter = parse_script(
        _source(
            '''    x: where(storage="smem")
    return x'''
        )
    )
    assert isinstance(constraint_metadata(parameter.params[0]), ScheduleConstraintMetadata)

    tuple_source = '''from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

@func
def tuple_value(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 4), "i64"]:
    values = tf.topk(x, k=4, axis=-1)
    ids = values[1]
    ids: where(storage="gmem")
    return ids
'''
    tuple_fn = parse_script(tuple_source)
    verify_function(tuple_fn)
    assert isinstance(tuple_fn.body, Call)
    assert isinstance(tuple_fn.body.type, TensorType)
    assert any(
        isinstance(expr, Call)
        and expr.target.__class__.__name__ == "TupleGetItem"
        and constraint_metadata(expr) is not None
        for expr in _walk(tuple_fn.body)
    )


@pytest.mark.parametrize(
    "body",
    [
        "    y: where() = tf.add(x, x)\n    return y",
        "    y: where(layout=()) = tf.add(x, x)\n    return y",
        "    y: where(layout=(_, {cta @ B(), cta @ P(\"sum\")})) = tf.add(x, x)\n    return y",
        "    y: where(layout=(D, D)) = tf.add(x, x)\n    return y",
        "    y: where(partial=P(\"sum\")) = tf.add(x, x)\n    return y",
        "    y: where(foo=1) = tf.add(x, x)\n    return y",
        "    y: where(layout=(1.5,)) = tf.add(x, x)\n    return y",
        "    y: where(storage=\"gmem\") = tf.add(x, x)\n    y: where(storage=\"gmem\")\n    return y",
    ],
)
def test_invalid_constraints_fail_at_source_annotation(body: str) -> None:
    with pytest.raises(VerifyError, match="where|layout|duplicate|binding"):
        parse_script(_source(body))


def test_tuple_and_subscript_annotation_subjects_are_rejected() -> None:
    with pytest.raises(VerifyError, match="bound plain Name|annotation lvalue"):
        parse_script(
            _source(
                """    value = tf.add(x, x)
    value[0]: where(storage="gmem")
    return value"""
            )
        )

    tuple_subject = '''from __future__ import annotations
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

@func
def tuple_subject(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
    pair = tf.topk(x, k=4, axis=-1)
    pair: where(storage="gmem")
    return x
'''
    with pytest.raises(VerifyError, match="tensor-valued"):
        parse_script(tuple_subject)
