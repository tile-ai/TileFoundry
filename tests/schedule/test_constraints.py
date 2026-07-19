from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Tuple, VerifyError
from tilefoundry.ir.types import TensorType
from tilefoundry.parser import parse_func_source
from tilefoundry.schedule.constraints import (
    WILDCARD,
    LayoutConstraint,
    LayoutDimKind,
    MeshConstraint,
    PartialConstraint,
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


def _constraint_signature(metadata: ScheduleConstraintMetadata):
    result = []
    for item in metadata.constraints:
        if isinstance(item, LayoutConstraint):
            result.append(
                (
                    type(item),
                    tuple((dim.index, dim.kind, dim.extent, dim.topology) for dim in item.dims),
                    tuple(item.attrs),
                )
            )
        elif isinstance(item, MeshConstraint):
            result.append((type(item), item.mesh))
        elif isinstance(item, StorageConstraint):
            result.append((type(item), item.storage))
        elif isinstance(item, PartialConstraint):
            result.append((type(item), item.reduction, item.topology))
    return tuple(result)


def test_constraints_round_trip_as_one_keyword_only_where_annotation() -> None:
    fn = parse_func_source(
        _source(
            """    y: where(layout=(_, 16 @ cta), mesh=cta_mesh, storage="gmem", partial=P("sum")) = tf.add(x, x)
    return y"""
        )
    )
    metadata = constraint_metadata(fn.body)
    assert isinstance(metadata, ScheduleConstraintMetadata)
    assert len(metadata.constraints) == 4
    layout, mesh, storage, partial = metadata.constraints
    assert isinstance(layout, LayoutConstraint)
    assert layout.physical_shape[0] is WILDCARD
    assert layout.dims[1].kind is LayoutDimKind.SPLIT
    assert isinstance(mesh, MeshConstraint)
    assert isinstance(storage, StorageConstraint)
    assert isinstance(partial, PartialConstraint)

    printed = as_script(fn)
    assert 'where(layout=(_, 16 @ cta), mesh=Mesh(' in printed
    assert 'storage="gmem"' in printed
    assert 'partial=P("sum")' in printed
    reparsed = parse_func_source(printed)
    assert _constraint_signature(constraint_metadata(reparsed.body)) == _constraint_signature(
        metadata
    )


def test_parameter_and_bound_tuple_get_item_are_valid_subjects() -> None:
    parameter = parse_func_source(
        _source(
            """    x: where(storage="smem")
    return x"""
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
    tuple_fn = parse_func_source(tuple_source)
    assert isinstance(tuple_fn.body, Call)
    assert isinstance(tuple_fn.body.type, TensorType)
    assert any(
        isinstance(expr, Call)
        and expr.target.__class__.__name__ == "TupleGetItem"
        and constraint_metadata(expr) is not None
        for expr in _walk(tuple_fn.body)
    )


def test_metadata_is_excluded_from_expr_equality_and_hashing() -> None:
    fn = parse_func_source(_source("    return tf.add(x, x)"))
    tagged = replace(
        fn.body,
        metadata=(
            ScheduleConstraintMetadata(
                constraints=(StorageConstraint(storage="gmem"),),
            ),
        ),
    )
    assert tagged == fn.body
    assert hash(tagged) == hash(fn.body)


@pytest.mark.parametrize(
    "body",
    [
        "    y: where() = tf.add(x, x)\n    return y",
        "    y: where(layout=()) = tf.add(x, x)\n    return y",
        "    y: where(layout=(1 @ cta, 2 @ cta)) = tf.add(x, x)\n    return y",
        "    y: where(foo=1) = tf.add(x, x)\n    return y",
        "    y: where(layout=(1.5,)) = tf.add(x, x)\n    return y",
        "    y: where(layout=(1,)) = tf.add(x, x)\n    y: where(storage=\"gmem\")\n    return y",
    ],
)
def test_invalid_or_repeated_constraints_fail_at_source_annotation(body: str) -> None:
    with pytest.raises(VerifyError, match="where|layout|duplicate"):
        parse_func_source(_source(body))


def test_tuple_and_subscript_annotation_subjects_are_rejected() -> None:
    with pytest.raises(VerifyError, match="bound plain Name|annotation lvalue"):
        parse_func_source(
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
        parse_func_source(tuple_subject)


def _walk(expr):
    if isinstance(expr, Call):
        yield expr
        for arg in expr.args:
            yield from _walk(arg)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk(element)
