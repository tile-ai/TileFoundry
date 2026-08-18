"""Pure-value whole-slice indexed copy."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.tensor.index_add import _infer_index_write
from tilefoundry.ir.hir.tensor.index_select import _norm_dim
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessMode,
    AccessQuantity,
    AccessRelations,
    BoundaryAccess,
    IndexedAccess,
    OutputStorage,
    StorageLink,
    elements_of,
    moves,
    register_access_relation,
)
from tilefoundry.visitor_registry.relation_build import identity_access


@register_op(name="index_copy")
class IndexCopy(Op):
    """Return ``dst`` with ``src`` slices copied to ``index`` along ``dim``."""

    dst = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="input", pattern=Tensor)
    src = ParamDef(kind="input", pattern=Tensor)
    dim = ParamDef(kind="attribute", annotation=int, default=0)


@register_typeinfer(IndexCopy)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return _infer_index_write(
        call,
        ctx,
        op_name="IndexCopy",
        index_dtypes=(DType.i64,),
    )


@register_eval(IndexCopy)
def _eval_index_copy(ctx):
    dst, index, src = (arg.data for arg in ctx.args)
    dim = _norm_dim(ctx.op.dim, dst.dim())
    return TensorValue(
        data=dst.clone().index_copy_(dim, index, src),
        type=ctx.result_type,
    )


__all__ = ["IndexCopy"]


@register_access_relation(IndexCopy)
def _index_copy_access(call: "Call", ctx) -> AccessRelations:
    """The rows the index names are replaced; the container around them is kept.

    Two questions, and the index answers only one. Which rows are written is
    chosen by its values, so that side is a lookup. Where the container lives is
    not: the whole destination is preserved through one affine identity link,
    because these are the same bytes whichever rows get overwritten.

    The payload is affine identity over its own occurrence domain: its
    coordinate is `i` where the destination's is `index[i]`, which is why a
    sharded version needs two mappings on one boundary and is refused earlier.
    """
    dst = ctx.local_type_of(call.args[0])
    index = ctx.local_type_of(call.args[1])
    src = ctx.local_type_of(call.args[2])
    rank = len(dst.shape)
    dim = call.target.dim + rank if call.target.dim < 0 else call.target.dim
    held = elements_of(dst)
    touched = elements_of(src)
    identity = identity_access(rank)
    preserve = StorageLink(
        kind="preserve",
        input=0,
        source=identity,
        output=identity,
        quantity=AccessQuantity(held, held),
    )
    return AccessRelations(
        inputs=(
            BoundaryAccess(
                identity,
                AccessQuantity(held, held),
                AccessMode.TRANSFER,
            ),
            moves(identity_access(1), elements_of(index)),
            moves(identity, touched),
        ),
        outputs=(
            BoundaryAccess(
                IndexedAccess(index_operand=1, axis=dim),
                AccessQuantity(touched, touched),
                AccessMode.WRITE,
                OutputStorage((preserve,)),
            ),
        ),
    )
