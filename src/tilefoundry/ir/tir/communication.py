"""Host enqueue, grouped issue, and stream/event completion primitives."""

from types import MappingProxyType

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types, utils

__all__ = [
    "CommEnqueue", "CommGroupStart", "CommGroupEnd", "EventRecord", "EventSynchronize",
    "StreamWaitEvent", "StreamSynchronize", "ExportEvent", "check_enqueue", "handle_type",
    "HANDLES", "verify_handles",
    "CommunicationStatement",
]


class CommunicationStatement(expr.Op):
    """A host resource effect verified without projecting device instruction tiles."""


@expr.register_op(dialect="T", category="communication")
class CommEnqueue(CommunicationStatement):
    """Enqueue an existing semantic transfer into explicitly owned buffers.

    See [tir §2.4](docs/spec/tir.md#24-host-communication).
    """

    src = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor(), effect=expr.MemoryEffect.READ)
    dst = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor(), effect=expr.MemoryEffect.WRITE)
    communicator = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)
    stream = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)
    count_matrix = expr.ParamDef(kind="input", optional=True, default=None, effect=expr.MemoryEffect.READ)
    operation = expr.ParamDef(kind="attribute", annotation=expr.CommunicationOp)


@expr.register_op(dialect="T", category="communication")
class CommGroupStart(CommunicationStatement):
    """Open a host issue group; nested groups close at the outermost end."""


@expr.register_op(dialect="T", category="communication")
class CommGroupEnd(CommunicationStatement):
    """Enqueue the outermost group without claiming device completion."""


@expr.register_op(dialect="T", category="communication")
class EventRecord(CommunicationStatement):
    """Record the current stream frontier in an event generation."""

    event = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)
    stream = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)


@expr.register_op(dialect="T", category="communication")
class StreamWaitEvent(CommunicationStatement):
    """Order future stream work after the recorded event without blocking the host."""

    stream = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)
    event = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)


@expr.register_op(dialect="T", category="communication")
class StreamSynchronize(CommunicationStatement):
    """Establish host completion of the stream's current work."""

    stream = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)


@expr.register_op(dialect="T", category="communication")
class EventSynchronize(CommunicationStatement):
    """Establish host completion of one captured event generation."""

    event = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)


@expr.register_op(dialect="T", category="communication")
class ExportEvent(CommunicationStatement):
    """Export a recorded completion frontier and its retained buffers to the caller."""

    event = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ)


def handle_type(value, kind):
    """Require the declared resource kind without interpreting native pointer values."""
    held = value.type
    if not isinstance(held, types.CommunicationHandleType) or held.kind != kind:
        raise expr.VerifyError(f"expected a {kind} communication handle")
    return held


def check_enqueue(op, args):
    """Reuse the semantic primitive's inference for buffer shapes and rank domains."""
    if not isinstance(op.operation, expr.CommunicationOp) or len(args) not in {4, 5}:
        raise expr.VerifyError("enqueue requires a communication operation and four or five operands")
    communicator = handle_type(args[2], "communicator")
    handle_type(args[3], "stream")
    semantic = op.operation
    if communicator.participants != semantic.participants or communicator.name != semantic.context:
        raise expr.VerifyError("communicator identity or ordered participant mesh differs from the operation")
    if not all(isinstance(arg.type, types.TensorType) for arg in (args[0], args[1], *args[4:])):
        raise expr.VerifyError("communication buffers and count metadata must be tensors")
    call = expr.Call(semantic, (args[0], *args[4:]), type=args[0].type)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    expected, destination = call.type, args[1].type
    if (expected.shape, expected.dtype) != (destination.shape, destination.dtype):
        raise expr.VerifyError("receive buffer shape or dtype differs from the communication result")
    return call


@verify.register(CommEnqueue)
def verify_enqueue(call, ctx):
    check_enqueue(call.target, call.args)


HANDLES = MappingProxyType({
    EventRecord: ("event", "stream"), StreamWaitEvent: ("stream", "event"),
    StreamSynchronize: ("stream",), EventSynchronize: ("event",), ExportEvent: ("event",),
    CommGroupStart: (), CommGroupEnd: (),
})


def verify_handles(call, ctx):
    expected = HANDLES[type(call.target)]
    if len(call.args) != len(expected):
        raise expr.VerifyError("wrong number of communication resource operands")
    for argument, kind in zip(call.args, expected):
        handle_type(argument, kind)


def infer_statement(call, ctx):
    return types.UnitType()


def communication_access(call, ctx):
    return access.iterating((), tuple(access.state(leaf) for arg in call.args for leaf in utils.leaves_of(ctx.type_of(arg))))


for operation in (CommEnqueue, *HANDLES):
    typeinfer.register(operation)(infer_statement)
    access.register_relation(operation)(communication_access)
    if operation is not CommEnqueue:
        verify.register(operation)(verify_handles)
