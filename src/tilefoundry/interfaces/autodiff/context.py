"""What one reverse-mode rule may read, and how it builds a new call."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.expr import op_registry

__all__ = ["Context", "build", "gradient_layout", "scalar"]


def gradient_layout(layout):
    """The layout a gradient takes for a value of *layout*: each ``Partial`` becomes ``Broadcast``.

    The gradient of a value is laid out like the value, except that a gradient is
    never left partial: DTensor's ``_normalize_placements_for_grad`` is the same rule.
    """
    shard = types.shard_layout_of(layout)
    if shard is None or not any(isinstance(attr, types.Partial) for attr in shard.attrs):
        return layout
    attrs = tuple(types.Broadcast() if isinstance(attr, types.Partial) else attr for attr in shard.attrs)
    return utils.canonical_layout(replace(shard, attrs=attrs))


def build(name: str, *args: expr.Expr, **attrs) -> expr.Call:
    """``tf.<name>(*args, **attrs)`` through the op registry, not typed yet."""
    schema = op_registry.lookup("tf", (), name)
    if not isinstance(schema, expr.OpSchema):
        raise ValueError(f"autodiff: no tf op named {name!r}")
    return expr.Call(type=types.TupleType(()), target=schema.builder(**attrs), args=args)


def scalar(value: int | float, dtype: expr.DType) -> expr.Constant:
    """A scalar literal of *dtype*, not typed in any context."""
    return expr.Constant(type=types.TensorType.from_scalar(dtype, storage=expr.StorageKind.UMAT), value=value)


@dataclass
class Context(typeinfer.TypeInferContext):
    """Type inference over the forward, plus construction of typed calls.

    A rule names the Ops it builds by their DSL name, so one Op file never
    imports another: the rule of Binary builds a reduce, and the rule of
    Reduce builds a binary.
    """

    def call(self, name: str, *args: expr.Expr, **attrs) -> expr.Call:
        """Build ``tf.<name>(*args, **attrs)`` through the op registry and infer its type.

        Only the new call is typed: every argument must already have a type in
        this context, so the forward the arguments come from is never retyped.
        """
        call = build(name, *args, **attrs)
        for arg in args:
            if id(arg) not in self.memo:
                raise ValueError(f"autodiff: tf.{name} got an argument with no type in this context")
        call.retype(typeinfer.TypeInferVisitor(owns_body=False).visit(call, self))
        return call

    def constant(self, value: int | float, dtype: expr.DType | None = None) -> expr.Constant:
        """A scalar literal typed in this context.

        Its dtype is *dtype*, else ``i64`` for an int and ``f32`` for a float.
        """
        if dtype is None:
            dtype = expr.DType.i64 if isinstance(value, int) else expr.DType.f32
        constant = scalar(value, dtype)
        return self.remember(constant, constant.type)

    def remember(self, node: expr.Expr, type_: types.Type) -> expr.Expr:
        """Record *type_* for *node*, a value built outside ``call``, and return *node*."""
        self.memo[id(node)] = (node, type_)
        return node

    def sum_to(self, gradient: expr.Expr, target: types.TensorType) -> expr.Expr:
        """Sum *gradient* over the axes a broadcast added, then reshape it to *target*'s shape."""
        shape = self.type_of(gradient).shape
        lead = len(shape) - len(target.shape)
        axes = (
            *range(lead),
            *(lead + axis for axis, extent in enumerate(target.shape) if extent == 1 and shape[lead + axis] != 1),
        )
        if axes:
            gradient = self.call("reduce", gradient, axes=axes, keepdim=True, kind=expr.ReduceKind.SUM)
        if self.type_of(gradient).shape != tuple(target.shape):
            gradient = self.call("reshape", gradient, new_shape=tuple(target.shape))
        return gradient
