"""Reverse-mode differentiation: one forward Module becomes a forward and a backward Module.

The forward returns what the original returned and, last, a ``saved`` tuple of
the values the backward reads. The backward takes the gradient of each floating
point result and ``saved``, and returns the gradient of each ``wrt`` parameter.

See [passes §7.4](docs/spec/passes.md#74-differentiate).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from tilefoundry import dispatch, verifier
from tilefoundry.interfaces import access, autodiff, typeinfer
from tilefoundry.ir import core, expr, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.hir import sharding

from . import recompute_selection
from .recompute_selection import is_projection, located, projection, select_value

__all__ = ["Saved", "differentiate", "saved_values", "select_value"]

GENERATED = "autodiff"


@dataclass(frozen=True)
class Saved:
    """One value the forward keeps for the backward.

    *storage* is the value whose bytes a view keeps alive, reached through each
    view's aliased operand; ``None`` when the value is its own storage.
    """

    name: str
    type: types.Type
    origin: str
    storage: Saved | None = None


def differentiate(
    module: core.Module, *, wrt: tuple[str, ...], recompute: tuple[str, ...] = ()
) -> tuple[core.Module, core.Module]:
    """Return ``(<Name>Forward, <Name>Backward)`` for the entry of *module*.

    *wrt* names the entry parameters whose gradients the backward returns, in
    that order. Each item of *recompute* selects what the backward computes
    again instead of reading it from ``saved``: ``Module[.function][:variable]``
    for each element of a definition, ``Root.slot[.index][.function][:variable]``
    for one element of a child slot of the root, ``:None`` for none of it.
    """
    if module.metadata.get(GENERATED) == "backward":
        raise ValueError(
            f"differentiate: {module.name!r} is a backward that differentiate generated; "
            "a second derivative is not supported"
        )
    wrt = tuple(wrt)
    if not wrt:
        raise ValueError(f"differentiate: {module.name!r}: wrt names no parameter")
    selection = recompute_selection.parse_selectors(module, tuple(recompute))
    tree = recompute_selection.expand(module, selection)
    selected = {id(value): value for value in recompute_selection.resolve(tree, selection)}
    return Differentiator(selected, module.target).module(tree, wrt)


def saved_values(forward: core.Module) -> tuple[Saved, ...]:
    """What *forward* keeps for its backward, in the order of the saved tuple."""
    if forward.metadata.get(GENERATED) != "forward":
        raise ValueError(f"saved_values: {forward.name!r} is not a forward that differentiate generated")
    return forward.metadata["saved"]


def is_float(type_: types.Type) -> bool:
    if isinstance(type_, types.TupleType):
        return any(is_float(item) for item in type_.fields)
    return isinstance(type_, types.TensorType) and isinstance(type_.dtype, expr.FloatDType)


def gradient_type(type_: types.Type) -> types.Type:
    """The type of the gradient of a value of *type_*: its layout without ``Partial``."""
    return ir_utils.map_tensor_types(type_, lambda leaf: replace(leaf, layout=autodiff.gradient_layout(leaf.layout)))


def fields_of(type_: types.Type) -> tuple[types.Type, ...]:
    return type_.fields if isinstance(type_, types.TupleType) else (type_,)


def leaves(value, path=()):
    """``(path, Expr)`` for each Expr in a nested gradient tuple."""
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    elif value is not None:
        yield path, value


def flat(type_: types.Type):
    """A tuple type as nested Python tuples of its tensor types."""
    return tuple(flat(item) for item in type_.fields) if isinstance(type_, types.TupleType) else type_


def at(structure, path):
    """The item of a nested gradient tuple at *path*, or ``None``."""
    for index in path:
        if structure is None:
            return None
        structure = structure[index]
    return structure


def placed(structure, path, value):
    """*structure* with *value* at *path*."""
    if not path:
        return value
    items = list(structure)
    items[path[0]] = placed(items[path[0]], path[1:], value)
    return tuple(items)


def empty_like(type_: types.Type):
    return tuple(empty_like(item) for item in type_.fields) if isinstance(type_, types.TupleType) else None


def capture_name(pair: tuple[expr.Var, expr.Expr]) -> str:
    """The name a printed region reads its argument by: what the parser sorts captures by."""
    _, value = pair
    return value.name if isinstance(value, expr.Var) else expr.binding_name(value) or ""


def pack(items: tuple[expr.Expr, ...]) -> expr.Tuple:
    return expr.Tuple(elements=tuple(items), type=types.TupleType(tuple(item.type for item in items)))


def unpack(value: expr.Expr, fields: tuple[types.Type, ...]) -> tuple[expr.Expr, ...]:
    """The projection of each field of *value*."""
    return tuple(projection(value, index, item) for index, item in enumerate(fields))


def repack(value: expr.Expr, type_: types.Type) -> expr.Expr:
    """*value* read as *type_*: its one result, or a tuple of each field's projection."""
    if not isinstance(type_, types.TupleType):
        return projection(value, 0, type_)
    return pack(unpack(value, type_.fields))


def trip_count(loop: core.LoopRegion) -> int:
    """How many iterations *loop* runs; a domain that is not three integers is refused."""
    domain = (loop.start, loop.extent, loop.step)
    if not all(isinstance(bound, int) and not isinstance(bound, bool) for bound in domain):
        raise ValueError(
            f"differentiate: a loop whose trip count is not static is not differentiated{located(loop)}"
        )
    return len(range(*domain))


def stacked(type_: types.Type, trip: int, owners, where_: str) -> types.Type:
    """*type_* with a leading, unsplit axis of *trip* rows; ``Split(a)`` turns ``Split(a + 1)``."""
    if isinstance(type_, types.TupleType):
        return types.TupleType(tuple(stacked(item, trip, owners, where_) for item in type_.fields))
    shape = (trip, *type_.shape)
    shard = types.shard_layout_of(type_.layout)
    if shard is None:
        return types.TensorType(shape, type_.dtype, types.Layout(shape), type_.storage)
    if any(isinstance(attr, types.Partial) for attr in shard.attrs):
        raise ValueError(f"differentiate: a loop body value read by the backward is Partial{where_}")
    axes = types.split_target_axes(shard, type_.shape)
    attrs = tuple(
        types.Split(axis + 1) if axis is not None else attr for attr, axis in zip(shard.attrs, axes)
    )
    layout = types.ShardLayout.from_logical(shape, shard.mesh, attrs, storage=type_.storage, owners=owners)
    return types.TensorType(shape, type_.dtype, layout, type_.storage)


def operands(node: expr.Expr) -> tuple[expr.Expr, ...]:
    """The values *node* reads in its own scope: a region's args, not its body."""
    if isinstance(node, (core.MeshRegion, core.LoopRegion)):
        return node.args
    return core.expr_children(node)


class ScopeOrder(core.ExprCollector):
    """The values of one scope, operands first, marking which are active and which invariant.

    A value is active when it reads an active value. It is invariant when only
    the scope's invariant parameters and constants reach it through Ops: in a
    loop body, the induction value and the values the body reads from outside.
    """

    def __init__(self, scope: "Scope") -> None:
        super().__init__()
        self.scope = scope

    def visit_operands(self, node: expr.Expr, ctx) -> tuple:
        return tuple(self.visit(child, ctx) for child in operands(node))

    def default_visit_leaf(self, node: expr.Expr, results: tuple, ctx: list[expr.Expr]) -> None:
        read = operands(node)
        if isinstance(node, (expr.Call, expr.Tuple, core.MeshRegion, core.LoopRegion)) and any(
            id(value) in self.scope.active for value in read
        ):
            self.scope.active.add(id(node))
        computed = isinstance(node, expr.Tuple) or (
            isinstance(node, expr.Call) and not isinstance(node.target, core.FunctionRef)
        )
        if isinstance(node, (expr.Constant, core.ShapeOf)) or (
            computed and all(id(value) in self.scope.invariant for value in read)
        ):
            self.scope.invariant.add(id(node))
        ctx.append(node)


@dataclass
class Derived:
    """What differentiating one function gives its callers."""

    forward: core.Function
    backward: core.Function
    wrt: tuple[int, ...]
    results: int
    cotangents: tuple[types.Type | None, ...]
    saved: types.TupleType
    gradients: types.Type
    saved_values: tuple[Saved, ...] = ()
    saved_params: tuple[int | None, ...] = ()


@dataclass
class Scope:
    """The values of a function body, or of one MeshRegion body inside it."""

    nodes: dict[int, expr.Expr]
    ctx: autodiff.Context
    parent: "Scope | None" = None
    region: core.MeshRegion | core.LoopRegion | None = None
    index: expr.Var | None = None
    active: set[int] = field(default_factory=set)
    gradients: dict[int, object] = field(default_factory=dict)
    slots: dict[object, expr.Expr] = field(default_factory=dict)
    region_params: list[tuple[expr.Var, expr.Expr]] = field(default_factory=list)
    extras: list[tuple] = field(default_factory=list)
    rebuilder: "Rebuilder | None" = None
    invariant: set[int] = field(default_factory=set)
    order: list[expr.Expr] = field(default_factory=list)

    def rebuilt(self, owner: "FunctionDifferentiator", node: expr.Expr) -> expr.Expr:
        """*node* of the backward with every forward value it reads replaced by its slot.

        The scope's ``Rebuilder`` is made on first use.
        """
        if self.rebuilder is None:
            self.rebuilder = Rebuilder(owner, self)
        return self.rebuilder.visit(node)

    def remember_gradient(self, node: expr.Expr, gradient) -> None:
        self.gradients[id(node)] = gradient

    def remember_slot(self, slot: tuple, value: expr.Expr) -> None:
        self.slots[slot] = value

    def walk(self, roots: tuple[expr.Expr, ...]) -> None:
        """Order the values *roots* reach, marking the active and the invariant ones."""
        self.order = []
        walker = ScopeOrder(self)
        for root in roots:
            walker.visit(root, self.order)
        self.nodes.update((id(node), node) for node in self.order)



def new_scope(roots, params, ctx, active: set[int], parent=None, region=None, index=None, invariant=()) -> Scope:
    scope = Scope(
        nodes={id(param): param for param in params}, ctx=ctx, parent=parent, region=region,
        active=active, index=index, invariant=set(invariant),
    )
    scope.walk(roots)
    return scope


@dataclass
class RegionBackward:
    """The backward of one forward MeshRegion, before its saved values are known."""

    scope: Scope
    node: core.MeshRegion


class Differentiator:
    """Differentiate a Module tree, each Module and function once.

    A child Module without its own target is typed under the root's *target*.
    """

    def __init__(self, recompute: dict[int, expr.Expr], target) -> None:
        self.recompute = recompute
        self.target = target
        self.modules: dict[int, tuple[core.Module, core.Module]] = {}
        self.functions: dict[int, Derived] = {}
        self.owned: dict[int, list[tuple[core.Function, Derived]]] = {}
        self.slots: dict[int, dict[str, tuple]] = {}

    def module(self, module: core.Module, wrt: tuple[str, ...] | None = None) -> tuple[core.Module, core.Module]:
        """The forward and backward of *module*; *wrt* ``None`` takes each trainable parameter."""
        if wrt is None and id(module) in self.modules:
            return self.modules[id(module)]
        entry = module.entry_function()
        if not isinstance(entry, core.Function) or entry.body is None:
            raise ValueError(f"differentiate: the entry of {module.name!r} is not a HIR function with a body")
        self.owned[id(module)] = []
        self.slots[id(module)] = {}
        self.function(module, entry, None if wrt is None else wrt_indices(module, entry, wrt))
        functions = sorted(self.owned[id(module)], key=lambda pair: pair[0] is not entry)
        names = [function.name for function, _ in functions]
        if "backward" in names[1:]:
            raise ValueError(f"differentiate: {module.name!r} declares a function named 'backward'")
        children_forward, children_backward = {}, {}
        for key, pairs in self.slots[id(module)].items():
            forwards, backwards = zip(*pairs)
            indexed = isinstance(module.children[key], tuple)
            children_forward[key] = forwards if indexed else forwards[0]
            children_backward[key] = backwards if indexed else backwards[0]

        forward = core.Module(
            name=f"{module.name}Forward",
            functions=tuple(derived.forward for _, derived in functions),
            entry=module.entry,
            children=children_forward,
            target=module.target,
            topologies=module.topologies,
            metadata={GENERATED: "forward", "saved": functions[0][1].saved_values},
        )
        backward = core.Module(
            name=f"{module.name}Backward",
            functions=tuple(derived.backward for _, derived in functions),
            entry="backward",
            children=children_backward,
            target=module.target,
            topologies=module.topologies,
            metadata={GENERATED: "backward"},
        )
        for generated in (forward, backward):
            for function in generated.functions:
                context = typeinfer.TypeInferContext(
                    module=generated, function=function, target=generated.target or self.target
                )
                typeinfer.infer_function(function, context)
                verifier.verify_function(function, module=generated)
        if wrt is None:
            self.modules[id(module)] = (forward, backward)
        return forward, backward

    def function(self, module: core.Module, function: core.Function, wrt: tuple[int, ...] | None) -> Derived:
        """Differentiate *function* of *module*; *wrt* ``None`` takes each trainable parameter."""
        if wrt is None:
            if id(function) in self.functions:
                return self.functions[id(function)]
            wrt = tuple(
                index for index, param in enumerate(function.params) if not param.is_const and is_float(param.type)
            )
        derived = FunctionDifferentiator(self, module, function, wrt).run()
        self.functions[id(function)] = derived
        self.owned[id(module)].append((function, derived))
        return derived

    def callee(self, module: core.Module, ref: core.FunctionRef, call: expr.Call) -> Derived:
        """Differentiate what *ref* calls from *module*, keeping the slot it lives in."""
        if not ref.path:
            _, function = core.resolve(module, ref)
            return self.function(module, function, None)
        if sum(isinstance(segment, str) for segment in ref.path) != 1 or ref.name is not None:
            raise ValueError(
                f"differentiate: the call of {ref.qualified_name!r} reaches past one child slot; "
                f"only a call of a child slot's entry is differentiated{located(call)}"
            )
        key = ref.path[0]
        slot = core.instances(module.children[key])
        pairs = self.slots[id(module)].get(key)
        if pairs is None:
            pairs = tuple(self.module(child) for child in slot)
            self.slots[id(module)][key] = pairs
        child, _ = core.resolve(module, ref)
        return self.functions[id(child.entry_function())]


def wrt_indices(module: core.Module, function: core.Function, wrt: tuple[str, ...]) -> tuple[int, ...]:
    """The positions of the *wrt* parameters of *function*, each checked to be trainable."""
    indices = []
    for name in wrt:
        matches = [index for index, param in enumerate(function.params) if param.name == name]
        if not matches:
            names = ", ".join(param.name for param in function.params)
            raise ValueError(
                f"differentiate: wrt {name!r} is not a parameter of {module.name}.{function.name} ({names})"
            )
        param = function.params[matches[0]]
        if param.is_const:
            raise ValueError(
                f"differentiate: wrt {name!r} is a ConstTensor, a constant with no gradient; "
                "declare a trained weight as an ordinary parameter"
            )
        if not all(isinstance(leaf.dtype, expr.FloatDType) for leaf in ir_utils.leaves_of(param.type)):
            raise ValueError(
                f"differentiate: wrt {name!r} is not a floating point tensor or a tuple of them"
            )
        indices.append(matches[0])
    return tuple(indices)


class FunctionDifferentiator(expr.Functor[None]):
    """The reverse pass over one function, then the forward and backward built from it.

    The reverse pass visits each value of a scope with ``(scope, gradient)``, after
    every value that reads it: ``reverse`` drives the visits in reverse
    ``ScopeOrder``, which a memoized visitor that recurses into operands cannot give.
    """

    def __init__(self, differ: Differentiator, module: core.Module, function: core.Function, wrt) -> None:
        super().__init__()
        self.differ = differ
        self.module = module
        self.function = function
        self.wrt = wrt
        self.placeholders: dict[int, tuple] = {}
        self.regions: dict[int, RegionBackward] = {}
        self.loops: dict[int, RegionBackward] = {}
        values = core.function_values(function)
        self.callees: dict[int, Derived] = {
            id(value): differ.callee(module, value.target, value)
            for value in values
            if isinstance(value, expr.Call) and isinstance(value.target, core.FunctionRef)
        }
        self.saved_keys: list[tuple] = []
        memo = {id(node): (node, node.type) for node in values}
        ctx = autodiff.Context(module=module, function=function, target=module.target or differ.target, memo=memo)
        active = {id(function.params[index]) for index in wrt}
        self.top = new_scope((function.body,), function.params, ctx, active)
        self.names = {
            name
            for node in values
            for name in (getattr(node, "name", None), expr.binding_name(node))
            if isinstance(name, str)
        }
        for loop in values:
            if isinstance(loop, core.LoopRegion):
                self.names.update(var.name for var in (loop.induction_var, *loop.params))
            elif isinstance(loop, core.MeshRegion):
                self.names.update(var.name for var in loop.params)
        self.saved = expr.Var(name=self.fresh("saved"), type=types.TupleType(()))

    def fresh(self, base: str) -> str:
        """*base*, or *base* with the first numeric suffix no name of this function takes."""
        name, suffix = base, 1
        while name in self.names:
            name, suffix = f"{base}_{suffix}", suffix + 1
        self.names.add(name)
        return name

    def run(self) -> Derived:
        function = self.function
        results = fields_of(function.return_type)
        cotangents = tuple(gradient_type(item) if is_float(item) else None for item in results)
        names = ["dy"] if sum(item is not None for item in cotangents) == 1 else [
            f"dy{index}" for index in range(len(cotangents))
        ]
        params, seeds = [], []
        for index, type_ in enumerate(cotangents):
            if type_ is None:
                seeds.append(empty_like(results[index]))
                continue
            param = expr.Var(name=self.fresh(names[0] if len(names) == 1 else names[index]), type=type_)
            self.top.ctx.remember(param, type_)
            params.append(param)
            seeds.append(self.split(param, self.top.ctx))
        seed = tuple(seeds) if isinstance(function.return_type, types.TupleType) else seeds[0]
        self.reverse(self.top, ((function.body, seed),))
        outputs = tuple(
            self.finish(self.top.gradients.get(id(function.params[index])), function.params[index].type, self.top.ctx)
            for index in self.wrt
        )
        gradients = types.TupleType(tuple(gradient_type(function.params[index].type) for index in self.wrt))
        returned = outputs[0] if len(outputs) == 1 else pack(outputs)
        body = self.top.rebuilt(self, returned)
        meshes = {
            repr(node.mesh): node.mesh for node in self.top.nodes.values() if isinstance(node, core.MeshRegion)
        }
        saved = types.TupleType(tuple(self.key_type(key) for key in self.saved_keys))
        self.saved.restamp(saved, saved)
        if len(meshes) == 1:
            (mesh,) = meshes.values()
            body = enclose(body, mesh, (*params, self.saved))
        backward = core.Function.from_parts(
            name="backward" if function is self.module.entry_function() else function.name,
            params=(*params, self.saved),
            body=body,
            return_type=gradients.fields[0] if len(outputs) == 1 else gradients,
        )
        forward = self.forward()
        return Derived(
            forward=forward,
            backward=backward,
            wrt=self.wrt,
            results=len(results),
            cotangents=cotangents,
            saved=self.saved.type,
            gradients=backward.return_type,
            saved_values=tuple(self.record(index, key) for index, key in enumerate(self.saved_keys)),
            saved_params=tuple(
                next((index for index, param in enumerate(function.params) if key[1] is param), None)
                if key[0] == "node" else None
                for key in self.saved_keys
            ),
        )

    def split(self, value: expr.Expr, ctx: autodiff.Context):
        """A gradient of tuple type as a Python tuple of its fields, at any depth."""
        type_ = ctx.type_of(value)
        if not isinstance(type_, types.TupleType):
            return value
        return tuple(
            self.split(ctx.call("tuple_get_item", value, ctx.constant(index)), ctx)
            for index in range(len(type_.fields))
        )

    def finish(self, gradient, type_: types.Type, ctx: autodiff.Context) -> expr.Expr:
        """*gradient* of a *type_* value, completed with zeros and laid out as ``gradient_type``."""
        if isinstance(type_, types.TupleType):
            parts = gradient if isinstance(gradient, tuple) else empty_like(type_)
            packed = pack(tuple(self.finish(part, item, ctx) for part, item in zip(parts, type_.fields)))
            return ctx.remember(packed, gradient_type(type_))
        if gradient is None:
            return ctx.call("zeros", type=gradient_type(type_))
        return self.laid_out(gradient, type_, ctx)

    def laid_out(self, gradient, type_: types.Type, ctx: autodiff.Context):
        """*gradient* of a *type_* value, resharded to ``gradient_layout``; None stays None.

        Where the value is Broadcast and its gradient Varying, every unit used the
        same value, so the units' gradients sum: a Bitcast states the sum first.
        """
        if gradient is None:
            return None
        if isinstance(type_, types.TupleType):
            return tuple(self.laid_out(part, item, ctx) for part, item in zip(gradient, type_.fields))
        wanted = gradient_type(type_)
        held = types.shard_layout_of(ctx.type_of(gradient).layout)
        if held is not None:
            states = ir_utils.attrs_by_axis(wanted, held.mesh)
            attrs = tuple(
                types.Partial() if attr == types.Varying() and states[key] == types.Broadcast() else attr
                for key, attr in zip(ir_utils.attrs_by_axis(ctx.type_of(gradient)), held.attrs, strict=True)
            )
            if attrs != held.attrs:
                gradient = ctx.call("bitcast", gradient, layout=replace(held, attrs=attrs))
        if ctx.type_of(gradient).layout != wanted.layout:
            gradient = ctx.call(
                "reshard", gradient, layout=types.normalize_layout(wanted.layout), storage=wanted.storage
            )
        return gradient

    def accumulate(self, scope: Scope, node: expr.Expr, gradient) -> None:
        if gradient is None or id(node) not in scope.active or not is_float(node.type):
            return
        held = scope.gradients.get(id(node))
        scope.remember_gradient(node, gradient if held is None else self.add(scope, node.type, held, gradient))

    def add(self, scope: Scope, type_: types.Type, left, right):
        if left is None:
            return right
        if right is None:
            return left
        if isinstance(type_, types.TupleType):
            return tuple(self.add(scope, item, a, b) for item, a, b in zip(type_.fields, left, right))
        ctx = scope.ctx
        if types.shard_layout_of(ctx.type_of(left).layout) != types.shard_layout_of(ctx.type_of(right).layout):
            left, right = (self.finish(value, type_, ctx) for value in (left, right))
        return ctx.call("add", left, right)

    def reverse(self, scope: Scope, seeds) -> None:
        """Walk the values of *scope* backward from *seeds*, ``(value, gradient)`` pairs."""
        for node, seed in seeds:
            self.accumulate(scope, node, seed)
        for node in reversed(scope.order):
            gradient = scope.gradients.get(id(node))
            if gradient is not None and not isinstance(node, expr.Var):
                self.visit(node, (scope, gradient))

    def visit_Tuple(self, node: expr.Tuple, ctx) -> None:
        scope, gradient = ctx
        for item, part in zip(node.elements, gradient):
            self.accumulate(scope, item, part)

    def visit_Call(self, node: expr.Call, ctx) -> None:
        scope, gradient = ctx
        if isinstance(node.target, core.FunctionRef):
            self.reverse_call(scope, node, gradient)
        else:
            self.reverse_op(scope, node, gradient)

    def reverse_op(self, scope: Scope, call: expr.Call, gradient) -> None:
        """Apply the rule of *call*'s Op to *gradient*, laid out as ``gradient_layout`` first.

        A Reshard's rule lays out its own result, so a Partial gradient reaches the
        source's layout in one Reshard.
        """
        op = type(call.target)
        try:
            rule = autodiff.registry.require(op)
        except dispatch.MissingHandlerError as error:
            raise ValueError(f"differentiate: {op.__name__} has no autodiff rule{located(call)}") from error
        if op is not sharding.Reshard:
            gradient = self.laid_out(gradient, call.type, scope.ctx)
        try:
            gradients = rule(call, gradient, scope.ctx)
        except (ValueError, expr.VerifyError) as error:
            raise ValueError(f"differentiate: {error}{located(call)}") from error
        for arg, part in zip(call.args, gradients):
            self.accumulate(scope, arg, part)

    def reverse_call(self, scope: Scope, call: expr.Call, gradient) -> None:
        ref = call.target
        derived = self.callees[id(call)]
        ctx = scope.ctx
        parts = gradient if isinstance(call.type, types.TupleType) else (gradient,)
        cotangents = []
        for part, wanted in zip(parts, derived.cotangents):
            if wanted is not None:
                cotangents.append(self.finish(part, wanted, ctx))
        placeholder = expr.Var(name="saved", type=derived.saved)
        ctx.remember(placeholder, derived.saved)
        self.placeholders[id(placeholder)] = ("callee", call)
        backward = expr.Call(
            core.FunctionRef(ref.path, ref.name if not ref.path else None),
            (*ref.indices(call.args), *cotangents, placeholder),
            type=derived.gradients,
        )
        ctx.remember(backward, derived.gradients)
        params = core.resolve(self.module, ref)[1].params
        supplied = dict(zip(
            (id(param) for param in params if not (ref.path and param.is_const)), ref.arguments(call.args)
        ))
        for order, index in enumerate(derived.wrt):
            value = backward if len(derived.wrt) == 1 else ctx.call("tuple_get_item", backward, ctx.constant(order))
            self.accumulate(scope, supplied[id(params[index])], self.split(value, ctx))

    def visit_MeshRegion(self, region: core.MeshRegion, ctx) -> None:
        scope, gradient = ctx
        ctx = scope.ctx.enter_region(region)
        active = {id(param) for param, arg in zip(region.params, region.args) if id(arg) in scope.active}
        inner = new_scope((region.body,), region.params, ctx, active, parent=scope, region=region)
        params, args = [], []
        seed = gradient
        for path, value in leaves(gradient):
            param = expr.Var(name=self.fresh(f"d{len(params)}"), type=scope.ctx.type_of(value))
            ctx.remember(param, param.type)
            params.append(param)
            args.append(value)
            seed = placed(seed, path, param)
        self.reverse(inner, ((region.body, seed),))
        outputs = []
        for index, param in enumerate(region.params):
            for path, value in leaves(inner.gradients.get(id(param))):
                outputs.append((index, path, value))
        if not outputs:
            return
        body = pack(tuple(value for _, _, value in outputs))
        ctx.remember(body, body.type)
        node = core.MeshRegion(mesh=region.mesh, body=body, params=tuple(params), args=tuple(args), type=body.type)
        scope.ctx.remember(node, body.type)
        self.regions[id(node)] = RegionBackward(inner, node)
        result = {}
        for position, (index, path, _) in enumerate(outputs):
            value = scope.ctx.call("tuple_get_item", node, scope.ctx.constant(position))
            structure = result.get(index, empty_like(region.params[index].type))
            result[index] = placed(structure, path, value)
        for index, structure in result.items():
            self.accumulate(scope, region.args[index], structure)

    def visit_LoopRegion(self, loop: core.LoopRegion, ctx) -> None:
        """A backward loop running the iterations in reverse.

        Its carries are the gradient of each forward carry and, for each value the
        body reads from outside, a sum of that value's gradient over the iterations.
        """
        scope, gradient = ctx
        trip = trip_count(loop)
        carried = len(loop.yield_values)
        if not carried:
            raise ValueError(f"differentiate: a gradient reaches a loop with no carries{located(loop)}")
        parts = gradient if isinstance(loop.type, types.TupleType) else (gradient,)
        if trip == 0:
            for arg, part in zip(loop.args, parts):
                self.accumulate(scope, arg, part)
            return
        ctx = scope.ctx.enter_region(loop)
        index = expr.Var(name=self.fresh("s"), type=loop.induction_var.type)
        ctx.remember(index, index.type)
        roots = (loop.body, *loop.yield_values)
        active = {id(param) for param, arg in zip(loop.params, loop.args) if id(arg) in scope.active}
        invariant = (loop.induction_var, *loop.params[carried:])
        inner = new_scope(
            roots, (loop.induction_var, *loop.params), ctx, active, scope, loop, index, map(id, invariant)
        )
        while woken := {
            id(param) for param, value in zip(loop.params, loop.yield_values)
            if id(value) in inner.active and id(param) not in inner.active
        }:
            inner.active.update(woken)
            inner.walk(roots)
        carries = [
            position for position in range(carried)
            if id(loop.params[position]) in inner.active and is_float(loop.params[position].type)
        ]
        for position in carries:
            if isinstance(loop.params[position].type, types.TupleType):
                raise ValueError(f"differentiate: a loop carry of tuple type is not differentiated{located(loop)}")
        captures = [
            position for position in range(carried, len(loop.params))
            if id(loop.params[position]) in inner.active and is_float(loop.params[position].type)
        ]
        params, inits = [], []
        for position in carries:
            param = loop.params[position]
            held = expr.Var(name=self.fresh(f"d{param.name}"), type=gradient_type(param.type))
            ctx.remember(held, held.type)
            params.append(held)
            inits.append(self.finish(parts[position], held.type, scope.ctx))
        sums = []
        for position in captures:
            for path, leaf in leaves(flat(loop.params[position].type)):
                held = expr.Var(name=self.fresh(f"g{loop.params[position].name}"), type=gradient_type(leaf))
                ctx.remember(held, held.type)
                sums.append((position, path, leaf, held))
                params.append(held)
                inits.append(scope.ctx.call("zeros", type=held.type))
        self.reverse(inner, [(loop.yield_values[position], held) for position, held in zip(carries, params)])
        updates = [
            self.finish(inner.gradients.get(id(loop.params[position])), held.type, ctx)
            for position, held in zip(carries, params)
        ]
        for position, path, leaf, held in sums:
            part = at(inner.gradients.get(id(loop.params[position])), path)
            updates.append(held if part is None else ctx.call("add", held, self.finish(part, leaf, ctx)))
        result_type = updates[0].type if len(updates) == 1 else types.TupleType(tuple(ctx.type_of(u) for u in updates))
        node = core.LoopRegion(
            induction_var=index, params=tuple(params), args=tuple(inits), body=updates[-1],
            yield_values=tuple(updates), extent=trip, step=1, start=0, type=result_type,
        )
        scope.ctx.remember(node, result_type)
        self.loops[id(node)] = RegionBackward(inner, node)

        def result(position):
            if len(updates) == 1:
                return node
            return scope.ctx.call("tuple_get_item", node, scope.ctx.constant(position))

        for slot, position in enumerate(carries):
            self.accumulate(scope, loop.args[position], result(slot))
        structures = {}
        for slot, (position, path, _, _) in enumerate(sums, start=len(carries)):
            held = structures.get(position, empty_like(loop.params[position].type))
            structures[position] = placed(held, path, result(slot))
        for position, structure in structures.items():
            self.accumulate(scope, loop.args[position], structure)

    def rebuild_region(self, backward: RegionBackward, scope: Scope) -> core.MeshRegion:
        node = backward.node
        inner = backward.scope
        body = inner.rebuilt(self, node.body)
        args = tuple(self.named(arg, scope) for arg in node.args)
        captures = sorted((*zip(node.params, args), *inner.region_params), key=capture_name)
        return replace(
            node,
            body=body,
            params=tuple(param for param, _ in captures),
            args=tuple(value for _, value in captures),
        )

    def named(self, node: expr.Expr, scope: Scope) -> expr.Expr:
        """*node* rebuilt with a binding name, so a region capturing it sorts by its printed name.

        Every later read of *node* in *scope* gets the same named value.
        """
        named = self.bound(scope.rebuilt(self, node), "grad")
        scope.rebuilder.keep(node, named)
        return named

    def rebuild_loop(self, backward: RegionBackward, scope: Scope) -> core.LoopRegion:
        node = backward.node
        inner = backward.scope
        updates = tuple(inner.rebuilt(self, value) for value in node.yield_values)
        captures = sorted(inner.region_params, key=capture_name)
        return replace(
            node,
            body=updates[-1],
            yield_values=updates,
            params=(*node.params, *(param for param, _ in captures)),
            args=(*(scope.rebuilt(self, arg) for arg in node.args), *(value for _, value in captures)),
        )

    def materialize(self, key: tuple, scope: Scope) -> expr.Expr:
        """The backward's access, in *scope*, to the forward value *key* names."""
        slot = (key[0], id(key[1]), *key[2:])
        held = scope.slots.get(slot)
        if held is not None:
            return held
        value = key[1]
        if key[0] == "node" and isinstance(value, (expr.Constant, core.ShapeOf)):
            result = replace(value)
        elif key[0] == "node" and isinstance(value, expr.Tuple):
            result = replace(value, elements=tuple(self.materialize(("node", item), scope) for item in value.elements))
        elif key[0] == "node" and (id(value) in self.differ.recompute or self.recomputed(value, scope)):
            result = replace(value, args=tuple(self.materialize(("node", arg), scope) for arg in value.args))
        elif scope.parent is None:
            result = projection(self.saved, len(self.saved_keys), self.key_type(key), self.read_name(key))
            self.saved_keys.append(key)
        elif isinstance(scope.region, core.LoopRegion):
            result = self.materialize_in_loop(key, scope)
        else:
            if key[0] == "node" and any(value is param for param in scope.region.params):
                index = next(i for i, param in enumerate(scope.region.params) if param is value)
                outer = self.capture(("node", scope.region.args[index]), scope.parent)
            else:
                scope.extras.append(key)
                outer = self.capture(("extra", scope, len(scope.extras) - 1), scope.parent)
            if is_projection(outer) and outer.args[0] is self.saved:
                index = outer.args[1].value
                result = projection(self.saved_in(scope), index, self.key_type(key), self.read_name(key))
            else:
                result = expr.Var(name=self.fresh(self.key_name(key)[0] or "saved"), type=self.key_type(key))
                scope.region_params.append((result, outer))
        scope.remember_slot(slot, result)
        return result

    def capture(self, key: tuple, scope: Scope) -> expr.Expr:
        """``materialize``, with a binding name, for a region of *scope* to capture.

        The parser sorts a region's captures by their printed names, so each one
        needs its name before the printer would make one up.
        """
        named = self.bound(self.materialize(key, scope), self.key_name(key)[0] or "saved")
        scope.remember_slot((key[0], id(key[1]), *key[2:]), named)
        return named

    def bound(self, value: expr.Expr, base: str) -> expr.Expr:
        """*value*, bound to a fresh name from *base* unless it already has a name."""
        if isinstance(value, expr.Var) or expr.binding_name(value):
            return value
        return replace(value, metadata=(*value.metadata, expr.BindingMetadata(name=self.fresh(base))))

    def saved_in(self, scope: Scope) -> expr.Var:
        """``saved`` as a parameter of the region *scope*, so its items are read inside the mesh."""
        held = scope.slots.get(("saved",))
        if held is None:
            held = expr.Var(name=self.saved.name, type=self.saved.type)
            scope.region_params.append((held, self.saved))
            scope.remember_slot(("saved",), held)
        return held

    def read_name(self, key: tuple) -> str:
        """A name of its own for the backward's read of the saved value *key*."""
        return self.fresh(self.key_name(key)[0] or "saved_value")

    def materialize_in_loop(self, key: tuple, scope: Scope) -> expr.Expr:
        """The backward loop's access to a forward loop value.

        The induction value is computed again from the backward's own index. A
        value the body reads from outside is one more invariant parameter. Every
        other value differs per iteration: the forward stacks it, and the backward
        reads the row of the iteration it is undoing.
        """
        loop, ctx, value = scope.region, scope.ctx, key[1]
        trip = trip_count(loop)
        ordinal = ctx.call("sub", ctx.constant(trip - 1, scope.index.type.dtype), scope.index)
        if key[0] == "node" and value is loop.induction_var:
            coordinate = ordinal
            if loop.step != 1:
                coordinate = ctx.call("mul", coordinate, ctx.constant(loop.step, scope.index.type.dtype))
            if loop.start != 0:
                coordinate = ctx.call("add", coordinate, ctx.constant(loop.start, scope.index.type.dtype))
            return coordinate
        if key[0] == "callee" and len(key) == 2:
            return self.callee_saved_in_loop(value, scope)
        carried = len(loop.yield_values)
        if key[0] == "node" and any(value is param for param in loop.params[carried:]):
            position = next(i for i, param in enumerate(loop.params) if param is value)
            outer = self.capture(("node", loop.args[position]), scope.parent)
            result = expr.Var(name=self.fresh(value.name), type=self.key_type(key))
            ctx.remember(result, result.type)
            scope.region_params.append((result, outer))
            return result
        held = self.key_type(key)
        owners = ctx.resolve_target().storage_owners
        scope.extras.append(key)
        outer = self.capture(("extra", scope, len(scope.extras) - 1), scope.parent)
        stacks = expr.Var(
            name=self.fresh(f"stack_{self.key_name(key)[0] or 'saved'}"),
            type=stacked(held, trip, owners, located(self.key_node(key))),
        )
        ctx.remember(stacks, stacks.type)
        scope.region_params.append((stacks, outer))
        return self.row(stacks, held, ordinal, ctx)

    def recomputed(self, value: expr.Expr, scope: Scope) -> bool:
        """Whether *value*, of a loop body, is an Op's result that only invariant values reach.

        The backward body computes such a value again instead of stacking it: an
        index, or one layer's slice of a stacked weight.
        """
        return isinstance(scope.region, core.LoopRegion) and isinstance(value, expr.Call) and (
            id(value) in scope.invariant
        )

    def callee_saved_in_loop(self, call: expr.Call, scope: Scope) -> expr.Expr:
        """A loop body callee's ``saved``, one item at a time.

        An item that is a parameter of the callee, passed a loop invariant value,
        is that value again; every other item is stacked on its own.
        """
        derived = self.callees[id(call)]
        ref = call.target
        params = core.resolve(self.module, ref)[1].params
        supplied = dict(zip(
            (index for index, param in enumerate(params) if not (ref.path and param.is_const)),
            ref.arguments(call.args),
        ))
        items = []
        for position, index in enumerate(derived.saved_params):
            arg = supplied.get(index)
            if arg is not None and id(arg) in scope.invariant:
                items.append(self.materialize(("node", arg), scope))
            else:
                items.append(self.materialize(("callee", call, position), scope))
        return scope.ctx.remember(pack(tuple(items)), derived.saved)

    def row(self, stacks: expr.Expr, type_: types.Type, row: expr.Expr, ctx: autodiff.Context) -> expr.Expr:
        """Row *row*, a scalar, of the stacks of a value of *type_*, shaped as that value."""
        if isinstance(type_, types.TupleType):
            value = pack(tuple(
                self.row(ctx.call("tuple_get_item", stacks, ctx.constant(index)), field, row, ctx)
                for index, field in enumerate(type_.fields)
            ))
            return ctx.remember(value, type_)
        starts = (row, *(ctx.constant(0, ctx.type_of(row).dtype) for _ in type_.shape))
        corner = ctx.remember(pack(starts), types.TupleType(tuple(ctx.type_of(start) for start in starts)))
        rank = len(type_.shape) + 1
        taken = ctx.call("slice", stacks, corner, sizes=(1, *type_.shape), strides=(1,) * rank)
        value = ctx.call("reshape", taken, new_shape=tuple(type_.shape))
        if ctx.type_of(value).layout == type_.layout:
            return value
        return ctx.call("reshard", value, layout=types.normalize_layout(type_.layout), storage=type_.storage)

    def key_node(self, key: tuple) -> expr.Expr:
        """The forward value *key* comes from: an inner region's value for an extra result."""
        if key[0] == "extra":
            _, scope, index = key
            return self.key_node(scope.extras[index])
        return key[1]

    def key_type(self, key: tuple) -> types.Type:
        if key[0] == "node":
            return key[1].type
        if key[0] == "callee":
            saved = self.callees[id(key[1])].saved
            return saved if len(key) == 2 else saved.fields[key[2]]
        _, scope, index = key
        inner = self.key_type(scope.extras[index])
        if isinstance(scope.region, core.LoopRegion):
            owners = scope.ctx.resolve_target().storage_owners
            return stacked(inner, trip_count(scope.region), owners, "")
        return inner

    def record(self, index: int, key: tuple) -> Saved:
        name, origin = self.key_name(key)
        return Saved(name or f"v{index}", self.key_type(key), origin, self.storage(key))

    def storage(self, key: tuple) -> Saved | None:
        """The storage a saved view keeps alive; None for a value that is its own, or not a node."""
        if key[0] != "node":
            return None
        owners = (self.module.target or self.differ.target).storage_owners
        root = key[1]
        while isinstance(root, expr.Call) and not isinstance(root.target, core.FunctionRef):
            position = access.aliased_operand(root, owners)
            if position is None:
                break
            root = root.args[position]
        if root is key[1]:
            return None
        name, origin = self.key_name(("node", root))
        return Saved(name or origin.replace("  ", "@"), root.type, origin)

    def key_name(self, key: tuple) -> tuple[str | None, str]:
        """The name and the origin a saved value is reported with."""
        if key[0] == "extra":
            _, scope, index = key
            name, origin = self.key_name(scope.extras[index])
            if isinstance(scope.region, core.LoopRegion):
                return name, f"stack of {trip_count(scope.region)}: {origin}"
            return name, origin
        if key[0] == "callee":
            ref = key[1].target
            item = "" if len(key) == 2 else f"[{key[2]}]"
            return f"{ref.qualified_name}.saved{item}", f"call {ref.qualified_name}  {expr.span_text(key[1], column=False, short=True) or ''}".rstrip()
        node = key[1]
        if isinstance(node, expr.Var):
            return node.name, "parameter" if any(node is param for param in self.function.params) else "carry"
        if not isinstance(node, expr.Call):
            name = type(node).__name__
        elif isinstance(node.target, core.FunctionRef):
            name = f"call {node.target.qualified_name}"
        else:
            kind = getattr(node.target, "kind", None)
            name = kind.value if isinstance(kind, Enum) else expr.op_registry.schema_of(type(node.target)).name
        return expr.binding_name(node), f"{name}  {expr.span_text(node, column=False, short=True) or ''}".rstrip()

    def forward(self) -> core.Function:
        function = self.function
        params = tuple(replace(param) for param in function.params)
        cloner = ForwardCloner(self)
        env = {id(old): new for old, new in zip(function.params, params)}
        body = cloner.visit(function.body, env)
        results = fields_of(function.return_type)
        if not isinstance(function.return_type, types.TupleType):
            kept = (body,)
        elif isinstance(body, expr.Tuple):
            kept = body.elements
        else:
            kept = unpack(body, results)
        saved = pack(tuple(cloner.saved_value(key, env) for key in self.saved_keys))
        return core.Function.from_parts(
            name=function.name,
            params=params,
            body=pack((*kept, saved)),
            return_type=types.TupleType((*results, self.saved.type)),
        )


def rebuild_structure(type_: types.Type, found) -> expr.Expr:
    """The values *found*, one per tensor leaf of *type_*, packed as *type_* nests them."""
    if isinstance(type_, types.TupleType):
        return pack(tuple(rebuild_structure(item, found) for item in type_.fields))
    return next(found)


class Rebuilder(core.ExprCloner):
    """A copy of one scope's backward values that reads each forward value through its slot.

    A forward value of the scope, a callee's saved placeholder, and a backward
    region or loop are known by identity; any other value is copied with its
    operands rebuilt.
    """

    def __init__(self, owner: FunctionDifferentiator, scope: Scope) -> None:
        super().__init__()
        self.owner = owner
        self.scope = scope

    def default_visit(self, node: expr.Expr, ctx) -> expr.Expr:
        owner, scope = self.owner, self.scope
        if id(node) in scope.nodes:
            return owner.materialize(("node", node), scope)
        if id(node) in owner.placeholders:
            return owner.materialize(owner.placeholders[id(node)], scope)
        if id(node) in owner.regions:
            return owner.rebuild_region(owner.regions[id(node)], scope)
        if id(node) in owner.loops:
            return owner.rebuild_loop(owner.loops[id(node)], scope)
        return super().default_visit(node, ctx)

    def keep(self, node: expr.Expr, value: expr.Expr) -> None:
        """Answer *value* for every later visit of *node*."""
        self.memo[id(node)] = (node, value)


class Inliner(core.BindingSubstitutionCloner):
    """A copy with each region of one mesh replaced by its body, its params bound to its args."""

    def __init__(self, mesh: types.Mesh) -> None:
        super().__init__()
        self.mesh = mesh

    def visit_MeshRegion(self, node: core.MeshRegion, ctx) -> expr.Expr:
        if node.mesh != self.mesh:
            return super().visit_MeshRegion(node, ctx)
        inner = dict(ctx)
        inner.update((id(param), self.visit(arg, ctx)) for param, arg in zip(node.params, node.args))
        return self.visit(node.body, inner)

    def visit_Call(self, node: expr.Call, ctx) -> expr.Expr:
        copied = super().visit_Call(node, ctx)
        held, index = copied.args[0] if copied.args else None, copied.args[-1] if copied.args else None
        if is_projection(copied) and isinstance(held, expr.Tuple) and isinstance(index, expr.Constant):
            return held.elements[index.value]
        return copied


def enclose(body: expr.Expr, mesh: types.Mesh, params: tuple[expr.Var, ...]) -> core.MeshRegion:
    """*body* as the whole-function region of *mesh*, the backward regions of that mesh inlined.

    A function that opens a mesh runs no op outside it. The region binds one
    copy of each parameter, in order, to that parameter; it prints as a
    ``with`` on the class body's mesh.
    """
    inner = tuple(replace(param) for param in params)
    region_body = Inliner(mesh).visit(body, {id(old): new for old, new in zip(params, inner)})
    return core.MeshRegion(mesh=mesh, params=inner, args=params, body=region_body, type=region_body.type)


class ForwardCloner(core.BindingSubstitutionCloner):
    """A copy of the forward that also returns, last, what the backward reads."""

    def __init__(self, owner: FunctionDifferentiator) -> None:
        super().__init__()
        self.owner = owner
        self.calls: dict[int, expr.Call] = {}
        self.regions: dict[int, tuple[core.MeshRegion, int]] = {}
        self.scopes = {
            id(backward.scope.region): backward.scope
            for backward in (*owner.regions.values(), *owner.loops.values())
        }
        self.loops: dict[int, tuple[core.LoopRegion, int, tuple[int, ...]]] = {}

    def visit_Call(self, node: expr.Call, ctx) -> expr.Expr:
        copied = super().visit_Call(node, ctx)
        if not isinstance(node.target, core.FunctionRef):
            return copied
        self.calls[id(node)] = copied
        return repack(copied, node.type)

    def visit_MeshRegion(self, node: core.MeshRegion, ctx) -> expr.Expr:
        copied = super().visit_MeshRegion(node, ctx)
        scope = self.scopes.get(id(node))
        if scope is None or not scope.extras:
            return copied
        inner = dict(ctx)
        inner.update((id(old), new) for old, new in zip(node.params, copied.params))
        results = fields_of(node.type)
        if isinstance(node.type, types.TupleType):
            kept = (
                copied.body.elements
                if isinstance(copied.body, expr.Tuple)
                else unpack(copied.body, results)
            )
        else:
            kept = (copied.body,)
        extras = tuple(self.saved_value(key, inner) for key in scope.extras)
        body = pack((*kept, *extras))
        region = replace(copied, body=body, type=body.type)
        self.regions[id(node)] = (region, len(results))
        return repack(region, node.type)

    def visit_LoopRegion(self, node: core.LoopRegion, ctx) -> expr.Expr:
        """The loop with one more carry per tensor the backward reads from an iteration.

        Each such carry is a stack: iteration ``(t - start) // step`` writes its
        row, and the stacks leave the loop as extra results.
        """
        copied = super().visit_LoopRegion(node, ctx)
        scope = self.scopes.get(id(node))
        if scope is None or not scope.extras:
            return copied
        inner = dict(ctx)
        inner[id(node.induction_var)] = copied.induction_var
        inner.update((id(old), new) for old, new in zip(node.params, copied.params))
        dtype = node.induction_var.type.dtype
        ordinal = copied.induction_var
        if node.start != 0:
            ordinal = autodiff.build("sub", ordinal, autodiff.scalar(node.start, dtype))
        if node.step != 1:
            ordinal = autodiff.build("floor_div", ordinal, autodiff.scalar(node.step, dtype))
        trip = trip_count(node)
        carried = len(node.yield_values)
        stacks, inits, updates, counts = [], [], [], []
        for key in scope.extras:
            value = self.saved_value(key, inner)
            type_ = self.owner.key_type(key)
            found = list(leaves(flat(type_)))
            counts.append(len(found))
            for path, leaf in found:
                piece = value
                for index in path:
                    piece = projection(piece, index, leaf)
                stack_type = stacked(leaf, trip, scope.ctx.resolve_target().storage_owners, "")
                stack = expr.Var(name=self.owner.fresh(f"stack_{self.owner.key_name(key)[0] or 'v'}"), type=stack_type)
                stacks.append(stack)
                inits.append(autodiff.build("zeros", type=stack_type))
                corner = pack((ordinal, *(autodiff.scalar(0, dtype) for _ in leaf.shape)))
                updates.append(autodiff.build(
                    "insert_slice", stack, autodiff.build("reshape", piece, new_shape=(1, *leaf.shape)), corner
                ))
        loop = replace(
            copied,
            params=(*copied.params[:carried], *stacks, *copied.params[carried:]),
            args=(*copied.args[:carried], *inits, *copied.args[carried:]),
            yield_values=(*copied.yield_values, *updates),
            body=updates[-1],
            type=types.TupleType(()),
        )
        self.loops[id(node)] = (loop, carried, tuple(counts))
        if carried == 1:
            return projection(loop, 0, node.type)
        return pack(tuple(projection(loop, index, item) for index, item in enumerate(node.type.fields)))

    def saved_value(self, key: tuple, env) -> expr.Expr:
        """The forward value *key* names, in the scope *env* binds."""
        if key[0] == "node":
            return self.visit(key[1], env)
        if key[0] == "callee":
            derived = self.owner.callees[id(key[1])]
            saved = projection(self.calls[id(key[1])], derived.results, derived.saved)
            return saved if len(key) == 2 else projection(saved, key[2], derived.saved.fields[key[2]])
        _, scope, index = key
        self.visit(scope.region, env)
        if isinstance(scope.region, core.LoopRegion):
            loop, carried, counts = self.loops[id(scope.region)]
            type_ = self.owner.key_type(key)
            first = carried + sum(counts[:index])
            found = [projection(loop, first + offset, leaf) for offset, (_, leaf) in enumerate(leaves(flat(type_)))]
            return rebuild_structure(type_, iter(found))
        region, results = self.regions[id(scope.region)]
        return projection(region, results + index, self.owner.key_type(key))
