"""Reverse-mode differentiation: one forward Module becomes a forward and a backward Module.

The forward returns what the original returned and, last, a ``saved`` tuple of
the values the backward reads. The backward takes the gradient of each floating
point result and ``saved``, and returns the gradient of each ``wrt`` parameter.

See [passes §7.4](docs/spec/passes.md#74-differentiate).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from tilefoundry import dispatch, verifier
from tilefoundry.interfaces import autodiff, typeinfer
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import tensor

__all__ = ["Saved", "differentiate", "saved_values", "select_value"]

GENERATED = "autodiff"


@dataclass(frozen=True)
class Saved:
    """One value the forward keeps for the backward."""

    name: str
    type: types.Type
    origin: str


def differentiate(
    module: core.Module, *, wrt: tuple[str, ...], recompute: tuple[str, ...] = ()
) -> tuple[core.Module, core.Module]:
    """Return ``(<Name>Forward, <Name>Backward)`` for the entry of *module*.

    *wrt* names the entry parameters whose gradients the backward returns, in
    that order. Each item of *recompute* is a ``Module[.function]:variable``
    selector of a value the backward computes again instead of reading it from
    ``saved``.
    """
    if module.metadata.get(GENERATED) == "backward":
        raise ValueError(
            f"differentiate: {module.name!r} is a backward that differentiate generated; "
            "a second derivative is not supported"
        )
    wrt = tuple(wrt)
    if not wrt:
        raise ValueError(f"differentiate: {module.name!r}: wrt names no parameter")
    selected = {id(value): value for value in (select_value(module, item) for item in recompute)}
    return Differentiator(selected).module(module, wrt)


def saved_values(forward: core.Module) -> tuple[Saved, ...]:
    """What *forward* keeps for its backward, in the order of the saved tuple."""
    if forward.metadata.get(GENERATED) != "forward":
        raise ValueError(f"saved_values: {forward.name!r} is not a forward that differentiate generated")
    return forward.metadata["saved"]


def select_value(module: core.Module, selector: str) -> expr.Expr:
    """Resolve ``Module[.function]:variable`` among the Modules reachable from *module*.

    The part before ``:`` is the selector of ``tilefoundry`` commands: a Module
    named by ``Module.name`` and, optionally, one of its functions; the entry
    when the function is left out. The variable is the name the author bound.
    """
    owner, colon, variable = selector.rpartition(":")
    if not colon or not owner or not variable:
        raise ValueError(f"recompute {selector!r}: expected Module[.function]:variable")
    name, _, function_name = owner.partition(".")
    if "." in function_name:
        raise ValueError(
            f"recompute {selector!r}: name a Module by its name, not by a path through child slots"
        )
    found = {id(each): each for each in reachable(module) if each.name == name}
    if not found:
        raise ValueError(f"recompute {selector!r}: no Module named {name!r} is reachable from {module.name!r}")
    if len(found) > 1:
        raise ValueError(f"recompute {selector!r}: {len(found)} different Modules are named {name!r}")
    (owner_module,) = found.values()
    function = owner_module.lookup(function_name) if function_name else owner_module.entry_function()
    if any(param.name == variable for param in function.params):
        raise ValueError(
            f"recompute {selector!r}: {variable!r} is a parameter; the backward reads a parameter, "
            "it never computes one"
        )
    matches = {
        id(value): value for value in core.collect_exprs(function.body) if expr.binding_name(value) == variable
    }
    if not matches:
        raise ValueError(f"recompute {selector!r}: {function.name!r} binds no value named {variable!r}")
    if len(matches) > 1:
        places = ", ".join(where(value) or "?" for value in matches.values())
        raise ValueError(f"recompute {selector!r}: {len(matches)} values are named {variable!r}: {places}")
    (value,) = matches.values()
    for loop in core.collect_exprs(function.body):
        inside = () if not isinstance(loop, core.LoopRegion) else (loop.body, *loop.yield_values)
        if any(value is inner for root in inside for inner in core.collect_exprs(root)):
            raise ValueError(
                f"recompute {selector!r}: {variable!r} is computed inside a loop; "
                "the first version saves every value of a loop body and recomputes none"
            )
    produced = value.args[0] if is_projection(value) else value
    if isinstance(produced, expr.Call) and isinstance(produced.target, core.FunctionRef):
        raise ValueError(
            f"recompute {selector!r}: {variable!r} is the result of calling "
            f"{produced.target.qualified_name!r}; only a value an Op computes is recomputed"
        )
    if not isinstance(value, expr.Call):
        raise ValueError(
            f"recompute {selector!r}: {variable!r} is a {type(value).__name__}; "
            "only a value an Op computes is recomputed"
        )
    return value


def reachable(module: core.Module) -> tuple[core.Module, ...]:
    """*module* and every Module in its child slots, at any depth."""
    seen: dict[int, core.Module] = {}
    pending = [module]
    while pending:
        each = pending.pop()
        if id(each) not in seen:
            seen[id(each)] = each
            for slot in each.children.values():
                pending.extend(core.instances(slot))
    return tuple(seen.values())


def span_of(node: expr.Expr, *, short: bool = False) -> str | None:
    """``file:line`` of *node*, the file by its base name when *short*."""
    span = expr.get_metadata(node, expr.SourceSpanMetadata)
    if span is None:
        return None
    return f"{Path(span.file).name if short else span.file}:{span.line}"


def where(node: expr.Expr) -> str:
    """Where the author wrote *node*, and the name they bound it to."""
    span = span_of(node)
    name = expr.binding_name(node)
    return " ".join(part for part in (span, f"({name})" if name else None) if part)


def located(node: expr.Expr) -> str:
    text = where(node)
    return f"\n  at {text}" if text else ""


def is_projection(node: expr.Expr) -> bool:
    return isinstance(node, expr.Call) and isinstance(node.target, tensor.TupleGetItem)


def is_float(type_: types.Type) -> bool:
    if isinstance(type_, types.TupleType):
        return any(is_float(item) for item in type_.fields)
    return isinstance(type_, types.TensorType) and isinstance(type_.dtype, expr.FloatDType)


def gradient_type(type_: types.Type) -> types.Type:
    """The type of the gradient of a value of *type_*: its layout without ``Partial``."""
    if isinstance(type_, types.TupleType):
        return types.TupleType(tuple(gradient_type(item) for item in type_.fields))
    return replace(type_, layout=autodiff.gradient_layout(type_.layout))


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


def projection(value: expr.Expr, index: int, type_: types.Type) -> expr.Call:
    """``value[index]`` as an untyped-by-inference Call; the final inference types it."""
    constant = expr.Constant(
        type=types.TensorType.from_scalar(expr.DType.i64, storage=expr.StorageKind.UMAT), value=index
    )
    return expr.Call(tensor.TupleGetItem(), (value, constant), type=type_)


def pack(items: tuple[expr.Expr, ...]) -> expr.Tuple:
    return expr.Tuple(elements=tuple(items), type=types.TupleType(tuple(item.type for item in items)))


def build(name: str, *args: expr.Expr, **attrs) -> expr.Call:
    """``tf.<name>(*args, **attrs)`` with a placeholder type; the final inference types it."""
    schema = expr.op_registry.lookup("tf", (), name)
    return expr.Call(schema.builder(**attrs), args, type=types.TupleType(()))


def integer(value: int, dtype: expr.DType) -> expr.Constant:
    return expr.Constant(type=types.TensorType.from_scalar(dtype, storage=expr.StorageKind.UMAT), value=value)


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


def scope_order(*roots: expr.Expr) -> list[expr.Expr]:
    """The values of one scope, operands first; a region's body is a scope of its own."""
    order: list[expr.Expr] = []
    seen: set[int] = set()
    stack = [(root, False) for root in reversed(roots)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        if isinstance(node, (core.MeshRegion, core.LoopRegion)):
            children = node.args
        else:
            children = core.expr_children(node)
        stack.extend((child, False) for child in reversed(children) if id(child) not in seen)
    return order


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
    rebuilt: dict[int, expr.Expr] = field(default_factory=dict)

    def activate(self, ids: set[int]) -> None:
        self.active.update(ids)

    def remember_gradient(self, node: expr.Expr, gradient) -> None:
        self.gradients[id(node)] = gradient

    def remember_slot(self, slot: tuple, value: expr.Expr) -> None:
        self.slots[slot] = value

    def remember_rebuilt(self, node: expr.Expr, value: expr.Expr) -> None:
        self.rebuilt[id(node)] = value


def mark_active(scope: Scope, order: list[expr.Expr]) -> None:
    """Add to ``scope.active`` each value of *order* that reads an active value."""
    for node in order:
        if id(node) in scope.active:
            continue
        if isinstance(node, (core.MeshRegion, core.LoopRegion)):
            children = node.args
        elif isinstance(node, (expr.Call, expr.Tuple)):
            children = core.expr_children(node)
        else:
            continue
        if any(id(child) in scope.active for child in children):
            scope.activate({id(node)})


def new_scope(roots, params, ctx, active: set[int], parent=None, region=None, index=None) -> Scope:
    nodes = {id(node): node for node in (*params, *scope_order(*roots))}
    return Scope(nodes=nodes, ctx=ctx, parent=parent, region=region, active=active, index=index)


@dataclass
class RegionBackward:
    """The backward of one forward MeshRegion, before its saved values are known."""

    scope: Scope
    node: core.MeshRegion


class Differentiator:
    """Differentiate a Module tree, each Module and function once."""

    def __init__(self, recompute: dict[int, expr.Expr]) -> None:
        self.recompute = recompute
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
        children_forward = {key: tuple(pair[0] for pair in slot) for key, slot in self.slots[id(module)].items()}
        children_backward = {key: tuple(pair[1] for pair in slot) for key, slot in self.slots[id(module)].items()}

        def slot_of(key, derived):
            return derived if isinstance(module.children[key], tuple) else derived[0]

        forward = core.Module(
            name=f"{module.name}Forward",
            functions=tuple(derived.forward for _, derived in functions),
            entry=module.entry,
            children={key: slot_of(key, value) for key, value in children_forward.items()},
            target=module.target,
            topologies=module.topologies,
            metadata={GENERATED: "forward", "saved": functions[0][1].saved_values},
        )
        backward = core.Module(
            name=f"{module.name}Backward",
            functions=tuple(derived.backward for _, derived in functions),
            entry="backward",
            children={key: slot_of(key, value) for key, value in children_backward.items()},
            target=module.target,
            topologies=module.topologies,
            metadata={GENERATED: "backward"},
        )
        for generated in (forward, backward):
            for function in generated.functions:
                typeinfer.infer_function(function, typeinfer.TypeInferContext(module=generated, function=function))
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
            if id(function) in self.functions:
                return self.functions[id(function)]
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
        if not all(
            isinstance(leaf, types.TensorType) and isinstance(leaf.dtype, expr.FloatDType)
            for leaf in flat_types(param.type)
        ):
            raise ValueError(
                f"differentiate: wrt {name!r} is not a floating point tensor or a tuple of them"
            )
        indices.append(matches[0])
    return tuple(indices)


def flat_types(type_: types.Type):
    if isinstance(type_, types.TupleType):
        for item in type_.fields:
            yield from flat_types(item)
    else:
        yield type_


class FunctionDifferentiator:
    """The reverse pass over one function, then the forward and backward built from it."""

    def __init__(self, differ: Differentiator, module: core.Module, function: core.Function, wrt) -> None:
        self.differ = differ
        self.module = module
        self.function = function
        self.wrt = wrt
        self.placeholders: dict[int, tuple] = {}
        self.regions: dict[int, RegionBackward] = {}
        self.loops: dict[int, RegionBackward] = {}
        self.callees: dict[int, Derived] = {
            id(value): differ.callee(module, value.target, value)
            for value in core.collect_exprs(function.body)
            if isinstance(value, expr.Call) and isinstance(value.target, core.FunctionRef)
        }
        self.saved_keys: list[tuple] = []
        memo = {}
        for node in core.function_values(function):
            memo[id(node)] = (node, node.type)
        ctx = autodiff.Context(module=module, function=function, memo=memo)
        active = {id(function.params[index]) for index in wrt}
        self.top = new_scope((function.body,), function.params, ctx, active)
        self.saved = expr.Var(name="saved", type=types.TupleType(()))

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
            param = expr.Var(name=names[0] if len(names) == 1 else names[index], type=type_)
            self.top.ctx.remember(param, type_)
            params.append(param)
            seeds.append(self.split(param, self.top.ctx))
        seed = tuple(seeds) if isinstance(function.return_type, types.TupleType) else seeds[0]
        self.reverse(self.top, (function.body,), ((function.body, seed),))
        outputs = tuple(
            self.finish(self.top.gradients.get(id(function.params[index])), function.params[index].type, self.top.ctx)
            for index in self.wrt
        )
        gradients = types.TupleType(tuple(gradient_type(function.params[index].type) for index in self.wrt))
        returned = outputs[0] if len(outputs) == 1 else pack(outputs)
        body = self.rebuild(returned, self.top)
        saved = types.TupleType(tuple(self.key_type(key) for key in self.saved_keys))
        self.saved.restamp(saved, saved)
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
            return pack(tuple(self.finish(part, item, ctx) for part, item in zip(parts, type_.fields)))
        wanted = gradient_type(type_)
        if gradient is None:
            return ctx.call("zeros", type=wanted)
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

    def reverse(self, scope: Scope, roots: tuple[expr.Expr, ...], seeds) -> None:
        """Walk the values *roots* reach backward from *seeds*, ``(value, gradient)`` pairs."""
        order = scope_order(*roots)
        mark_active(scope, order)
        for node, seed in seeds:
            self.accumulate(scope, node, seed)
        for node in reversed(order):
            gradient = scope.gradients.get(id(node))
            if gradient is None or isinstance(node, expr.Var):
                continue
            if isinstance(node, expr.Tuple):
                for item, part in zip(node.elements, gradient):
                    self.accumulate(scope, item, part)
            elif isinstance(node, core.MeshRegion):
                self.reverse_region(scope, node, gradient)
            elif isinstance(node, core.LoopRegion):
                self.reverse_loop(scope, node, gradient)
            elif isinstance(node.target, core.FunctionRef):
                self.reverse_call(scope, node, gradient)
            else:
                self.reverse_op(scope, node, gradient)

    def reverse_op(self, scope: Scope, call: expr.Call, gradient) -> None:
        op = type(call.target)
        try:
            rule = autodiff.registry.require(op)
        except dispatch.MissingHandlerError as error:
            raise ValueError(f"differentiate: {op.__name__} has no autodiff rule{located(call)}") from error
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
                cotangents.append(self.cotangent(part, wanted, ctx))
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

    def cotangent(self, part, wanted: types.Type, ctx: autodiff.Context) -> expr.Expr:
        """*part* as the gradient argument a callee's backward declares as *wanted*."""
        if isinstance(wanted, types.TupleType):
            parts = part if isinstance(part, tuple) else (None,) * len(wanted.fields)
            elements = tuple(self.cotangent(item, field, ctx) for item, field in zip(parts, wanted.fields))
            value = pack(elements)
            ctx.remember(value, wanted)
            return value
        if part is None:
            return ctx.call("zeros", type=wanted)
        if ctx.type_of(part).layout != wanted.layout:
            return ctx.call("reshard", part, layout=types.normalize_layout(wanted.layout), storage=wanted.storage)
        return part

    def reverse_region(self, scope: Scope, region: core.MeshRegion, gradient) -> None:
        ctx = scope.ctx.enter_region(region)
        active = {id(param) for param, arg in zip(region.params, region.args) if id(arg) in scope.active}
        inner = new_scope((region.body,), region.params, ctx, active, parent=scope, region=region)
        params, args = [], []
        seed = gradient
        for path, value in leaves(gradient):
            param = expr.Var(name=f"d{len(params)}", type=scope.ctx.type_of(value))
            ctx.remember(param, param.type)
            params.append(param)
            args.append(value)
            seed = placed(seed, path, param)
        self.reverse(inner, (region.body,), ((region.body, seed),))
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

    def reverse_loop(self, scope: Scope, loop: core.LoopRegion, gradient) -> None:
        """A backward loop running the iterations in reverse.

        Its carries are the gradient of each forward carry and, for each value the
        body reads from outside, a sum of that value's gradient over the iterations.
        """
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
        index = expr.Var(name="s", type=loop.induction_var.type)
        ctx.remember(index, index.type)
        roots = (loop.body, *loop.yield_values)
        active = {id(param) for param, arg in zip(loop.params, loop.args) if id(arg) in scope.active}
        inner = new_scope(roots, (loop.induction_var, *loop.params), ctx, active, scope, loop, index)
        order = scope_order(*roots)
        while True:
            mark_active(inner, order)
            woken = {
                id(param) for param, value in zip(loop.params, loop.yield_values)
                if id(value) in inner.active and id(param) not in inner.active
            }
            if not woken:
                break
            inner.activate(woken)
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
            held = expr.Var(name=f"d{param.name}", type=gradient_type(param.type))
            ctx.remember(held, held.type)
            params.append(held)
            inits.append(self.cotangent(parts[position], held.type, scope.ctx))
        sums = []
        for position in captures:
            for path, leaf in leaves(flat(loop.params[position].type)):
                held = expr.Var(name=f"g{loop.params[position].name}", type=gradient_type(leaf))
                ctx.remember(held, held.type)
                sums.append((position, path, leaf, held))
                params.append(held)
                inits.append(scope.ctx.call("zeros", type=held.type))
        self.reverse(inner, roots, [(loop.yield_values[position], held) for position, held in zip(carries, params)])
        updates = [
            self.cotangent(inner.gradients.get(id(loop.params[position])), held.type, ctx)
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

    def rebuild(self, node: expr.Expr, scope: Scope) -> expr.Expr:
        """*node* of the backward with every forward value it reads replaced by its slot."""
        held = scope.rebuilt.get(id(node))
        if held is not None:
            return held
        if id(node) in scope.nodes:
            result = self.materialize(("node", node), scope)
        elif id(node) in self.placeholders:
            result = self.materialize(self.placeholders[id(node)], scope)
        elif id(node) in self.regions:
            result = self.rebuild_region(self.regions[id(node)], scope)
        elif id(node) in self.loops:
            result = self.rebuild_loop(self.loops[id(node)], scope)
        elif isinstance(node, expr.Call):
            result = replace(node, args=tuple(self.rebuild(arg, scope) for arg in node.args))
        elif isinstance(node, expr.Tuple):
            result = replace(node, elements=tuple(self.rebuild(item, scope) for item in node.elements))
        else:
            result = node
        scope.remember_rebuilt(node, result)
        return result

    def rebuild_region(self, backward: RegionBackward, scope: Scope) -> core.MeshRegion:
        node = backward.node
        inner = backward.scope
        body = self.rebuild(node.body, inner)
        return replace(
            node,
            body=body,
            params=(*node.params, *(param for param, _ in inner.region_params)),
            args=(*(self.rebuild(arg, scope) for arg in node.args), *(value for _, value in inner.region_params)),
        )

    def rebuild_loop(self, backward: RegionBackward, scope: Scope) -> core.LoopRegion:
        node = backward.node
        inner = backward.scope
        updates = tuple(self.rebuild(value, inner) for value in node.yield_values)
        return replace(
            node,
            body=updates[-1],
            yield_values=updates,
            params=(*node.params, *(param for param, _ in inner.region_params)),
            args=(*(self.rebuild(arg, scope) for arg in node.args), *(value for _, value in inner.region_params)),
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
        elif key[0] == "node" and id(value) in self.differ.recompute:
            result = replace(value, args=tuple(self.materialize(("node", arg), scope) for arg in value.args))
        elif scope.parent is None:
            result = projection(self.saved, len(self.saved_keys), self.key_type(key))
            self.saved_keys.append(key)
        elif isinstance(scope.region, core.LoopRegion):
            result = self.materialize_in_loop(key, scope)
        else:
            if key[0] == "node" and any(value is param for param in scope.region.params):
                index = next(i for i, param in enumerate(scope.region.params) if param is value)
                outer = self.materialize(("node", scope.region.args[index]), scope.parent)
            else:
                scope.extras.append(key)
                outer = self.materialize(("extra", scope, len(scope.extras) - 1), scope.parent)
            result = expr.Var(name=self.key_name(key)[0] or "saved", type=self.key_type(key))
            scope.region_params.append((result, outer))
        scope.remember_slot(slot, result)
        return result

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
        carried = len(loop.yield_values)
        if key[0] == "node" and any(value is param for param in loop.params[carried:]):
            position = next(i for i, param in enumerate(loop.params) if param is value)
            outer = self.materialize(("node", loop.args[position]), scope.parent)
            result = expr.Var(name=value.name, type=self.key_type(key))
            ctx.remember(result, result.type)
            scope.region_params.append((result, outer))
            return result
        held = self.key_type(key)
        owners = ctx.resolve_target().storage_owners
        scope.extras.append(key)
        outer = self.materialize(("extra", scope, len(scope.extras) - 1), scope.parent)
        stacks = expr.Var(
            name=f"stack_{self.key_name(key)[0] or 'saved'}",
            type=stacked(held, trip, owners, located(self.key_node(key))),
        )
        ctx.remember(stacks, stacks.type)
        scope.region_params.append((stacks, outer))
        return self.row(stacks, held, ordinal, ctx)

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
        return ctx.call("reshape", taken, new_shape=tuple(type_.shape))

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
            return self.callees[id(key[1])].saved
        _, scope, index = key
        inner = self.key_type(scope.extras[index])
        if isinstance(scope.region, core.LoopRegion):
            owners = scope.ctx.resolve_target().storage_owners
            return stacked(inner, trip_count(scope.region), owners, "")
        return inner

    def record(self, index: int, key: tuple) -> Saved:
        name, origin = self.key_name(key)
        return Saved(name or f"v{index}", self.key_type(key), origin)

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
            return f"{ref.qualified_name}.saved", f"call {ref.qualified_name}  {span_of(key[1], short=True) or ''}".rstrip()
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
        return expr.binding_name(node), f"{name}  {span_of(node, short=True) or ''}".rstrip()

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
            kept = tuple(projection(body, index, item) for index, item in enumerate(results))
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
        if not isinstance(node.type, types.TupleType):
            return projection(copied, 0, node.type)
        return pack(tuple(projection(copied, index, item) for index, item in enumerate(node.type.fields)))

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
                else tuple(projection(copied.body, index, item) for index, item in enumerate(results))
            )
        else:
            kept = (copied.body,)
        extras = tuple(self.saved_value(key, inner) for key in scope.extras)
        body = pack((*kept, *extras))
        region = replace(copied, body=body, type=body.type)
        self.regions[id(node)] = (region, len(results))
        if not isinstance(node.type, types.TupleType):
            return projection(region, 0, node.type)
        return pack(tuple(projection(region, index, item) for index, item in enumerate(results)))

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
            ordinal = build("sub", ordinal, integer(node.start, dtype))
        if node.step != 1:
            ordinal = build("floor_div", ordinal, integer(node.step, dtype))
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
                stack = expr.Var(name="stack", type=stack_type)
                stacks.append(stack)
                inits.append(build("zeros", type=stack_type))
                corner = pack((ordinal, *(integer(0, dtype) for _ in leaf.shape)))
                updates.append(build(
                    "insert_slice", stack, build("reshape", piece, new_shape=(1, *leaf.shape)), corner
                ))
        loop = replace(
            copied,
            params=(*copied.params[:carried], *stacks, *copied.params[carried:]),
            args=(*copied.args[:carried], *inits, *copied.args[carried:]),
            yield_values=(*copied.yield_values, *updates),
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
            return projection(self.calls[id(key[1])], self.owner.callees[id(key[1])].results, key[1].type)
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
