"""One admission check and tile projection for HIR and TIR calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import accumulate
from math import prod

from tilefoundry import services
from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types, utils
from tilefoundry.ir.types import int_tuple, layout_algebra


class RefusalError(ValueError):
    """An instruction's failed node and the bindings reached before that refusal."""

    def __init__(self, reason, *, path=(), op=None):
        self.reason = reason
        self.path = tuple(path.items()) if isinstance(path, dict) else tuple(path)
        atom = getattr(op, "atom", None)
        declaration = (
            atom if isinstance(atom, type) else type(atom) if atom is not None else type(op)
        )
        self.instruction = "" if op is None else expr.get_registered_name(declaration)
        super().__init__(reason)

    def __str__(self):
        return services.describe_refusal(self.instruction, self.path, self.reason)


class RelationContext:
    def __init__(self, type_list: dict[int, types.TensorType]) -> None:
        self.held = type_list

    def type_of(self, node: expr.Var) -> types.TensorType:
        return self.held[id(node)]


def operand_relations(op: expr.Op, operand_types: tuple[types.TensorType, ...]) -> tuple[access.Relation, ...]:
    """The registered access relations of *op* over operands of these types."""
    args = tuple(
        expr.Var(name=f"operand{index}", type=type_) for index, type_ in enumerate(operand_types)
    )
    call = expr.Call(target=op, args=args, type=types.UnitType())
    return access.relations_of(call, RelationContext(dict(zip(map(id, args), operand_types))))


def operand_pattern(param: expr.ParamDef, op: expr.Op, bindings: dict) -> pattern.Tensor | None:
    declared = param.pattern.read_on(op) if hasattr(param.pattern, "read_on") else param.pattern
    selected = pattern.selected(declared, bindings)
    return selected if isinstance(selected, pattern.Tensor) else None


def iteration_shape(relations: tuple[access.Relation, ...]) -> tuple[int, ...]:
    space = access.iteration_universe(relations)
    if space is None:
        raise ValueError("instruction access relations state no iteration space")
    return tuple(
        int(space.dim_max(axis).max_val().num_si()) + 1 for axis in range(space.tuple_dim())
    )


@dataclass(frozen=True)
class Projection:
    params: tuple
    writes: tuple
    patterns: tuple
    tiles: tuple
    rests: tuple
    repeat: tuple
    order: tuple
    shape: tuple
    logical_rests: tuple = ()
    bind_masks: tuple = ()
    axes: tuple = ()
    serial: tuple = ()


def call_params(op, count):
    schema = expr.schema_of(type(op))
    if schema is None:
        raise RefusalError(f"{type(op).__name__} is not a registered operation", op=op)
    params = tuple(p for p in schema.signature if p.kind == "input")
    required = sum(not p.optional for p in params)
    if not required <= count <= len(params):
        raise RefusalError(
            f"{type(op).__name__} expects operands {tuple(p.name for p in params)} ({required}..{len(params)}), got {count}",
            op=op,
        )
    optional = tuple(p for p in params if p.optional)
    included = {id(p) for p in optional[: count - required]}
    return tuple(p for p in params if not p.optional or id(p) in included)


def project(op, operand_types, mesh, *, bindings=None, repeat=None, order=None, fragments=True):
    operand_types = tuple(operand_types)
    params = call_params(op, len(operand_types))
    writes = tuple(param for param in params if param.effect and param.effect & expr.MemoryEffect.WRITE)
    bindings = (
        dict(getattr(getattr(op, "atom", None), "bindings", {}))
        if bindings is None
        else dict(bindings)
    )
    patterns = tuple(operand_pattern(param, op, bindings) for param in params)
    whole = {param.name: type_ for param, type_ in zip(params, operand_types, strict=True)}
    if (
        getattr(op, "atom", None) is None
        and fragments
        and repeat is None
        and order is None
        and not any(pattern.declared_shape(declared, bindings) is not None for declared in patterns)
    ):
        return Projection(params, writes, patterns, tuple(operand_types), (), (), (), ())
    try:
        whole_relations = operand_relations(op, tuple(whole[param.name] for param in params))
    except ValueError as error:
        raise RefusalError(str(error), path=bindings, op=op) from error
    whole_shape = iteration_shape(whole_relations)
    whole_types = tuple(whole[param.name] for param in params)
    shapes = tuple(
        pattern.declared_shape(declared, bindings) or tuple(type_.shape)
        for declared, type_ in zip(patterns, whole_types, strict=True)
    )
    provisional = tuple(
        types.TensorType(shape, type_.dtype, type_.layout, type_.storage)
        for shape, type_ in zip(shapes, whole_types, strict=True)
    )
    try:
        single = operand_relations(op, provisional)
    except ValueError as error:
        raise RefusalError(str(error), path=bindings, op=op) from error
    single_shape = iteration_shape(single)
    if len(whole_shape) != len(single_shape):
        raise RefusalError(
            "single-issue and scheduled iteration ranks differ", path=bindings, op=op
        )
    inferred = []
    for whole_extent, single_extent in zip(whole_shape, single_shape, strict=True):
        if single_extent < 1 or whole_extent % single_extent:
            raise RefusalError(
                f"iteration extent {whole_extent} is not divisible by "
                f"single-issue extent {single_extent}",
                path=bindings,
                op=op,
            )
        inferred.append(whole_extent // single_extent)
    inferred = tuple(inferred)
    repeat = inferred if repeat is None else repeat
    fixed = any(pattern.declared_shape(declared, bindings) is not None for declared in patterns)
    if not fixed and any(count != 1 for count in repeat):
        raise RefusalError("transfer tiling is not yet supported", path=bindings, op=op)
    if repeat != inferred:
        raise RefusalError(
            f"repeat {repeat} conflicts with inferred repeat {inferred}", path=bindings, op=op
        )
    order = tuple(range(len(repeat))) if order is None else order
    if sorted(order) != list(range(len(repeat))):
        raise RefusalError(
            f"order {order} is not a permutation of iteration dimensions", path=bindings, op=op
        )
    declaration = getattr(op, "atom", op)
    participant = (
        declaration.execution_mesh
        if isinstance(declaration, type)
        else getattr(declaration, "required_execution_mesh", getattr(declaration, "execution_mesh", None))
    )
    if not isinstance(participant, types.Mesh):
        return Projection(params, writes, patterns, whole_types, (), repeat, order, single_shape)
    if not fragments:
        inner_types = tuple(
            types.TensorType(shape, type_.dtype, type_.layout, type_.storage)
            for shape, type_ in zip(shapes, whole_types, strict=True)
        )
        return Projection(params, writes, patterns, inner_types, (), repeat, order, single_shape)
    tiles, rests, axes, logical_rests = [], [], [], []
    try:
        for param, type_, shape, declared, boundary in zip(
            params, whole_types, shapes, patterns, single[: len(params)], strict=True
        ):
            name = param.name
            top = type_.layout if isinstance(type_.layout, types.ComposedLayout) else None
            outer = top.outer if top is not None else type_.layout
            if isinstance(outer, types.ShardLayout):
                outer = utils.physical_layout(outer)
            shard = outer if isinstance(outer, types.ShardLayout) else None
            address = shard.layout if shard is not None else outer
            primitive = address.outer if isinstance(address, types.ComposedLayout) else address
            if not isinstance(primitive, types.Layout):
                raise ValueError(f"{name}: cannot divide {type(primitive).__name__}")
            profile = int_tuple.compatible_profile(primitive.shape, type_.shape)
            grouped = types.unflatten(types.flatten(primitive), profile)
            split = (
                {attr.axis for attr in shard.attrs if isinstance(attr, types.Split)} if shard else set()
            )
            try:
                tiler = utils.axis_tiler(profile, shape, splits=split)
            except ValueError as error:
                first = 0
                for axis, (mode, size) in enumerate(zip(profile, shape, strict=True)):
                    widths = types.flatten(mode)
                    count = prod(widths) // size
                    for index, prefix in enumerate(accumulate(widths, lambda x, y: x * y)):
                        if count % prefix and prefix % count:
                            raise ValueError(
                                f"{name} axis {axis}: repeat is not leftmost at mode {first + index} "
                                f"(extent {widths[index]})"
                            ) from error
                    first += len(widths)
                raise ValueError(f"{name}: {error}") from error
            compact = types.Layout(
                profile,
                tuple(
                    types.compact_row_major(mode) if isinstance(mode, tuple) else 1 for mode in profile
                ),
            )
            logical_rest = types.get(layout_algebra.zipped_divide(compact, tiler, major="row"), 1)
            divided = layout_algebra.zipped_divide(grouped, tiler, major="row")
            tile, rest = types.get(divided, 0), types.get(divided, 1)
            flat, remainder = types.flatten(tile), types.flatten(rest)
            flat = replace(
                flat,
                strides=tuple(
                    original if extent == 1 else stride
                    for extent, original, stride in zip(
                        types.flatten(primitive.shape),
                        types.flatten(primitive.strides),
                        flat.strides,
                        strict=True,
                    )
                ),
            )
            layout_axes = int_tuple.mode_axes(profile)
            kept = tuple(
                i
                for i, (before, after) in enumerate(
                    zip(types.flatten(primitive.shape), flat.shape, strict=True)
                )
                if before == 1 or after != 1 or layout_axes.count(layout_axes[i]) == 1
            )
            mapping = {old: new for new, old in enumerate(kept)}
            if any(isinstance(mode, tuple) for mode in primitive.shape):
                parts = tuple(
                    (
                        tuple(flat.shape[i] for i in kept if layout_axes[i] == axis),
                        tuple(flat.strides[i] for i in kept if layout_axes[i] == axis),
                    )
                    for axis in range(len(shape))
                )
                tile = types.Layout(
                    tuple(s[0] if len(s) == 1 else s for s, _ in parts),
                    tuple(d[0] if len(d) == 1 else d for _, d in parts),
                )
            else:
                tile = types.Layout(
                    tuple(flat.shape[i] for i in kept), tuple(flat.strides[i] for i in kept)
                )
            broadcast = shard is None or not any(
                isinstance(attr, (types.Split, types.Partial)) for attr in shard.attrs
            )
            if isinstance(address, types.ComposedLayout):
                tile = replace(address, outer=tile, offset=0 if broadcast else address.offset)
            if top is not None and top.inner is not None:
                tile = types.ComposedLayout(top.inner, 0, tile)
            if shard is not None:
                tile_attrs = tuple(
                    types.Split(mapping[attr.axis])
                    if isinstance(attr, types.Split)
                    and attr.axis in mapping
                    and remainder.shape[attr.axis] == 1
                    else types.Broadcast()
                    if isinstance(attr, types.Split)
                    else attr
                    for attr in shard.attrs
                )
                source = utils.participant_layout(replace(shard, attrs=tile_attrs), participant)
                frame, dropped = utils.participant_frame(source.mesh, participant)
                tile = types.ShardLayout(tile, source.attrs[dropped:], frame)
                rest = types.ShardLayout(
                    rest,
                    tuple(
                        types.Split(attr.axis)
                        if isinstance(attr, types.Split) and remainder.shape[attr.axis] != 1
                        else types.Broadcast()
                        for attr in shard.attrs
                    ),
                    shard.mesh,
                )
            tile = types.TensorType(shape, type_.dtype, tile, type_.storage)
            mapped = access.projected_axes(boundary)
            physical = rest.layout if isinstance(rest, types.ShardLayout) else rest
            profile = int_tuple.compatible_profile(
                physical.shape, tuple(1 if axis is None else repeat[axis] for axis in mapped)
            )
            regrouped = types.unflatten(types.flatten(physical), profile)
            rests.append(
                replace(rest, layout=regrouped) if isinstance(rest, types.ShardLayout) else regrouped
            )
            logical_rests.append(types.unflatten(types.flatten(logical_rest), regrouped.shape))
            tiles.append(tile)
            axes.append(mapped)
        shared = {}
        for param, rest, mapped in zip(params, rests, axes, strict=True):
            if not isinstance(rest, types.ShardLayout):
                continue
            flat = types.flatten(rest.layout)
            logical_axes = int_tuple.mode_axes(rest.shape)
            for axis in set(logical_axes):
                iteration_axis = mapped[axis]
                if iteration_axis is None:
                    continue
                modes = tuple(
                    i for i, a in enumerate(logical_axes) if a == axis and flat.shape[i] != 1
                )
                splits = tuple(
                    (modes.index(attr.axis), flat.shape[attr.axis])
                    for attr in rest.attrs
                    if isinstance(attr, types.Split) and attr.axis in modes
                )
                if not splits:
                    continue
                if iteration_axis in shared and shared[iteration_axis][0] != splits:
                    raise ValueError(
                        f"{param.name}: repeat mode on iteration axis {iteration_axis} has conflicting Split extents"
                    )
                shared[iteration_axis] = (
                    splits,
                    tuple(flat.shape[i] for i in modes),
                    rest.mesh,
                    tuple(
                        (mesh_axis, modes.index(attr.axis))
                        for mesh_axis, attr in enumerate(rest.attrs)
                        if isinstance(attr, types.Split) and attr.axis in modes
                    ),
                )
        for param, rest, mapped in zip(params, rests, axes, strict=True):
            physical = rest.layout if isinstance(rest, types.ShardLayout) else rest
            for axis, iteration_axis in enumerate(mapped):
                if iteration_axis not in shared:
                    continue
                mode = types.get(physical, axis)
                active = tuple(width for width in types.flatten(mode.shape) if width != 1)
                expected = shared[iteration_axis][1]
                try:
                    int_tuple.compatible_profile(active, expected)
                except ValueError as error:
                    raise ValueError(
                        f"{param.name} iteration axis {iteration_axis}: repeat mode {active} cannot factor as {expected}"
                    ) from error
        for index, (rest, mapped) in enumerate(zip(rests, axes, strict=True)):
            physical = rest.layout if isinstance(rest, types.ShardLayout) else rest
            bindings_on_axes = [
                (axis, shared[iteration_axis])
                for axis, iteration_axis in enumerate(mapped)
                if iteration_axis in shared
            ]
            if not bindings_on_axes:
                continue
            rest_mesh = rest.mesh if isinstance(rest, types.ShardLayout) else bindings_on_axes[0][1][2]
            rest_attrs = (
                list(rest.attrs)
                if isinstance(rest, types.ShardLayout)
                else [types.Broadcast()] * len(types.flatten(rest_mesh.layout.shape))
            )
            logical_axes = int_tuple.mode_axes(physical.shape)
            flat = types.flatten(physical)
            for axis, (_, _, source_mesh, bound) in bindings_on_axes:
                modes = [i for i, a in enumerate(logical_axes) if a == axis and flat.shape[i] != 1]
                shift = len(rest_attrs) - len(types.flatten(source_mesh.layout.shape))
                for mesh_axis, mode in bound:
                    rest_attrs[mesh_axis + shift] = types.Split(modes[mode])
            rests[index] = types.ShardLayout(physical, tuple(rest_attrs), rest_mesh)
    except ValueError as error:
        raise RefusalError(str(error), path=bindings, op=op) from error
    bind_masks = []
    for rest in rests:
        physical = rest.layout if isinstance(rest, types.ShardLayout) else rest
        bound = (
            {attr.axis for attr in rest.attrs if isinstance(attr, types.Split)}
            if isinstance(rest, types.ShardLayout)
            else set()
        )
        bind_masks.append(
            int_tuple.unflatten(
                tuple(index in bound for index in range(len(types.flatten(physical.shape)))),
                physical.shape,
            )
        )
    serial = tuple(
        count // prod(width for _, width in shared[axis][0]) if axis in shared else count
        for axis, count in enumerate(repeat)
    )
    return Projection(
        params,
        writes,
        patterns,
        tuple(tiles),
        tuple(rests),
        repeat,
        order,
        single_shape,
        tuple(logical_rests),
        tuple(bind_masks),
        tuple(axes),
        serial,
    )


def pattern_refusal(matcher: pattern.Matcher, op, parameter=None) -> RefusalError:
    refusal = matcher.refusal
    path = tuple((refusal.bindings if refusal is not None else matcher.bindings).items())
    return RefusalError(services.refusal_reason(refusal, parameter), path=path, op=op)


def selects(op, capability) -> bool:
    """Whether *op*'s attribute, as a class or an instance, names *capability*'s declaration."""
    value = getattr(op, capability.attribute)
    declaration = value if isinstance(value, type) else type(value)
    return issubclass(declaration, capability.declaration)


def admit_capabilities(op, target) -> None:
    """Refuse *op* if the target's architecture does not list a capability the op uses."""
    if target is None:
        return
    listed = expr.target_capabilities(target)
    for capability in expr.capabilities_of(type(op)):
        if capability.name is None or capability.name in listed:
            continue
        if capability.attribute is not None and not selects(op, capability):
            continue
        raise RefusalError(
            f"needs {capability.name}, which {target.identity} does not list", op=op
        )


def admit_fields(op, operands, mesh, matcher, *, target=None):
    """Admit attributes, execution mesh and operand fields before tile projection."""
    matcher.bindings.update(getattr(op, "bindings", {}))
    admit_capabilities(op, target)
    for param in expr.schema_of(type(op)).signature:
        if param.kind != "attribute" or param.name == "execution_mesh":
            continue
        value = getattr(op, param.name)
        if param.name == "atom" and isinstance(value, type) and value in param.pattern.declarations:
            continue
        if value is None and param.optional:
            continue
        if (
            isinstance(param.annotation, type)
            and issubclass(param.annotation, Enum)
            and not isinstance(value, param.annotation)
        ):
            raise RefusalError(
                f"{param.name} must be {param.annotation.__name__}, got {type(value).__name__}",
                op=op,
            )
        if param.annotation is int and not isinstance(value, int):
            raise RefusalError(f"{param.name} must be int, got {type(value).__name__}", op=op)
        if param.pattern is not None and not matcher.match(param.pattern, value):
            raise pattern_refusal(matcher, op, param.name)
    if not matcher.match(pattern.declared_execution_mesh(op, target), mesh):
        raise pattern_refusal(matcher, op, "mesh")
    for param, type_ in zip(call_params(op, len(operands)), operands, strict=True):
        declared = operand_pattern(param, op, matcher.bindings)
        if isinstance(declared, pattern.Tensor):
            declared = replace(declared, shape=None, layout=None)
        if declared is not None and not matcher.match(declared, type_):
            raise pattern_refusal(matcher, op, param.name)


def check_call(op, operands, mesh, *, frame, repeat=None, order=None, projection=None):
    """Match whole fields, project atom tiles, then check one closed declaration.

    Operands may be types for candidate admission or typed expressions for
    verification; expressions retain literal values needed by leaf verifiers.
    """
    params = call_params(op, len(operands))
    if not access.relation_registry.has(type(op)):
        raise RefusalError(
            f"{type(op).__name__} states no access relations; register one with register_access_relation",
            op=op,
        )
    handler = verify.registry.get(type(op), None)
    if (
        handler is None
        and not any(param.pattern is not None for param in params)
        and not pattern.between_rules(type(op))
    ):
        raise RefusalError(
            f"{type(op).__name__} has no input pattern, between relation, or registered verifier",
            op=op,
        )
    whole = tuple(value.type if isinstance(value, expr.Expr) else value for value in operands)
    atom = getattr(op, "atom", None)
    matcher = pattern.Matcher(
        dict(getattr(atom, "bindings", {})),
        mesh=mesh,
        owners=frame.target.storage_owners
        if frame is not None and frame.target is not None
        else expr.StorageOwners((), {}),
    )
    admit_fields(op, whole, mesh, matcher, target=None if frame is None else frame.target)
    try:
        tiles = (
            projection
            if projection is not None
            else project(op, whole, mesh, bindings=matcher.bindings, repeat=repeat, order=order)
        )
    except RefusalError:
        raise
    except ValueError as error:
        raise RefusalError(str(error), path=matcher.bindings, op=op) from error
    try:
        typeinfer.check_input_reachability(op, whole, mesh)
    except ValueError as error:
        raise RefusalError(str(error), path=matcher.bindings, op=op) from error
    declaration = getattr(op, "atom", op)
    participant = getattr(
        declaration, "required_execution_mesh", getattr(declaration, "execution_mesh", None)
    )
    needs_frame = any(
        isinstance(declared, pattern.Tensor)
        and isinstance(declared.layout, pattern.ShardLayout)
        and types.shard_layout_of(type_.layout) is None
        for declared, type_ in zip(tiles.patterns, tiles.tiles, strict=True)
    )
    if needs_frame and isinstance(participant, types.Mesh) and mesh is not None:
        selected = utils.participant_layout(
            types.ShardLayout(
                types.Layout(()),
                tuple(types.Broadcast() for _ in types.flatten(mesh.layout.shape)),
                mesh,
            ),
            participant,
        ).mesh
        try:
            frame_mesh, _ = utils.participant_frame(selected, participant)
        except ValueError as error:
            raise RefusalError(
                f"scope {mesh!r} does not contain participant frame {participant!r}",
                path=matcher.bindings,
                op=op,
            ) from error
        matcher.use_mesh(frame_mesh)
    view = getattr(atom, "match_view", None)
    for param, declared, type_ in zip(params, tiles.patterns, tiles.tiles, strict=True):
        match_type = type_ if view is None else view(param.name, type_)
        if declared is not None and not matcher.match(declared, match_type):
            refusal = pattern_refusal(matcher, op, param.name)
            if (
                isinstance(declared, pattern.Tensor)
                and isinstance(declared.layout, pattern.ShardLayout)
                and isinstance(declared.layout.attrs, tuple)
                and matcher.mesh is not None
            ):
                states = utils.attrs_by_axis(match_type, matcher.mesh, matcher.owners)
                for axis, (required, (key, held)) in enumerate(
                    zip(declared.layout.attrs, states.items())
                ):
                    level = key[0]
                    if required == types.Broadcast() and isinstance(held, types.Varying):
                        refusal = RefusalError(
                            f"{refusal.reason}; mesh axis {axis} ({level}) is Varying; "
                            f"reshard({param.name}, @ {{B()}}) to declare equal values",
                            path=refusal.path,
                            op=op,
                        )
                        break
            raise refusal
    named = dict(zip((p.name for p in params), tiles.tiles, strict=True))
    for rule in pattern.between_rules(type(op)):
        if not rule.holds(named):
            raise RefusalError(rule.refused(named), path=matcher.bindings, op=op)
    if matcher.solve() is not True:
        raise pattern_refusal(matcher, op)
    if handler is not None:
        args = tuple(
            value if isinstance(value, expr.Constant) else expr.Var(param.name, type=type_)
            for value, param, type_ in zip(operands, params, tiles.tiles, strict=True)
        )
        call = expr.Call(op, args, type=types.UnitType())
        ctx = (
            replace(
                frame,
                memo={
                    id(arg): (arg, type_) for arg, type_ in zip(args, tiles.tiles, strict=True)
                },
                current_mesh=mesh,
            )
            if isinstance(frame, verify.Context)
            else verify.Context(
                module=None if frame is None else frame.module, function=None if frame is None else frame.function, target=None if frame is None else frame.target, topologies=None if frame is None else frame.topologies, current_mesh=mesh, mesh_scope=() if mesh is None else (mesh,)
            )
        )
        try:
            handler(call, ctx)
        except expr.VerifyError as error:
            raise RefusalError(str(error), path=matcher.bindings, op=op) from error
    return dict(matcher.bindings), tiles


def verify_call(op, args, mesh, *, frame):
    """Verify one scheduled TIR call; a schedule refusal becomes a ``VerifyError``."""
    try:
        check_call(op, args, mesh, frame=frame)
    except RefusalError as error:
        raise expr.VerifyError(str(error)) from error


__all__ = [
    "Projection",
    "RefusalError",
    "admit_fields",
    "call_params",
    "check_call",
    "operand_relations",
    "project",
]

services.registry.register(services.verify_call, verify_call)

