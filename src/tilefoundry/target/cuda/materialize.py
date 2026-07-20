"""Private reconstruction of one solved CTA planning graph."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.metadata import remove_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.shard import (
    Broadcast,
    Layout,
    Mesh,
    ShardLayout,
    Topology,
    try_c_order_strides,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule.constraints import ScheduleConstraintMetadata
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .planner import OpCandidate, PlanningProblem
from .solver import PlanningSolution


def _tensor_leaves(type_value: Type, path: tuple[int, ...] = ()) -> tuple[tuple[tuple[int, ...], TensorType], ...]:
    if isinstance(type_value, TensorType):
        return ((path, type_value),)
    if isinstance(type_value, TupleType):
        return tuple(
            leaf
            for index, field in enumerate(type_value.fields)
            for leaf in _tensor_leaves(field, path + (index,))
        )
    return ()


def _clean(expr: Expr) -> Expr:
    return remove_metadata(expr, ScheduleConstraintMetadata)


def _clean_metadata(expr: Expr) -> tuple:
    return tuple(
        value for value in expr.metadata if type(value) is not ScheduleConstraintMetadata
    )


@dataclass
class _GraphIndex:
    problem: PlanningProblem
    expr_refs: dict[tuple[tuple[int, ...], int], tuple[int, ...]]
    scope_exprs: dict[tuple[tuple[int, ...], int], Expr]
    param_refs: dict[tuple[tuple[int, ...], int], tuple[int, ...]]
    call_paths: dict[tuple[tuple[int, ...], int], tuple[int, ...]]
    region_ids: dict[int, int]
    _next_instance: int = 1

    @classmethod
    def build(cls, problem: PlanningProblem) -> "_GraphIndex":
        index = cls(problem, {}, {}, {}, {}, {
            id(region.source): region_id for region_id, region in problem.regions.items()
        })
        root_env: dict[int, tuple[int, ...]] = {}
        for param in problem.root.params:
            refs = index._source_refs(param, ())
            root_env[id(param)] = refs
            index.param_refs[((), id(param))] = refs
            index._record((), param, refs)
        index._walk_expr(problem.root.body, (), root_env)
        expected = tuple(path for path, _ in problem.function_instances)
        actual = [()]
        actual.extend(sorted(index.call_paths.values(), key=lambda path: path))
        if set(actual) != set(expected):
            raise RuntimeError(
                "P4 materialization: source traversal did not reproduce P2 function paths"
            )
        return index

    def _source_refs(self, source: Expr, path: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(value_id for value_id, _value in sorted(
            (
                (value_id, value)
                for value_id, value in self.problem.values.items()
                if value.source is source and value.function_path == path
            ),
            key=lambda item: item[1].leaf_path,
        ))

    def _record(self, path: tuple[int, ...], expr: Expr, refs: tuple[int, ...]) -> None:
        self.expr_refs[(path, id(expr))] = refs
        for value_id in refs:
            self.scope_exprs.setdefault((path, value_id), expr)

    def _region_refs(self, region: GridRegionExpr, path: tuple[int, ...], env):
        region_id = self.region_ids.get(id(region))
        if region_id is None:
            raise RuntimeError("P4 materialization: unknown GridRegion source")
        info = self.problem.regions[region_id]
        for value in region.init_args:
            self._walk_expr(value, path, env)
        carry_refs = tuple(
            tuple(value_id for value_id, _leaf_path in sorted(
                (
                    (value_id, value.leaf_path)
                    for value_id, value in self.problem.values.items()
                    if value.source is phi
                    and value.function_path == path + (region_id,)
                    and value.role == "carry"
                ),
                key=lambda item: item[1],
            ))
            for phi in region.carried_args
        )
        body_path = path + (region_id,)
        body_env = dict(env)
        for phi, refs in zip(region.carried_args, carry_refs):
            body_env[id(phi)] = refs
            self.param_refs[(body_path, id(phi))] = refs
            self._record(body_path, phi, refs)
        self._walk_expr(region.body, body_path, body_env)
        for value in region.yield_values:
            self._walk_expr(value, body_path, body_env)
        result_refs = info.result_value_ids if region.carried_args else self.expr_refs[(body_path, id(region.body))]
        self._record(path, region, tuple(result_refs))
        return tuple(result_refs)

    def _walk_expr(self, expr: Expr | None, path: tuple[int, ...], env) -> tuple[int, ...]:
        if expr is None:
            return ()
        cached = self.expr_refs.get((path, id(expr)))
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            refs = env.get(id(expr), self._source_refs(expr, path))
            self._record(path, expr, refs)
            return refs
        if isinstance(expr, Constant):
            self._record(path, expr, ())
            return ()
        if isinstance(expr, Tuple):
            refs = tuple(ref for value in expr.elements for ref in self._walk_expr(value, path, env))
            self._record(path, expr, refs)
            return refs
        if isinstance(expr, GridRegionExpr):
            return self._region_refs(expr, path, env)
        if not isinstance(expr, Call):
            self._record(path, expr, ())
            return ()
        arg_refs = tuple(self._walk_expr(arg, path, env) for arg in expr.args)
        if isinstance(expr.target, Function):
            child_path = path + (self._next_instance,)
            self._next_instance += 1
            self.call_paths[(path, id(expr))] = child_path
            child_env = dict(zip((id(param) for param in expr.target.params), arg_refs))
            for param, refs in zip(expr.target.params, arg_refs):
                self.param_refs[(child_path, id(param))] = refs
                self._record(child_path, param, refs)
            refs = self._walk_expr(expr.target.body, child_path, child_env)
            self._record(path, expr, refs)
            return refs
        if isinstance(expr.target, TupleGetItem):
            source_refs = arg_refs[0] if arg_refs else ()
            fields = _tensor_leaves(expr.args[0].type)
            index = expr.target.index
            if index < 0 or index >= len(expr.args[0].type.fields):  # type: ignore[union-attr]
                raise RuntimeError("P4 materialization: invalid TupleGetItem source")
            start = sum(
                len(_tensor_leaves(field))
                for field in expr.args[0].type.fields[:index]  # type: ignore[union-attr]
            )
            refs = source_refs[start:start + len(_tensor_leaves(fields[index][1]))]
            self._record(path, expr, refs)
            return refs
        refs = self._source_refs(expr, path)
        if len(refs) != len(_tensor_leaves(expr.type)):
            raise RuntimeError(
                f"P4 materialization: missing source Value map for {type(expr.target).__name__}"
            )
        self._record(path, expr, refs)
        return refs


@dataclass
class _Instance:
    path: tuple[int, ...]
    source: Function
    params: tuple[Var, ...]
    env: dict[int, Expr]
    clone: Function | None = None
    expr_cache: dict[int, Expr] | None = None

    def __post_init__(self) -> None:
        self.expr_cache = {}


class _Materializer:
    def __init__(self, problem: PlanningProblem, solution: PlanningSolution) -> None:
        self.problem = problem
        self.solution = solution
        self.index = _GraphIndex.build(problem)
        self.selected_candidates = set(solution.selected_candidate_ids)
        self.selected_buckets = set(solution.selected_bucket_ids)
        self.candidate_by_site: dict[int, int] = {}
        self.producer_by_bucket: dict[int, int] = {}
        self.bucket_exprs: dict[tuple[tuple[int, ...], int], Expr] = {}
        self.broadcast_exprs: dict[tuple[tuple[int, ...], int, object], Expr] = {}
        self.instances: dict[tuple[int, ...], _Instance] = {}
        self._validate_solution()

    def _fail(self, message: str) -> RuntimeError:
        return RuntimeError(f"P4 materialization: {message}")

    def _validate_solution(self) -> None:
        if len(self.selected_candidates) != len(self.solution.selected_candidate_ids):
            raise self._fail("solution contains duplicate candidate IDs")
        if len(self.selected_buckets) != len(self.solution.selected_bucket_ids):
            raise self._fail("solution contains duplicate bucket IDs")
        if any(candidate_id not in self.problem.candidates for candidate_id in self.selected_candidates):
            raise self._fail("solution selects an unknown candidate")
        if any(bucket_id not in self.problem.buckets for bucket_id in self.selected_buckets):
            raise self._fail("solution selects an unknown bucket")
        for site_id, candidates in self.problem.authored_candidates.items():
            selected = tuple(candidate_id for candidate_id in candidates if candidate_id in self.selected_candidates)
            if len(selected) != 1:
                raise self._fail(f"site {site_id} does not have exactly one selected candidate")
            self.candidate_by_site[site_id] = selected[0]
        for candidate_id in self.selected_candidates:
            candidate = self.problem.candidates[candidate_id]
            for bucket_id in (*candidate.input_bucket_ids, *candidate.output_bucket_ids):
                if bucket_id not in self.selected_buckets:
                    raise self._fail(
                        f"candidate {candidate_id} references unselected bucket {bucket_id}"
                    )
            for bucket_id in candidate.output_bucket_ids:
                previous = self.producer_by_bucket.setdefault(bucket_id, candidate_id)
                if previous != candidate_id:
                    raise self._fail(f"bucket {bucket_id} has multiple selected producers")
        for bucket_id in self.selected_buckets:
            bucket = self.problem.buckets[bucket_id]
            if bucket.is_source:
                if bucket_id in self.producer_by_bucket:
                    raise self._fail(f"source bucket {bucket_id} also has a selected producer")
            elif bucket_id not in self.producer_by_bucket:
                role = self.problem.values[bucket.value_id].role
                if role not in {"carry", "result"}:
                    raise self._fail(f"non-source bucket {bucket_id} has no selected producer")
        offsets = dict(self.solution.bucket_offsets)
        if len(offsets) != len(self.solution.bucket_offsets):
            raise self._fail("solution contains duplicate bucket offsets")
        if set(offsets) != self.selected_buckets:
            raise self._fail("solution bucket offsets do not match selected buckets")
        for bucket_id, offset in offsets.items():
            if not isinstance(offset, int) or offset < 0:
                raise self._fail(f"bucket {bucket_id} has an invalid CTA offset")
            count = self._bucket_count(bucket_id)
            if offset + count > self.problem.topology.size:
                raise self._fail(f"bucket {bucket_id} placement exceeds the root CTA topology")
            fixed = self.problem.buckets[bucket_id].fixed_offset
            if fixed is not None and fixed != offset:
                raise self._fail(f"bucket {bucket_id} violates its fixed CTA offset")

    def _bucket_count(self, bucket_id: int) -> int:
        type_value = self.problem.types[self.problem.buckets[bucket_id].type_id]
        if not isinstance(type_value, TensorType) or not isinstance(type_value.layout, ShardLayout):
            return 1
        shape = type_value.layout.mesh.layout.shape
        if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0:
            raise self._fail(f"bucket {bucket_id} does not have a rank-one static CTA count")
        return shape[0]

    def _placed_tensor_type(
        self, bucket_id: int, template: TensorType | None = None
    ) -> TensorType:
        original = self.problem.types[self.problem.buckets[bucket_id].type_id]
        if not isinstance(original, TensorType):
            raise self._fail(f"bucket {bucket_id} is not a tensor bucket")
        if template is None:
            template = original
        offset = dict(self.solution.bucket_offsets)[bucket_id]
        count = self._bucket_count(bucket_id)
        full = Mesh(
            Topology(self.problem.topology.name, self.problem.topology.size),
            Layout((self.problem.topology.size,), (1,)),
        )
        placed_mesh = full[offset:offset + count]
        if isinstance(original.layout, ShardLayout):
            layout = replace(original.layout, mesh=placed_mesh)
        elif (
            isinstance(template.layout, ShardLayout)
            and self._layout_fits_count(template.layout, count)
        ):
            layout = replace(template.layout, mesh=placed_mesh)
        else:
            layout = ShardLayout(
                layout=Layout(template.shape, try_c_order_strides(template.shape)),
                attrs=(Broadcast(),),
                mesh=placed_mesh,
            )
        return replace(template, layout=layout, storage=StorageKind.GMEM)

    @staticmethod
    def _layout_fits_count(layout: ShardLayout, count: int) -> bool:
        for attr in layout.attrs:
            if not hasattr(attr, "axis"):
                continue
            axis = attr.axis
            shape = layout.layout.shape
            if axis >= len(shape) or not isinstance(shape[axis], int):
                return False
            if shape[axis] % count:
                return False
        return True

    def _placed_value_type(
        self, value_id: int, template: TensorType | None = None
    ) -> Type:
        return self._placed_tensor_type(self._selected_bucket_for_value(value_id), template)

    def _selected_bucket_for_value(self, value_id: int) -> int:
        matches = tuple(
            bucket_id
            for bucket_id in self.selected_buckets
            if self.problem.buckets[bucket_id].value_id == value_id
        )
        if len(matches) != 1:
            raise self._fail(f"value {value_id} does not have one selected bucket")
        return matches[0]

    def _materialize_type(
        self,
        type_value: Type,
        value_ids: tuple[int, ...] = (),
        bucket_ids: tuple[int, ...] | None = None,
    ) -> Type:
        iterator = iter(bucket_ids if bucket_ids is not None else value_ids)

        def visit(current: Type) -> Type:
            if isinstance(current, TensorType):
                try:
                    selected = next(iterator)
                    if bucket_ids is None:
                        return self._placed_value_type(selected, current)
                    return self._placed_tensor_type(selected, current)
                except StopIteration as exc:
                    raise self._fail("type leaf count does not match the planning graph") from exc
            if isinstance(current, TupleType):
                return TupleType(tuple(visit(field) for field in current.fields))
            return current

        result = visit(type_value)
        try:
            next(iterator)
        except StopIteration:
            return result
        raise self._fail("planning graph has extra tensor leaves")

    def _infer(self, call: Call) -> Call:
        try:
            inferred = TypeInferContext(module=self.problem.module).type_of(call)
        except Exception as exc:
            raise self._fail(
                f"type inference failed while rebuilding {type(call.target).__name__}: {exc}"
            ) from exc
        return replace(call, type=inferred)

    def _materialize_result(
        self, value: Expr, bucket_ids: tuple[int, ...], context: str
    ) -> Expr:
        expected = self._materialize_type(value.type, bucket_ids=bucket_ids)
        if value.type == expected:
            return value
        if isinstance(value.type, TupleType) and isinstance(expected, TupleType):
            elements: list[Expr] = []
            offset = 0
            for index, field_type in enumerate(value.type.fields):
                field_count = len(_tensor_leaves(field_type))
                field_ids = bucket_ids[offset:offset + field_count]
                offset += field_count
                if isinstance(value, Tuple):
                    field = value.elements[index]
                else:
                    field = self._infer(
                        Call(
                            type=field_type,
                            target=TupleGetItem(index=index),
                            args=(value,),
                            loc=getattr(value, "loc", None),
                        )
                    )
                elements.append(self._materialize_result(field, field_ids, context))
            if offset != len(bucket_ids):
                raise self._fail(f"{context} tuple leaf count does not match selected buckets")
            return Tuple(
                elements=tuple(elements),
                type=TupleType(tuple(element.type for element in elements)),
                metadata=_clean_metadata(value) if isinstance(value, Tuple) else (),
            )
        if (
            isinstance(value.type, TensorType)
            and isinstance(expected, TensorType)
            and value.type.shape == expected.shape
            and value.type.dtype == expected.dtype
            and value.type.storage == expected.storage
            and isinstance(expected.layout, ShardLayout)
        ):
            converted = self._infer(
                Call(
                    type=expected,
                    target=Reshard(layout=expected.layout, storage=expected.storage),
                    args=(value,),
                    loc=getattr(value, "loc", None),
                )
            )
            if converted.type == expected:
                return converted
        raise self._fail(
            f"{context} type mismatch: got {value.type!r}, expected {expected!r}"
        )

    def _coerce(self, value: Expr, bucket_id: int) -> Expr:
        template = value.type if isinstance(value.type, TensorType) else None
        expected = self._placed_tensor_type(bucket_id, template)
        if value.type == expected:
            return value
        if not isinstance(value.type, TensorType):
            raise self._fail(f"bucket {bucket_id} expects a tensor value")
        if (
            value.type.shape != expected.shape
            or value.type.dtype != expected.dtype
            or value.type.storage != expected.storage
        ):
            raise self._fail(
                f"bucket {bucket_id} type mismatch: got {value.type!r}, expected {expected!r}"
            )
        producer_id = self.producer_by_bucket.get(bucket_id)
        if producer_id is not None and not isinstance(self.problem.candidates[producer_id].op, Reshard):
            raise self._fail(
                f"bucket {bucket_id} requires an unselected conversion for {value.type!r}"
            )
        call = Call(
            type=expected,
            target=Reshard(layout=expected.layout, storage=expected.storage),
            args=(value,),
            loc=getattr(value, "loc", None),
        )
        return self._infer(call)

    def _scope_expr(self, path: tuple[int, ...], value_id: int) -> Expr:
        expr = self.index.scope_exprs.get((path, value_id))
        if expr is not None:
            return expr
        owner = self.problem.values[value_id].function_path
        expr = self.index.scope_exprs.get((owner, value_id))
        if expr is None:
            raise self._fail(f"value {value_id} has no source expression in scope {path}")
        return expr

    def _selected_bucket_expr(
        self, bucket_id: int, path: tuple[int, ...], base: Expr | None = None
    ) -> Expr:
        key = (path, bucket_id)
        cached = self.bucket_exprs.get(key)
        if cached is not None and base is None:
            return cached
        if bucket_id not in self.selected_buckets:
            raise self._fail(f"bucket {bucket_id} is not selected")
        if base is None:
            base = self._rebuild_expr(self._scope_expr(path, self.problem.buckets[bucket_id].value_id), path)
        producer_id = self.producer_by_bucket.get(bucket_id)
        if producer_id is None:
            result = self._coerce(base, bucket_id)
        else:
            candidate = self.problem.candidates[producer_id]
            if candidate.site_id is None:
                if len(candidate.input_bucket_ids) != 1 or len(candidate.output_bucket_ids) != 1:
                    raise self._fail(f"synthesized candidate {producer_id} is not unary")
                source = self._selected_bucket_expr(candidate.input_bucket_ids[0], path)
                target = self._placed_tensor_type(candidate.output_bucket_ids[0])
                call = Call(
                    type=target,
                    target=Reshard(layout=target.layout, storage=target.storage),
                    args=(source,),
                    loc=getattr(source, "loc", None),
                )
                result = self._infer(call)
            else:
                result = self._coerce(base, bucket_id)
        if base is None:
            self.bucket_exprs[key] = result
        elif key not in self.bucket_exprs:
            self.bucket_exprs[key] = result
        return result

    def _rebuild_arg(self, expr: Expr, bucket_ids: list[int], path: tuple[int, ...]) -> Expr:
        rebuilt = self._rebuild_expr(expr, path)
        leaves = _tensor_leaves(expr.type)
        if not leaves:
            return rebuilt
        if not bucket_ids:
            return rebuilt
        if len(bucket_ids) < len(leaves):
            raise self._fail(
                f"candidate input leaves do not match {type(expr).__name__}"
            )
        selected = tuple(bucket_ids.pop(0) for _ in leaves)
        return self._coerce_structured_arg(rebuilt, expr.type, selected, path)

    def _project(self, value: Expr, leaf_path: tuple[int, ...]) -> Expr:
        current = value
        for index in leaf_path:
            if isinstance(current, Tuple):
                try:
                    current = current.elements[index]
                except IndexError as exc:
                    raise self._fail("tuple projection is outside the rebuilt value") from exc
                continue
            if not isinstance(current.type, TupleType):
                raise self._fail("tuple projection reached a non-tuple value")
            if index < 0 or index >= len(current.type.fields):
                raise self._fail("tuple projection index is outside the rebuilt value")
            field_type = current.type.fields[index]
            current = self._infer(
                Call(
                    type=field_type,
                    target=TupleGetItem(index=index),
                    args=(current,),
                    loc=getattr(current, "loc", None),
                )
            )
        return current

    def _coerce_structured_arg(
        self,
        value: Expr,
        source_type: Type,
        bucket_ids: tuple[int, ...],
        path: tuple[int, ...],
    ) -> Expr:
        iterator = iter(bucket_ids)

        def visit(current: Expr, current_type: Type, leaf_path: tuple[int, ...]) -> Expr:
            if isinstance(current_type, TensorType):
                try:
                    bucket_id = next(iterator)
                except StopIteration as exc:
                    raise self._fail("structured candidate input has too few buckets") from exc
                return self._selected_bucket_expr(bucket_id, path, current)
            if isinstance(current_type, TupleType):
                elements = []
                for index, field_type in enumerate(current_type.fields):
                    child = self._project(current, (index,))
                    elements.append(visit(child, field_type, leaf_path + (index,)))
                return Tuple(
                    elements=tuple(elements),
                    type=TupleType(tuple(element.type for element in elements)),
                    metadata=_clean_metadata(current) if isinstance(current, Tuple) else (),
                )
            return current

        result = visit(value, source_type, ())
        try:
            next(iterator)
        except StopIteration:
            return result
        raise self._fail("structured candidate input has too many buckets")

    def _candidate_for_expr(self, expr: Call, path: tuple[int, ...]) -> OpCandidate:
        refs = self.index.expr_refs.get((path, id(expr)), ())
        if not refs:
            raise self._fail(f"operation {type(expr.target).__name__} has no planned output values")
        site_ids = {self.problem.values[value_id].producer_site_id for value_id in refs}
        if len(site_ids) != 1 or None in site_ids:
            raise self._fail(f"operation {type(expr.target).__name__} has inconsistent site identity")
        candidate_id = self.candidate_by_site.get(next(iter(site_ids)))
        if candidate_id is None:
            raise self._fail(f"operation site {next(iter(site_ids))} has no selected candidate")
        candidate = self.problem.candidates[candidate_id]
        if candidate.source_call is None or type(candidate.op) is not type(expr.target):
            raise self._fail(f"selected candidate {candidate_id} does not match the authored operation")
        return candidate

    def _materialized_op(self, candidate: OpCandidate) -> object:
        if isinstance(candidate.op, Reshard):
            target = self._placed_tensor_type(candidate.output_bucket_ids[0])
            return Reshard(layout=target.layout, storage=target.storage)
        return candidate.op

    def _broadcast_shards(self, value: Expr, path: tuple[int, ...], mesh=None) -> Expr:
        if not isinstance(value.type, TensorType) or not isinstance(value.type.layout, ShardLayout):
            return value
        layout = value.type.layout
        if all(isinstance(attr, Broadcast) for attr in layout.attrs) and (
            mesh is None or layout.mesh == mesh
        ):
            return value
        key = (path, id(value), mesh)
        cached = self.broadcast_exprs.get(key)
        if cached is not None:
            return cached
        target_layout = replace(
            layout,
            attrs=tuple(Broadcast() for _attr in layout.attrs),
            mesh=layout.mesh if mesh is None else mesh,
        )
        result = self._infer(
            Call(
                type=replace(value.type, layout=target_layout),
                target=Reshard(layout=target_layout, storage=value.type.storage),
                args=(value,),
                loc=getattr(value, "loc", None),
            )
        )
        self.broadcast_exprs[key] = result
        return result

    def _rebuild_grid(self, region: GridRegionExpr, path: tuple[int, ...], instance: _Instance) -> Expr:
        region_id = self.index.region_ids.get(id(region))
        if region_id is None:
            raise self._fail("GridRegion source is absent from the planning problem")
        init_args = tuple(self._rebuild_expr(value, path) for value in region.init_args)
        body_path = path + (region_id,)
        carry_ids = tuple(
            value_id
            for old_phi in region.carried_args
            for value_id, _leaf_path in sorted(
                (
                    (value_id, value.leaf_path)
                    for value_id, value in self.problem.values.items()
                    if value.source is old_phi
                    and value.function_path == body_path
                    and value.role == "carry"
                ),
                key=lambda item: item[1],
            )
        )
        carry_iter = iter(carry_ids)
        carried = []
        body_env = dict(instance.env)
        for old_phi in region.carried_args:
            field_count = len(_tensor_leaves(old_phi.type))
            refs = tuple(next(carry_iter) for _ in range(field_count))
            new_type = self._materialize_type(old_phi.type, value_ids=refs)
            new_phi = replace(old_phi, type=new_type, metadata=_clean_metadata(old_phi))
            carried.append(new_phi)
            body_env[id(old_phi)] = new_phi
        induction = replace(
            region.induction_var,
            metadata=_clean_metadata(region.induction_var),
        )
        body_env[id(region.induction_var)] = induction
        body = self._rebuild_expr_with_env(region.body, body_path, body_env)
        yields = tuple(self._rebuild_expr_with_env(value, body_path, body_env) for value in region.yield_values)
        rebuilt = replace(
            region,
            induction_var=induction,
            carried_args=tuple(carried),
            init_args=init_args,
            body=body,
            yield_values=yields,
            metadata=_clean_metadata(region),
        )
        rebuilt = self._infer_grid(rebuilt)
        return rebuilt

    def _infer_grid(self, grid: GridRegionExpr) -> Expr:
        inferred = TypeInferContext(module=self.problem.module).type_of(grid)
        return replace(grid, type=inferred)

    def _rebuild_expr_with_env(self, expr: Expr, path: tuple[int, ...], env: dict[int, Expr]) -> Expr:
        old = self.instances.get(path)
        if old is None:
            parent = self.instances.get(path[:-1])
            if parent is None:
                raise self._fail(f"GridRegion scope {path} has no owning function instance")
            old = _Instance(path, parent.source, parent.params, env)
            self.instances[path] = old
            try:
                return self._rebuild_expr(expr, path)
            finally:
                self.instances.pop(path, None)
        saved = old.env
        old.env = env
        try:
            return self._rebuild_expr(expr, path)
        finally:
            old.env = saved

    def _rebuild_expr(self, expr: Expr | None, path: tuple[int, ...]) -> Expr:
        if expr is None:
            raise self._fail("cannot rebuild a missing function body")
        instance = self.instances[path]
        cached = instance.expr_cache.get(id(expr))
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            result = instance.env.get(id(expr))
            if result is None:
                result = _clean(expr)
            instance.expr_cache[id(expr)] = result
            return result
        if isinstance(expr, Constant):
            result = _clean(expr)
            instance.expr_cache[id(expr)] = result
            return result
        if isinstance(expr, Tuple):
            elements = tuple(self._rebuild_expr(element, path) for element in expr.elements)
            result = replace(
                expr,
                elements=elements,
                type=TupleType(tuple(element.type for element in elements)),
                metadata=_clean_metadata(expr),
            )
            instance.expr_cache[id(expr)] = result
            return result
        if isinstance(expr, GridRegionExpr):
            result = self._rebuild_grid(expr, path, instance)
            instance.expr_cache[id(expr)] = result
            return result
        if not isinstance(expr, Call):
            raise self._fail(f"unsupported HIR expression {type(expr).__name__}")
        if isinstance(expr.target, Function):
            args = tuple(self._rebuild_expr(arg, path) for arg in expr.args)
            child_path = self.index.call_paths.get((path, id(expr)))
            if child_path is None:
                raise self._fail(f"helper call {expr.target.name!r} has no lexical path")
            target = self._build_helper(child_path, expr.target, args)
            result = self._infer(
                replace(expr, target=target, args=args, metadata=_clean_metadata(expr))
            )
            instance.expr_cache[id(expr)] = result
            return result
        if isinstance(expr.target, TupleGetItem):
            source = self._rebuild_expr(expr.args[0], path)
            result = self._infer(
                replace(expr, args=(source,), metadata=_clean_metadata(expr))
            )
            instance.expr_cache[id(expr)] = result
            return result
        candidate = self._candidate_for_expr(expr, path)
        buckets = list(candidate.input_bucket_ids)
        args = tuple(self._rebuild_arg(arg, buckets, path) for arg in expr.args)
        if buckets:
            raise self._fail(f"candidate {candidate.site_id} has unused input buckets")
        call = Call(
            type=expr.type,
            target=self._materialized_op(candidate),
            args=args,
            loc=expr.loc,
            metadata=_clean_metadata(expr),
        )
        try:
            result = self._infer(call)
        except RuntimeError as exc:
            message = str(exc)
            normalized_message = message.lower()
            if (
                "reshard" not in normalized_message
                and "must not be split-sharded" not in normalized_message
                and "different meshes" not in normalized_message
                and "split on the same mesh axes" not in normalized_message
            ):
                raise
            mesh = next(
                (
                    arg.type.layout.mesh
                    for arg in args
                    if isinstance(arg.type, TensorType)
                    and isinstance(arg.type.layout, ShardLayout)
                ),
                None,
            )
            call = replace(call, args=tuple(
                self._broadcast_shards(arg, path, mesh) for arg in args
            ))
            try:
                result = self._infer(call)
            except RuntimeError as retry_exc:
                meshes = tuple(
                    getattr(getattr(arg.type, "layout", None), "mesh", None)
                    for arg in call.args
                )
                raise self._fail(
                    f"selected operation {candidate.site_id} remains invalid after "
                    f"broadcast normalization: {retry_exc}; meshes={meshes!r}"
                ) from retry_exc
        result = self._materialize_result(
            result,
            candidate.output_bucket_ids,
            f"selected operation {candidate.site_id}",
        )
        instance.expr_cache[id(expr)] = result
        for bucket_id in candidate.output_bucket_ids:
            self.bucket_exprs.setdefault((path, bucket_id), result)
        return result

    def _build_root(self) -> Function:
        source = self.problem.root
        env: dict[int, Expr] = {}
        params = []
        for param in source.params:
            refs = self.index.param_refs[((), id(param))]
            param_type = self._materialize_type(param.type, value_ids=refs)
            new_param = replace(param, type=param_type, metadata=_clean_metadata(param))
            params.append(new_param)
            env[id(param)] = new_param
            for value_id, (leaf_path, _type) in zip(refs, _tensor_leaves(param.type)):
                bucket_id = self._selected_bucket_for_value(value_id)
                self.bucket_exprs[((), bucket_id)] = self._project(new_param, leaf_path)
        instance = _Instance((), source, tuple(params), env)
        self.instances[()] = instance
        body = self._rebuild_expr(source.body, ())
        result = Function.build(
            name=source.name,
            params=tuple(params),
            body=body,
            return_type=body.type,
            topologies=source.topologies,
            specializations=source.specializations,
            target=source.target,
            loc=source.loc,
        )
        result = replace(result, metadata=tuple(
            value for value in source.metadata if type(value) is not ScheduleConstraintMetadata
        ))
        instance.clone = result
        return result

    def _build_helper(self, path: tuple[int, ...], source: Function, args: tuple[Expr, ...]) -> Function:
        cached = self.instances.get(path)
        if cached is not None and cached.clone is not None:
            return cached.clone
        if cached is not None:
            return cached.clone  # type: ignore[return-value]
        params = tuple(
            replace(param, type=arg.type, metadata=_clean_metadata(param))
            for param, arg in zip(source.params, args)
        )
        env = {id(old): new for old, new in zip(source.params, params)}
        instance = _Instance(path, source, params, env)
        self.instances[path] = instance
        body = self._rebuild_expr(source.body, path)
        name = f"{source.name}__cta_{'_'.join(str(part) for part in path)}"
        clone = Function.build(
            name=name,
            params=params,
            body=body,
            return_type=body.type,
            topologies=source.topologies,
            specializations=source.specializations,
            target=source.target,
            loc=source.loc,
        )
        clone = replace(clone, metadata=tuple(
            value for value in source.metadata if type(value) is not ScheduleConstraintMetadata
        ))
        instance.clone = clone
        return clone

    def build(self) -> Module:
        root = self._build_root()
        cloned = {id(instance.source) for path, instance in self.instances.items() if path and instance.clone is not None}
        functions = []
        for function in self.problem.module.functions:
            if function is self.problem.root:
                functions.append(root)
                continue
            if id(function) in cloned:
                functions.extend(
                    instance.clone
                    for path, instance in sorted(self.instances.items())
                    if path and instance.source is function and instance.clone is not None
                )
                continue
            functions.append(function)
        module_function_ids = {id(function) for function in self.problem.module.functions}
        extra_clones = [
            instance.clone
            for path, instance in sorted(self.instances.items())
            if path and instance.clone is not None and id(instance.source) not in module_function_ids
        ]
        functions.extend(extra_clones)
        result = Module(
            name=self.problem.module.name,
            functions=tuple(functions),
            entry=self.problem.module.entry,
            topologies=self.problem.module.topologies,
            metadata=dict(self.problem.module.metadata),
        )
        try:
            verify_module(result.functions)
        except Exception as exc:
            raise self._fail(f"rebuilt Module verification failed: {exc}") from exc
        return result


def materialize_planning_solution(
    problem: PlanningProblem,
    solution: PlanningSolution,
) -> Module:
    """Rebuild one selected P3 graph into a fresh verified HIR Module."""
    if not isinstance(problem, PlanningProblem):
        raise TypeError(f"problem must be a PlanningProblem, got {type(problem).__name__}")
    if not isinstance(solution, PlanningSolution):
        raise TypeError(f"solution must be a PlanningSolution, got {type(solution).__name__}")
    if solution.status not in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}:
        raise ValueError(f"solution has unsupported status {solution.status!r}")
    return _Materializer(problem, solution).build()


__all__ = ["materialize_planning_solution"]
