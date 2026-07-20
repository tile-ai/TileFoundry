"""Private finite CTA parallel-planning problem construction."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Mapping

from tilefoundry.ir.core import Call, Constant, Expr, Op, Tuple, Var, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import TensorType, TupleType, Type, make_shard_tensor_type
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import (
    Broadcast,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    Split,
    Topology,
    try_c_order_strides,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    constraint_metadata,
    is_layout_wildcard,
)
from tilefoundry.target.cuda.target import CudaTarget
from tilefoundry.visitor_registry.contexts import Cost, CostContext, TypeInferContext
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor

# Importing the module installs the private evaluator registrations. It is
# deliberately not re-exported from either the public schedule or target API.
from . import cost as _cost  # noqa: F401

PlacementRelation = Literal["SAME_INTERVAL", "CONTAINED"]


@dataclass(frozen=True)
class ValueInfo:
    source: Expr
    leaf_path: tuple[int, ...]
    is_const: bool
    source_bucket_ids: tuple[int, ...] = ()
    producer_site_id: int | None = None
    function_path: tuple[int, ...] = ()
    is_final_output: bool = False


@dataclass(frozen=True)
class CandidateBucket:
    value_id: int
    type_id: int
    candidate_ids: tuple[int, ...]
    fixed_offset: int | None
    is_source: bool


@dataclass(frozen=True)
class BucketRequirement:
    value_id: int
    bucket_ids: tuple[int, ...]
    source: Expr
    metadata: ScheduleConstraintMetadata


@dataclass(frozen=True)
class CandidateDependency:
    parent_candidate_id: int
    input_index: int
    child_bucket_id: int
    placement_relation: PlacementRelation | None


@dataclass(frozen=True)
class OpCandidate:
    op: Op
    input_bucket_ids: tuple[int, ...]
    output_bucket_ids: tuple[int, ...]
    output_alias_input_indices: tuple[int | None, ...]
    active_mesh: Mesh
    topology_count: int
    local_cost: Cost
    duration_ns: int
    total_hbm_bytes: int
    hbm_demand_bytes_per_ns: int
    moved_bytes: int
    site_id: int | None = None
    source_call: Call | None = None
    source_types: tuple[Type, ...] = ()
    output_types: tuple[Type, ...] = ()


@dataclass(frozen=True)
class RegionInfo:
    source: GridRegionExpr
    parent_region_id: int | None
    trip_count: int
    operation_site_ids: tuple[int, ...]
    init_use_ids: tuple[int, ...]
    backedge_use_ids: tuple[int, ...]
    carry_infos: tuple["RegionCarryInfo", ...] = ()
    result_value_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RegionCarryInfo:
    init_value_id: int
    carried_value_id: int
    yield_value_id: int
    result_value_id: int


@dataclass(frozen=True)
class PlanningProblem:
    module: Module
    root: Function
    topology: Topology
    types: tuple[Type, ...]
    values: Mapping[int, ValueInfo]
    buckets: Mapping[int, CandidateBucket]
    candidates: Mapping[int, OpCandidate]
    authored_candidates: Mapping[int, tuple[int, ...]]
    dependencies: tuple[CandidateDependency, ...]
    requirements: tuple[BucketRequirement, ...]
    root_value_ids: tuple[int, ...]
    regions: Mapping[int, RegionInfo]
    candidate_enclosing_regions: Mapping[int, int | None] = MappingProxyType({})
    value_availability_regions: Mapping[int, int | None] = MappingProxyType({})
    site_order: tuple[int, ...] = ()
    function_instances: tuple[tuple[tuple[int, ...], Function], ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass
class _Site:
    site_id: int
    call: Call
    function_path: tuple[int, ...]
    input_value_ids: tuple[tuple[int, ...], ...]
    output_value_ids: tuple[int, ...]


def _tensor_leaves(type: Type, path: tuple[int, ...] = ()) -> tuple[tuple[tuple[int, ...], TensorType], ...]:
    if isinstance(type, TensorType):
        return ((path, type),)
    if isinstance(type, TupleType):
        return tuple(
            leaf
            for index, field in enumerate(type.fields)
            for leaf in _tensor_leaves(field, path + (index,))
        )
    return ()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _mesh(count: int) -> Mesh:
    return Mesh(
        Topology("cta", count),
        Layout(shape=(count,), strides=(1,)),
    )


def _type_mesh(type: TensorType, fallback: Mesh) -> Mesh:
    if isinstance(type.layout, ShardLayout):
        mesh = type.layout.mesh
        if len(mesh.axes) != 1:
            raise ValueError("CTA planning requires rank-one candidate meshes")
        return mesh
    return fallback


def _is_tensor_type(type: Type) -> bool:
    return isinstance(type, TensorType)


def _type_storage_ok(type: TensorType) -> bool:
    return type.storage is StorageKind.GMEM


def _same_logical_tensor(a: TensorType, b: TensorType) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.storage == b.storage


def _layout_matches(actual: object, constraint: LayoutConstraint) -> bool:
    if not isinstance(actual, ShardLayout):
        return False
    expected = constraint.layout
    shape = tuple(actual.layout.shape)
    if len(shape) != len(expected.shape):
        return False
    for got, want in zip(shape, expected.shape):
        if not is_layout_wildcard(want) and got != want:
            return False
    for topology, attr in constraint.bindings:
        if actual.mesh.topology.name != topology:
            return False
        try:
            index = actual.mesh.names.index(topology)
        except ValueError:
            index = 0
        if index >= len(actual.attrs) or actual.attrs[index] != attr:
            return False
    return True


def _bucket_matches(type: TensorType, constraints: tuple[object, ...]) -> bool:
    for constraint in constraints:
        if isinstance(constraint, LayoutConstraint) and not _layout_matches(type.layout, constraint):
            return False
        if isinstance(constraint, MeshConstraint):
            if not isinstance(type.layout, ShardLayout) or type.layout.mesh != constraint.mesh:
                return False
        if isinstance(constraint, StorageConstraint) and type.storage != constraint.storage:
            return False
    return True


def _placement_relation(type: TensorType, mesh: Mesh) -> PlacementRelation | None:
    if not isinstance(type.layout, ShardLayout):
        return "SAME_INTERVAL" if mesh.layout.shape == (1,) else "CONTAINED"
    if type.layout.mesh == mesh:
        return "SAME_INTERVAL"
    attrs = type.layout.attrs
    if attrs and all(isinstance(attr, Broadcast) for attr in attrs):
        return "CONTAINED"
    return None


class _Planner:
    def __init__(self, module: Module, root: Function, target: CudaTarget, topology: Topology) -> None:
        self.module = module
        self.root = root
        self.target = target
        self.topology = topology
        self.types: list[Type] = []
        self.type_ids: dict[Type, int] = {}
        self.values: dict[int, ValueInfo] = {}
        self.value_types: dict[int, tuple[Type, ...]] = {}
        self.buckets: dict[int, CandidateBucket] = {}
        self._bucket_candidates: dict[int, list[int]] = {}
        self._bucket_by_value_type: dict[tuple[int, int], int] = {}
        self.candidates: dict[int, OpCandidate] = {}
        self.authored_candidates: dict[int, tuple[int, ...]] = {}
        self.dependencies: list[CandidateDependency] = []
        self.requirement_annotations: list[tuple[tuple[int, ...], Expr, ScheduleConstraintMetadata]] = []
        self.requirements: list[BucketRequirement] = []
        self.regions: dict[int, RegionInfo] = {}
        self.candidate_enclosing_regions: dict[int, int | None] = {}
        self.value_availability_regions: dict[int, int | None] = {}
        self.site_enclosing_regions: dict[int, int | None] = {}
        self.sites: list[_Site] = []
        self.site_order: list[int] = []
        self.function_instances: list[tuple[tuple[int, ...], Function]] = []
        self._expr_values: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
        self._param_values: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
        self._active_functions: set[int] = set()
        self._active_regions: list[int] = []
        self._next_value = 0
        self._next_site = 0
        self._next_candidate = 0
        self._next_bucket = 0
        self._next_region = 0
        self._next_use = 0
        self.required_meshes: list[Mesh] = []
        self.counts = self._resource_counts()

    def _resource_counts(self) -> tuple[int, ...]:
        extents: set[int] = {1}
        seen_functions: set[int] = set()

        def visit_type(type: Type) -> None:
            for _, tensor in _tensor_leaves(type):
                if isinstance(tensor.layout, ShardLayout):
                    for attr in tensor.layout.attrs:
                        if isinstance(attr, Split) and attr.axis < len(tensor.layout.layout.shape):
                            dim = tensor.layout.layout.shape[attr.axis]
                            if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                                extents.add(dim)
                    for dim in tensor.layout.mesh.layout.shape:
                        if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                            extents.add(dim)

        def visit_expr(expr: Expr | None) -> None:
            if expr is None:
                return
            visit_type(expr.type)
            metadata = constraint_metadata(expr)
            if metadata is not None:
                for constraint in metadata.constraints:
                    if isinstance(constraint, LayoutConstraint):
                        for _, attr in constraint.bindings:
                            if isinstance(attr, Split) and attr.axis < len(constraint.layout.shape):
                                dim = constraint.layout.shape[attr.axis]
                                if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                                    extents.add(dim)
                    elif isinstance(constraint, MeshConstraint) and constraint.mesh is not None:
                        self.required_meshes.append(constraint.mesh)
                        for dim in constraint.mesh.layout.shape:
                            if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                                extents.add(dim)
            if isinstance(expr, Call):
                if isinstance(expr.target, Function):
                    visit_function(expr.target)
                for arg in expr.args:
                    visit_expr(arg)
            elif isinstance(expr, Tuple):
                for element in expr.elements:
                    visit_expr(element)
            elif isinstance(expr, GridRegionExpr):
                for value in (*expr.init_args, expr.body, *expr.yield_values):
                    visit_expr(value)

        def visit_function(function: Function) -> None:
            if id(function) in seen_functions:
                return
            seen_functions.add(id(function))
            for param in function.params:
                visit_expr(param)
            visit_expr(function.body)

        visit_function(self.root)
        result = {1, self.topology.size}  # type: ignore[arg-type]
        for extent in extents:
            for count in range(1, min(extent, self.topology.size) + 1):
                if extent % count == 0:
                    result.add(count)
        return tuple(sorted(result))

    def _intern(self, type: Type) -> int:
        type_id = self.type_ids.get(type)
        if type_id is None:
            type_id = len(self.types)
            self.type_ids[type] = type_id
            self.types.append(type)
        return type_id

    def _legal_types(self, base: TensorType) -> tuple[Type, ...]:
        if not _type_storage_ok(base):
            raise ValueError(
                f"P2: tensor at {getattr(self.root, 'loc', None) or '<unknown>'} "
                f"requires unsupported storage {base.storage!r}; CTA planning supports GMEM only"
            )
        result: list[TensorType] = [base]
        if isinstance(base.layout, ShardLayout):
            return tuple(result)
        for count in self.counts:
            if count == 1:
                continue
            meshes = [_mesh(count)]
            meshes.extend(
                mesh for mesh in self.required_meshes
                if mesh.layout.shape == (count,) and mesh not in meshes
            )
            for mesh in meshes:
                replicated = TensorType(
                    shape=base.shape,
                    dtype=base.dtype,
                    storage=base.storage,
                    layout=ShardLayout(
                        layout=Layout(shape=base.shape, strides=try_c_order_strides(base.shape)),
                        attrs=(Broadcast(),),
                        mesh=mesh,
                    ),
                )
                result.append(replicated)
                result.append(
                    replace(replicated, layout=replace(replicated.layout, attrs=(Partial("sum"),)))  # type: ignore[arg-type]
                )
                for axis, dim in enumerate(base.shape):
                    if isinstance(dim, int) and not isinstance(dim, bool) and dim % count == 0:
                        result.append(
                            make_shard_tensor_type(
                                base.shape, base.dtype, base.storage, mesh, (Split(axis),)
                            )
                        )
        dedup: list[TensorType] = []
        for type in result:
            if type not in dedup:
                dedup.append(type)
                self._intern(type)
        return tuple(dedup)

    def _new_value(
        self,
        source: Expr,
        type: TensorType,
        leaf_path: tuple[int, ...],
        function_path: tuple[int, ...],
        *,
        is_const: bool = False,
        producer_site_id: int | None = None,
    ) -> int:
        if not _type_storage_ok(type):
            loc = getattr(source, "loc", None) or getattr(self.root, "loc", None) or "<unknown>"
            raise ValueError(
                f"P2: tensor storage {type.storage!r} at {loc} is unsupported; "
                "scheduled tensor values must reside in GMEM"
            )
        value_id = self._next_value
        self._next_value += 1
        info = ValueInfo(
            source=source,
            leaf_path=leaf_path,
            is_const=is_const,
            producer_site_id=producer_site_id,
            function_path=function_path,
        )
        self.values[value_id] = info
        self.value_availability_regions[value_id] = (
            self._active_regions[-1] if self._active_regions else None
        )
        self.value_types[value_id] = self._legal_types(type)
        return value_id

    def _init_buckets(self) -> None:
        for value_id, types in self.value_types.items():
            source_bucket_ids: list[int] = []
            for type in types:
                type_id = self._intern(type)
                bucket_id = self._next_bucket
                self._next_bucket += 1
                self._bucket_by_value_type[(value_id, type_id)] = bucket_id
                self._bucket_candidates[bucket_id] = []
                self.buckets[bucket_id] = CandidateBucket(
                    value_id=value_id,
                    type_id=type_id,
                    candidate_ids=(),
                    fixed_offset=None,
                    is_source=self.values[value_id].producer_site_id is None,
                )
                if self.values[value_id].producer_site_id is None:
                    source_bucket_ids.append(bucket_id)
            if source_bucket_ids:
                self.values[value_id] = replace(
                    self.values[value_id], source_bucket_ids=tuple(source_bucket_ids)
                )

    def _source_value(self, param: Var, function_path: tuple[int, ...]) -> tuple[int, ...]:
        refs = tuple(
            self._new_value(param, type, path, function_path, is_const=param.is_const)
            for path, type in _tensor_leaves(param.type)
        )
        self._param_values[(function_path, id(param))] = refs
        self._record_requirement(refs, param)
        return refs

    def _record_requirement(self, refs: tuple[int, ...], source: Expr) -> None:
        metadata = constraint_metadata(source)
        if metadata is not None:
            for ref in refs:
                self.requirement_annotations.append(( (ref,), source, metadata))

    def _process_expr(
        self,
        expr: Expr | None,
        function: Function,
        function_path: tuple[int, ...],
        env: Mapping[int, tuple[int, ...]],
    ) -> tuple[int, ...]:
        if expr is None:
            return ()
        key = (function_path, id(expr))
        cached = self._expr_values.get(key)
        if cached is not None:
            return cached
        if isinstance(expr, Var):
            refs = env.get(id(expr))
            if refs is None:
                refs = self._param_values.get((function_path, id(expr)), ())
            self._expr_values[key] = refs
            return refs
        if isinstance(expr, Constant):
            self._expr_values[key] = ()
            return ()
        if isinstance(expr, Tuple):
            refs = tuple(
                ref
                for element in expr.elements
                for ref in self._process_expr(element, function, function_path, env)
            )
            self._expr_values[key] = refs
            self._record_requirement(refs, expr)
            return refs
        if isinstance(expr, GridRegionExpr):
            refs = self._process_region(expr, function, function_path, env)
            self._expr_values[key] = refs
            self._record_requirement(refs, expr)
            return refs
        if not isinstance(expr, Call):
            self._expr_values[key] = ()
            return ()
        arg_refs = tuple(
            self._process_expr(arg, function, function_path, env) for arg in expr.args
        )
        target = expr.target
        if isinstance(target, Function):
            call_path = function_path + (len(self.function_instances),)
            helper_env = dict(zip(
                (id(param) for param in target.params), arg_refs
            ))
            self._process_function(target, call_path, helper_env, parent_call=expr)
            refs = self._process_expr(target.body, target, call_path, helper_env)
            self._expr_values[key] = refs
            self._record_requirement(refs, expr)
            return refs
        if isinstance(target, (PrimFunction, Launch, SymbolRef)):
            raise ValueError(
                f"P2: kernel boundary {type(target).__name__} at "
                f"{expr.loc or '<unknown>'} is not a CTA planning operation"
            )
        if isinstance(target, TupleGetItem):
            source_refs = arg_refs[0] if arg_refs else ()
            fields = _tensor_leaves(expr.args[0].type)
            index = target.index
            if index < 0 or index >= len(fields):
                raise ValueError(f"P2: TupleGetItem index {index} is out of range at {expr.loc or '<unknown>'}")
            start = sum(len(_tensor_leaves(field)) for field in expr.args[0].type.fields[:index])  # type: ignore[union-attr]
            refs = source_refs[start:start + len(_tensor_leaves(expr.args[0].type.fields[index]))]  # type: ignore[union-attr]
            self._expr_values[key] = refs
            self._record_requirement(refs, expr)
            return refs
        site_id = self._next_site
        self._next_site += 1
        output_refs = tuple(
            self._new_value(expr, type, path, function_path, producer_site_id=site_id)
            for path, type in _tensor_leaves(expr.type)
        )
        site = _Site(site_id, expr, function_path, arg_refs, output_refs)
        self.sites.append(site)
        self.site_order.append(site_id)
        self.site_enclosing_regions[site_id] = (
            self._active_regions[-1] if self._active_regions else None
        )
        if self._active_regions:
            region_id = self._active_regions[-1]
            info = self.regions[region_id]
            self.regions[region_id] = replace(
                info, operation_site_ids=(*info.operation_site_ids, site_id)
            )
        self._expr_values[key] = output_refs
        self._record_requirement(output_refs, expr)
        return output_refs

    def _process_region(
        self,
        region: GridRegionExpr,
        function: Function,
        function_path: tuple[int, ...],
        env: Mapping[int, tuple[int, ...]],
    ) -> tuple[int, ...]:
        start = static_dim_value(region.start)
        stop = static_dim_value(region.extent)
        step = static_dim_value(region.step)
        context = f"GridRegion at {region.loc or getattr(function, 'loc', None) or '<unknown>'}"
        if start is None or stop is None or step is None:
            raise ValueError(f"P2: {context} requires static start, stop, and step")
        if start < 0 or step <= 0:
            raise ValueError(f"P2: {context} has invalid start/step ({start}, {step})")
        trip_count = _ceil_div(stop - start, step) if stop > start else 0
        if trip_count <= 0:
            raise ValueError(f"P2: {context} has non-positive trip count")
        region_id = self._next_region
        self._next_region += 1
        info = RegionInfo(
            source=region,
            parent_region_id=self._active_regions[-1] if self._active_regions else None,
            trip_count=trip_count,
            operation_site_ids=(),
            init_use_ids=tuple(self._new_use() for _ in region.init_args),
            backedge_use_ids=tuple(self._new_use() for _ in region.yield_values),
        )
        self.regions[region_id] = info
        init_ref_groups = tuple(
            self._process_expr(value, function, function_path, env)
            for value in region.init_args
        )
        init_refs = tuple(ref for refs in init_ref_groups for ref in refs)
        phi_env = dict(env)
        for phi, refs in zip(region.carried_args, init_ref_groups):
            phi_env[id(phi)] = refs
        self._active_regions.append(region_id)
        try:
            body_refs = self._process_expr(region.body, function, function_path + (region_id,), phi_env)
            yield_ref_groups = tuple(
                self._process_expr(value, function, function_path + (region_id,), phi_env)
                for value in region.yield_values
            )
            yield_refs = tuple(ref for refs in yield_ref_groups for ref in refs)
        finally:
            self._active_regions.pop()
        result_refs = yield_refs if region.carried_args else body_refs
        parent_region_id = info.parent_region_id
        for ref in result_refs:
            self.value_availability_regions[ref] = parent_region_id
        carry_infos = tuple(
            RegionCarryInfo(init_value_id=init_ref, carried_value_id=init_ref,
                            yield_value_id=yield_ref, result_value_id=yield_ref)
            for init_ref, yield_ref in zip(init_refs, yield_refs)
        )
        self.regions[region_id] = replace(
            self.regions[region_id],
            carry_infos=carry_infos,
            result_value_ids=result_refs,
        )
        return result_refs

    def _new_use(self) -> int:
        value = self._next_use
        self._next_use += 1
        return value

    def _process_function(
        self,
        function: Function,
        function_path: tuple[int, ...],
        env: Mapping[int, tuple[int, ...]],
        *,
        parent_call: Call | None = None,
    ) -> None:
        if function.body is None:
            raise ValueError(
                f"P2: helper function {function.name!r} has no body at "
                f"{parent_call.loc if parent_call else function.loc or '<unknown>'}"
            )
        if id(function) in self._active_functions:
            raise ValueError(
                f"P2: recursive helper call to {function.name!r} at "
                f"{parent_call.loc if parent_call else '<unknown>'}"
            )
        if function is not self.root:
            if function.target is not None and function.target != self.target:
                raise ValueError(
                    f"P2: helper {function.name!r} Target conflicts at "
                    f"{parent_call.loc if parent_call else '<unknown>'}"
                )
            if function.topologies:
                raise ValueError(
                    f"P2: helper {function.name!r} declares program topologies at "
                    f"{parent_call.loc if parent_call else '<unknown>'}"
                )
        self._active_functions.add(id(function))
        self.function_instances.append((function_path, function))
        try:
            if function is self.root:
                for param in function.params:
                    self._source_value(param, function_path)
            else:
                for param, refs in zip(function.params, env.values()):
                    self._param_values[(function_path, id(param))] = refs
            self._process_expr(function.body, function, function_path, env)
        finally:
            self._active_functions.remove(id(function))

    def _retag(self, expr: Expr, types: itertools.chain) -> Expr:
        if isinstance(expr, TensorType):  # pragma: no cover - Type is not Expr
            return expr
        if isinstance(expr, Tuple):
            new_elements = tuple(self._retag(element, types) for element in expr.elements)
            if new_elements == expr.elements:
                return expr
            return replace(expr, elements=new_elements)
        if isinstance(expr.type, TensorType):
            try:
                type = next(types)
            except StopIteration:
                return expr
            return replace(expr, type=type)
        return expr

    def _candidate_call(
        self, site: _Site, input_types: tuple[Type, ...]
    ) -> tuple[Call, tuple[Type, ...]]:
        type_iter = iter(input_types)
        args = tuple(self._retag(arg, type_iter) for arg in site.call.args)
        call = replace(site.call, args=args)
        return call, input_types

    def _active_mesh_for_outputs(self, outputs: tuple[TensorType, ...]) -> tuple[Mesh, int]:
        meshes = tuple(_type_mesh(output, _mesh(1)) for output in outputs)
        mesh = meshes[0] if meshes else _mesh(1)
        if any(other != mesh for other in meshes[1:]):
            raise ValueError("P2: multi-output candidate leaves require one shared Mesh")
        if len(mesh.axes) != 1:
            raise ValueError("P2: candidate Mesh must be rank one")
        count = mesh.layout.shape[0]
        if not isinstance(count, int) or not 1 <= count <= self.topology.size:  # type: ignore[operator]
            raise ValueError(f"P2: candidate Mesh extent {count!r} is outside CTA topology")
        return mesh, count

    def _target_facts(
        self, call: Call, cost: Cost, mesh: Mesh, count: int
    ) -> tuple[int, int, int, int]:
        device = self.target.device
        if isinstance(call.target, Reshard):
            moved = cost.bytes
            duration = _ceil_div(moved * 1_000_000_000, device.hbm_bandwidth_bytes_per_second)
            demand = _ceil_div(moved, duration) if duration else 0
            return duration, moved, demand, moved
        if count == 0:
            raise ValueError("P2: only Reshard candidates may have topology_count=0")
        total_flops = {dtype: value * count for dtype, value in cost.flops.items()}
        compute = 0
        for dtype, flops in total_flops.items():
            compute += _ceil_div(
                flops * 1_000_000_000 * device.sm_count,
                device.peak_for(dtype) * count,
            )
        total_bytes = cost.bytes * count
        memory = _ceil_div(
            total_bytes * 1_000_000_000,
            device.hbm_bandwidth_bytes_per_second,
        ) if total_bytes else 0
        duration = max(compute, memory, 1) if cost.flops or cost.bytes else 0
        demand = _ceil_div(total_bytes, duration) if duration else 0
        return duration, total_bytes, demand, 0

    def _add_candidate(
        self,
        site_id: int | None,
        call: Call,
        input_bucket_ids: tuple[int, ...],
        output_bucket_ids: tuple[int, ...],
        source_types: tuple[Type, ...],
        output_types: tuple[Type, ...],
        cost: Cost,
        *,
        reshard: bool = False,
    ) -> int:
        tensor_outputs = tuple(type for type in output_types if isinstance(type, TensorType))
        mesh, count = self._active_mesh_for_outputs(tensor_outputs)
        if reshard:
            count = 0
        duration, total_bytes, demand, moved = self._target_facts(call, cost, mesh, count)
        aliases = tuple(
            0 if isinstance(call.target, (Reshape, Transpose)) and index == 0 else None
            for index, _ in enumerate(output_bucket_ids)
        )
        candidate = OpCandidate(
            op=call.target,
            input_bucket_ids=input_bucket_ids,
            output_bucket_ids=output_bucket_ids,
            output_alias_input_indices=aliases,
            active_mesh=mesh,
            topology_count=count,
            local_cost=cost,
            duration_ns=duration,
            total_hbm_bytes=total_bytes,
            hbm_demand_bytes_per_ns=demand,
            moved_bytes=moved,
            site_id=site_id,
            source_call=None if reshard else call,
            source_types=source_types,
            output_types=output_types,
        )
        candidate_id = self._next_candidate
        self._next_candidate += 1
        self.candidates[candidate_id] = candidate
        if site_id is None:
            value_id = self.buckets[output_bucket_ids[0]].value_id
            self.candidate_enclosing_regions[candidate_id] = (
                self.value_availability_regions.get(value_id)
            )
        else:
            self.candidate_enclosing_regions[candidate_id] = self.site_enclosing_regions.get(site_id)
        for bucket_id in output_bucket_ids:
            self._bucket_candidates[bucket_id].append(candidate_id)
        for input_index, bucket_id in enumerate(input_bucket_ids):
            value_type = self.types[self.buckets[bucket_id].type_id]
            relation = _placement_relation(value_type, mesh) if isinstance(value_type, TensorType) else None
            self.dependencies.append(
                CandidateDependency(candidate_id, input_index, bucket_id, relation)
            )
        return candidate_id

    def _generate_site(self, site: _Site) -> None:
        choices = []
        for refs in site.input_value_ids:
            choices.extend(refs)
        bucket_options = []
        for value_id in choices:
            bucket_options.append(tuple(
                bucket_id
                for (candidate_value, _), bucket_id in self._bucket_by_value_type.items()
                if candidate_value == value_id
            ))
        combinations = itertools.product(*bucket_options) if bucket_options else ((),)
        found: list[int] = []
        seen: set[tuple] = set()
        for input_bucket_ids in combinations:
            input_types = tuple(self.types[self.buckets[bucket_id].type_id] for bucket_id in input_bucket_ids)
            candidate_call, selected_types = self._candidate_call(site, input_types)
            ctx = TypeInferContext(module=self.module)
            try:
                output_type = TypeInferVisitor(ctx).visit(candidate_call)
            except (TypeError, ValueError, NotImplementedError, VerifyError, IndexError):
                continue
            output_leaves = _tensor_leaves(output_type)
            if len(output_leaves) != len(site.output_value_ids):
                continue
            output_types = tuple(type for _, type in output_leaves)
            try:
                output_bucket_ids = tuple(
                    self._bucket_by_value_type[(value_id, self._intern(type))]
                    for value_id, type in zip(site.output_value_ids, output_types)
                )
                cost_ctx = CostContext(
                    module=self.module,
                    selected_types={id(arg): type for arg, type in zip(candidate_call.args, selected_types)},
                    selected_output_type=output_type,
                )
                cost = CostEvaluator(cost_ctx).visit_Call(candidate_call)
                mesh, count = self._active_mesh_for_outputs(output_types)
            except KeyError:
                continue
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"P2: cost evaluation for {type(site.call.target).__name__} at "
                    f"{site.call.loc or '<unknown>'} failed: {exc}"
                ) from exc
            key = (
                input_bucket_ids,
                output_bucket_ids,
                mesh,
                count,
                tuple(sorted(cost.flops.items(), key=lambda item: item[0].name)),
                cost.bytes,
            )
            if key in seen:
                continue
            seen.add(key)
            candidate_id = self._add_candidate(
                site.site_id,
                candidate_call,
                tuple(input_bucket_ids),
                output_bucket_ids,
                selected_types,
                output_types,
                cost,
            )
            found.append(candidate_id)
        if not found:
            raise ValueError(
                f"P2: operation {type(site.call.target).__name__} at "
                f"{site.call.loc or '<unknown>'} has no legal candidates"
            )
        self.authored_candidates[site.site_id] = tuple(found)

    def _synthesized_reshards(self) -> None:
        for value_id, types in self.value_types.items():
            source_buckets = tuple(
                bucket_id
                for (candidate_value, _), bucket_id in self._bucket_by_value_type.items()
                if candidate_value == value_id and self._bucket_candidates[bucket_id]
            )
            for target in types:
                if not isinstance(target, TensorType):
                    continue
                target_bucket = self._bucket_by_value_type[(value_id, self._intern(target))]
                if self._bucket_candidates[target_bucket]:
                    continue
                for source_bucket in source_buckets:
                    source = self.types[self.buckets[source_bucket].type_id]
                    if not isinstance(source, TensorType) or not _same_logical_tensor(source, target):
                        continue
                    if source == target:
                        continue
                    op = Reshard(layout=target.layout, storage=StorageKind.GMEM)
                    source_expr = self.values[value_id].source
                    source_var = Var(type=source, name="reshard_source", loc=getattr(source_expr, "loc", None))
                    call = Call(type=target, target=op, args=(source_var,), loc=getattr(source_expr, "loc", None))
                    cost_ctx = CostContext(
                        module=self.module,
                        selected_types={id(source_var): source},
                        selected_output_type=target,
                    )
                    try:
                        cost = CostEvaluator(cost_ctx).visit_Call(call)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"P2: cost evaluation for synthesized Reshard at "
                            f"{getattr(source_expr, 'loc', None) or '<unknown>'} failed: {exc}"
                        ) from exc
                    self._add_candidate(
                        None,
                        call,
                        (source_bucket,),
                        (target_bucket,),
                        (source,),
                        (target,),
                        cost,
                        reshard=True,
                    )
                    break

    def _finish_buckets(self) -> None:
        for bucket_id, bucket in tuple(self.buckets.items()):
            fixed_offset = None
            for _, source, metadata in self.requirement_annotations:
                if bucket.value_id not in _refs_from_annotation(self, source):
                    continue
                for constraint in metadata.constraints:
                    if isinstance(constraint, MeshConstraint) and constraint.mesh is not None:
                        layout = constraint.mesh.layout
                        if hasattr(layout, "offset"):
                            fixed_offset = layout.offset
            self.buckets[bucket_id] = replace(
                bucket,
                candidate_ids=tuple(self._bucket_candidates[bucket_id]),
                fixed_offset=fixed_offset,
            )
        for refs, source, metadata in self.requirement_annotations:
            value_id = refs[0]
            matching = tuple(
                bucket_id
                for (candidate_value, type_id), bucket_id in self._bucket_by_value_type.items()
                if candidate_value == value_id
                and isinstance(self.types[type_id], TensorType)
                and _bucket_matches(self.types[type_id], metadata.constraints)
            )
            if not matching:
                raise ValueError(
                    f"P2: no candidate bucket satisfies where constraint at "
                    f"{getattr(source, 'loc', None) or '<unknown>'}"
                )
            self.requirements.append(BucketRequirement(value_id, matching, source, metadata))

    def _root_connected(self, value_id: int, seen: set[int]) -> bool:
        if value_id in seen:
            return True
        seen.add(value_id)
        bucket_ids = tuple(
            bucket_id for (candidate_value, _), bucket_id in self._bucket_by_value_type.items()
            if candidate_value == value_id
        )
        if not bucket_ids:
            return False
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            if bucket.is_source:
                return True
            for candidate_id in self._bucket_candidates[bucket_id]:
                candidate = self.candidates[candidate_id]
                if all(self._root_connected(self.buckets[child].value_id, seen) for child in candidate.input_bucket_ids):
                    return True
        return False

    def build(self) -> PlanningProblem:
        self._process_function(self.root, (), {}, parent_call=None)
        self._init_buckets()
        for site in self.sites:
            self._generate_site(site)
        self._synthesized_reshards()
        self._finish_buckets()
        root_refs = self._expr_values.get(((), id(self.root.body)), ())
        if not root_refs:
            raise ValueError("P2: root function has no tensor result value")
        for value_id in root_refs:
            self.values[value_id] = replace(self.values[value_id], is_final_output=True)
            if not self._root_connected(value_id, set()):
                raise ValueError(
                    f"P2: no legal root-connected candidate path for value {value_id}"
                )
        return PlanningProblem(
            module=self.module,
            root=self.root,
            topology=self.topology,
            types=tuple(self.types),
            values=MappingProxyType(dict(self.values)),
            buckets=MappingProxyType(dict(self.buckets)),
            candidates=MappingProxyType(dict(self.candidates)),
            authored_candidates=MappingProxyType(dict(self.authored_candidates)),
            dependencies=tuple(self.dependencies),
            requirements=tuple(self.requirements),
            root_value_ids=tuple(root_refs),
            regions=MappingProxyType(dict(self.regions)),
            candidate_enclosing_regions=MappingProxyType(dict(self.candidate_enclosing_regions)),
            value_availability_regions=MappingProxyType(dict(self.value_availability_regions)),
            site_order=tuple(self.site_order),
            function_instances=tuple(self.function_instances),
            diagnostics=(
                f"ops={len(self.sites)}",
                f"candidates={len(self.candidates)}",
                f"buckets={len(self.buckets)}",
                f"reshards={sum(type(candidate.op) is Reshard and candidate.site_id is None for candidate in self.candidates.values())}",
            ),
        )


def _refs_from_annotation(planner: _Planner, source: Expr) -> tuple[int, ...]:
    for refs, candidate_source, _ in planner.requirement_annotations:
        if candidate_source is source:
            return refs
    return ()


def _validate_entry(module: Module, root: Function) -> tuple[CudaTarget, Topology]:
    if not isinstance(root, Function):
        raise TypeError(f"P2: planning root must be a HIR Function, got {type(root).__name__}")
    if not any(function is root for function in module.functions):
        raise ValueError(f"P2: root function {root.name!r} is not a member of Module {module.name!r}")
    if not isinstance(root.target, CudaTarget):
        raise ValueError(
            f"P2: root {root.name!r} requires an explicit CudaTarget, got "
            f"{type(root.target).__name__ if root.target is not None else 'None'}"
        )
    cta = tuple(topology for topology in root.topologies if topology.name == "cta")
    if len(cta) != 1:
        raise ValueError(f"P2: root {root.name!r} requires exactly one CTA topology")
    count = static_dim_value(cta[0].size)
    if count is None or not 1 <= count <= root.target.device.sm_count:
        raise ValueError(f"P2: root {root.name!r} requires a static CTA extent within device capacity")
    return root.target, Topology("cta", count)


def build_planning_problem(module: Module, root: Function) -> PlanningProblem:
    """Build one deterministic private finite CTA planning problem."""
    target, topology = _validate_entry(module, root)
    return _Planner(module, root, target, topology).build()


__all__ = [
    "BucketRequirement",
    "CandidateBucket",
    "CandidateDependency",
    "OpCandidate",
    "PlanningProblem",
    "RegionCarryInfo",
    "RegionInfo",
    "ValueInfo",
    "build_planning_problem",
]
