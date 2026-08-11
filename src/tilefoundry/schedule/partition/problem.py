"""The closed, target-free constraint input for partition scheduling.

Given what the program states and what the hardware was asked once, this
enumerates every legal placement of every value, every operation that can produce
it, and the Reshard operations needed where no direct edge connects two otherwise
legal choices. Each candidate carries its own already-computed duration and
traffic, so the solve that follows reads numbers instead of rates: after this
stage nothing consults a Target again.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Mapping

from tilefoundry.ir.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    StorageConstraint,
    is_layout_wildcard,
)
from tilefoundry.ir.core import Call, Expr, Op, Tuple, Var, VerifyError, source_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import TensorType, Type, make_shard_tensor_type
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
from tilefoundry.visitor_registry.contexts import (
    Cost,
    CostContext,
    FunctionScope,
    TypeInferContext,
)
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor

from ..errors import ScheduleError
from .facts import PartitionFacts
from .program import (
    OperationSite,
    PartitionProgram,
    RegionInfo,
    ValueInfo,
    ceil_div,
    expr_location,
    tensor_leaves,
)

PlacementRelation = Literal["SAME_INTERVAL", "CONTAINED"]


class PartitionProblemError(ScheduleError):
    """The program and projected facts cannot form a finite partition problem.

    A scheduling failure, and reachable as one: a caller asking this
    layer to schedule something catches what the layer raises, and a
    capability that cannot be scheduled is recorded against that. Sitting
    outside `ScheduleError` made a limit of this algorithm unstateable
    except as a bare `ValueError`, which is also what a caller passing
    nonsense gets -- so the two could not be told apart.
    """


@dataclass(frozen=True)
class CandidateBucket:
    """One value held in one concrete type, and who can produce it that way."""

    value_id: int
    type_id: int
    candidate_ids: tuple[int, ...]
    fixed_offset: int | None
    is_source: bool


@dataclass(frozen=True)
class BucketRequirement:
    """The buckets an authored placement constraint admits for one value."""

    value_id: int
    bucket_ids: tuple[int, ...]
    source: Expr
    metadata: ScheduleConstraintMetadata


@dataclass(frozen=True)
class CandidateDependency:
    """One candidate's demand on one input bucket, and how they must overlap."""

    parent_candidate_id: int
    input_index: int
    child_bucket_id: int
    placement_relation: PlacementRelation | None


@dataclass(frozen=True)
class OpCandidate:
    """One way to run one operation, priced against the projected facts.

    Authored computation and synthesized Reshard are the same concept here: both
    produce buckets from buckets at a stated cost. A Reshard is recognised by
    having no authored site rather than by a separate record type.
    """

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
class PartitionProblem:
    """A complete finite problem with no Target object or callback."""

    module: Module
    root: Function
    topology: Topology
    extent: int
    facts: PartitionFacts
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


def _mesh(count: int, topology: str) -> Mesh:
    return Mesh((Topology(topology, count),), Layout(shape=(count,), strides=(1,)))


def _type_mesh(type: TensorType, fallback: Mesh) -> Mesh:
    if isinstance(type.layout, ShardLayout):
        mesh = type.layout.mesh
        if len(mesh.layout.shape) != 1:
            raise PartitionProblemError(
                "partition requires rank-one candidate meshes"
            )
        return mesh
    return fallback


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
        if actual.mesh.topologies[0].name != topology:
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
        if isinstance(constraint, LayoutConstraint) and not _layout_matches(
            type.layout, constraint
        ):
            return False
        if isinstance(constraint, MeshConstraint):
            if (
                not isinstance(type.layout, ShardLayout)
                or type.layout.mesh != constraint.mesh
            ):
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


class _Closer:
    def __init__(
        self,
        program: PartitionProgram,
        facts: PartitionFacts,
        topology: Topology,
        extent: int,
    ) -> None:
        self.program = program
        self.facts = facts
        self.topology = topology
        self.extent = extent
        self.types: list[Type] = []
        self.type_ids: dict[Type, int] = {}
        self.values: dict[int, ValueInfo] = dict(program.values)
        self.value_types: dict[int, tuple[Type, ...]] = {}
        self.buckets: dict[int, CandidateBucket] = {}
        self._bucket_candidates: dict[int, list[int]] = {}
        self._bucket_by_value_type: dict[tuple[int, int], int] = {}
        self.candidates: dict[int, OpCandidate] = {}
        self.authored_candidates: dict[int, tuple[int, ...]] = {}
        self.dependencies: list[CandidateDependency] = []
        self.requirements: list[BucketRequirement] = []
        self.candidate_enclosing_regions: dict[int, int | None] = {}
        self._next_candidate = 0
        self._next_bucket = 0
        self.counts = self._resource_counts()

    def _resource_counts(self) -> tuple[int, ...]:
        """The parallel-position counts a candidate may divide work over.

        An extent the program mentions contributes each of its divisors up to the
        topology's own extent: dividing a dimension over more positions than the
        level has is not a placement this level can hold.
        """
        result = {1, self.extent}
        for extent in self.program.observed_extents:
            for count in range(1, min(extent, self.extent) + 1):
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
        result: list[TensorType] = [base]
        if isinstance(base.layout, ShardLayout):
            return tuple(result)
        level = self.topology.name
        for count in self.counts:
            if count == 1:
                continue
            meshes = [_mesh(count, level)]
            meshes.extend(
                mesh
                for mesh in self.program.required_meshes
                if mesh.layout.shape == (count,) and mesh not in meshes
            )
            for mesh in meshes:
                replicated = TensorType(
                    shape=base.shape,
                    dtype=base.dtype,
                    storage=base.storage,
                    layout=ShardLayout(
                        layout=Layout(
                            shape=base.shape, strides=try_c_order_strides(base.shape)
                        ),
                        attrs=(Broadcast(),),
                        mesh=mesh,
                    ),
                )
                result.append(replicated)
                result.append(
                    replace(
                        replicated,
                        layout=replace(
                            replicated.layout,
                            attrs=(Partial("sum"),),  # type: ignore[arg-type]
                        ),
                    )
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

    def _init_buckets(self) -> None:
        for value_id, base in self.program.value_base_types.items():
            self.value_types[value_id] = self._legal_types(base)
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
                    is_source=(
                        self.values[value_id].role == "normal"
                        and self.values[value_id].producer_site_id is None
                    ),
                )
                if self.buckets[bucket_id].is_source:
                    source_bucket_ids.append(bucket_id)
            if source_bucket_ids:
                self.values[value_id] = replace(
                    self.values[value_id], source_bucket_ids=tuple(source_bucket_ids)
                )

    def _retag(self, expr: Expr, types: "itertools.chain | object") -> Expr:
        if isinstance(expr, Tuple):
            new_elements = tuple(self._retag(element, types) for element in expr.elements)
            if new_elements == expr.elements:
                return expr
            return replace(expr, elements=new_elements)
        if isinstance(expr.type, TensorType):
            try:
                type = next(types)  # type: ignore[call-overload]
            except StopIteration:
                return expr
            return replace(expr, type=type)
        return expr

    def _candidate_call(
        self, site: OperationSite, input_types: tuple[Type, ...]
    ) -> tuple[Call, tuple[Type, ...]]:
        type_iter = iter(input_types)
        args = tuple(self._retag(arg, type_iter) for arg in site.call.args)
        return replace(site.call, args=args), input_types

    def _active_mesh_for_outputs(
        self, outputs: tuple[TensorType, ...]
    ) -> tuple[Mesh, int]:
        fallback = _mesh(1, self.topology.name)
        meshes = tuple(_type_mesh(output, fallback) for output in outputs)
        mesh = meshes[0] if meshes else fallback
        if any(other != mesh for other in meshes[1:]):
            raise PartitionProblemError(
                "multi-output candidate leaves require one shared Mesh"
            )
        if len(mesh.layout.shape) != 1:
            raise PartitionProblemError("candidate Mesh must be rank one")
        count = mesh.layout.shape[0]
        if not isinstance(count, int) or not 1 <= count <= self.extent:
            raise PartitionProblemError(
                f"candidate Mesh extent {count!r} is outside topology "
                f"{self.topology.name!r} extent {self.extent}"
            )
        return mesh, count

    def _price(
        self, call: Call, cost: Cost, count: int
    ) -> tuple[int, int, int, int]:
        """What one candidate costs, in the units the solve reasons in.

        A Reshard is charged as pure traffic, so its duration follows the
        bandwidth alone. Everything else is charged the worse of its compute and
        its traffic, with compute scaled by how much of the device the candidate
        occupies.
        """
        facts = self.facts
        if isinstance(call.target, Reshard):
            moved = cost.bytes
            duration = ceil_div(
                moved * 1_000_000_000, facts.memory_bandwidth_bytes_per_second
            )
            demand = ceil_div(moved, duration) if duration else 0
            return duration, moved, demand, moved
        if count == 0:
            raise PartitionProblemError(
                "only Reshard candidates may have topology_count=0"
            )
        compute = 0
        for dtype, flops in cost.flops.items():
            compute += ceil_div(
                flops * count * 1_000_000_000 * facts.parallel_units,
                facts.peak_flops(dtype) * count,
            )
        total_bytes = cost.bytes * count
        memory = (
            ceil_div(total_bytes * 1_000_000_000, facts.memory_bandwidth_bytes_per_second)
            if total_bytes
            else 0
        )
        duration = max(compute, memory, 1) if cost.flops or cost.bytes else 0
        demand = ceil_div(total_bytes, duration) if duration else 0
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
        tensor_outputs = tuple(
            type for type in output_types if isinstance(type, TensorType)
        )
        mesh, count = self._active_mesh_for_outputs(tensor_outputs)
        if isinstance(call.target, Reshard):
            count = 0
        duration, total_bytes, demand, moved = self._price(call, cost, count)
        aliases = tuple(
            0 if isinstance(call.target, (Reshape, Transpose)) and index == 0 else None
            for index, _ in enumerate(output_bucket_ids)
        )
        candidate_id = self._next_candidate
        self._next_candidate += 1
        self.candidates[candidate_id] = OpCandidate(
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
        if site_id is None:
            value_id = self.buckets[output_bucket_ids[0]].value_id
            self.candidate_enclosing_regions[candidate_id] = (
                self.program.value_availability_regions.get(value_id)
            )
        else:
            self.candidate_enclosing_regions[candidate_id] = (
                self.program.site_enclosing_regions.get(site_id)
            )
        for bucket_id in output_bucket_ids:
            self._bucket_candidates[bucket_id].append(candidate_id)
        for input_index, bucket_id in enumerate(input_bucket_ids):
            value_type = self.types[self.buckets[bucket_id].type_id]
            relation = (
                _placement_relation(value_type, mesh)
                if isinstance(value_type, TensorType)
                else None
            )
            self.dependencies.append(
                CandidateDependency(candidate_id, input_index, bucket_id, relation)
            )
        return candidate_id

    def _generate_site(self, site: OperationSite) -> None:
        choices: list[int] = []
        for refs in site.input_value_ids:
            choices.extend(refs)
        bucket_options = []
        for value_id in choices:
            bucket_options.append(
                tuple(
                    bucket_id
                    for (candidate_value, _), bucket_id in self._bucket_by_value_type.items()
                    if candidate_value == value_id
                )
            )
        combinations = itertools.product(*bucket_options) if bucket_options else ((),)
        found: list[int] = []
        seen: set[tuple] = set()
        for input_bucket_ids in combinations:
            input_types = tuple(
                self.types[self.buckets[bucket_id].type_id]
                for bucket_id in input_bucket_ids
            )
            candidate_call, selected_types = self._candidate_call(site, input_types)
            scope = FunctionScope(self.program.module, self.program.root)
            ctx = TypeInferContext(scope=scope)
            try:
                output_type = TypeInferVisitor(ctx).visit(candidate_call)
            except (TypeError, ValueError, NotImplementedError, VerifyError, IndexError):
                continue
            output_leaves = tensor_leaves(output_type)
            if len(output_leaves) != len(site.output_value_ids):
                continue
            output_types = tuple(type for _, type in output_leaves)
            try:
                output_bucket_ids = tuple(
                    self._bucket_by_value_type[(value_id, self._intern(type))]
                    for value_id, type in zip(site.output_value_ids, output_types)
                )
                cost_ctx = CostContext(
                    scope=scope,
                    selected_types={
                        id(arg): type
                        for arg, type in zip(candidate_call.args, selected_types)
                    },
                    selected_output_type=output_type,
                    level=self.topology.name,
                    topologies=self.program.module.effective_topologies(),
                )
                cost = CostEvaluator(cost_ctx).visit_Call(candidate_call)
                mesh, count = self._active_mesh_for_outputs(output_types)
            except KeyError:
                continue
            except (TypeError, ValueError) as exc:
                raise PartitionProblemError(
                    f"cost evaluation for {type(site.call.target).__name__} at "
                    f"{expr_location(site.call)} failed: {exc}"
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
            found.append(
                self._add_candidate(
                    site.site_id,
                    candidate_call,
                    tuple(input_bucket_ids),
                    output_bucket_ids,
                    selected_types,
                    output_types,
                    cost,
                )
            )
        if not found:
            raise PartitionProblemError(
                f"operation {type(site.call.target).__name__} at "
                f"{expr_location(site.call)} has no legal candidates"
            )
        self.authored_candidates[site.site_id] = tuple(found)

    def _synthesized_reshards(self) -> None:
        """Connect legal buckets no authored operation can reach directly.

        A bucket that some authored candidate already produces needs nothing: a
        Reshard is introduced only where a legal placement would otherwise have
        no producer at all, and only from a source holding the same logical
        tensor.
        """
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
                    if not isinstance(source, TensorType) or not _same_logical_tensor(
                        source, target
                    ):
                        continue
                    if source == target:
                        continue
                    op = Reshard(layout=target.layout, storage=StorageKind.GMEM)
                    source_expr = self.values[value_id].source
                    metadata = source_metadata(source_expr)
                    source_var = Var(type=source, name="reshard_source", metadata=metadata)
                    call = Call(
                        type=target, target=op, args=(source_var,), metadata=metadata
                    )
                    cost_ctx = CostContext(
                        scope=FunctionScope(self.program.module, self.program.root),
                        selected_types={id(source_var): source},
                        selected_output_type=target,
                        level=self.topology.name,
                        topologies=self.program.module.effective_topologies(),
                    )
                    try:
                        cost = CostEvaluator(cost_ctx).visit_Call(call)
                    except (TypeError, ValueError) as exc:
                        raise PartitionProblemError(
                            f"cost evaluation for synthesized Reshard at "
                            f"{expr_location(source_expr)} failed: {exc}"
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

    def _refs_from_annotation(self, source: Expr) -> tuple[int, ...]:
        for refs, candidate_source, _ in self.program.requirement_annotations:
            if candidate_source is source:
                return refs
        return ()

    def _finish_buckets(self) -> None:
        for bucket_id, bucket in tuple(self.buckets.items()):
            fixed_offset = None
            for _, source, metadata in self.program.requirement_annotations:
                if bucket.value_id not in self._refs_from_annotation(source):
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
        for refs, source, metadata in self.program.requirement_annotations:
            value_id = refs[0]
            matching = tuple(
                bucket_id
                for (candidate_value, type_id), bucket_id in self._bucket_by_value_type.items()
                if candidate_value == value_id
                and isinstance(self.types[type_id], TensorType)
                and _bucket_matches(self.types[type_id], metadata.constraints)
            )
            if not matching:
                raise PartitionProblemError(
                    f"no candidate bucket satisfies where constraint at "
                    f"{expr_location(source)}"
                )
            self.requirements.append(
                BucketRequirement(value_id, matching, source, metadata)
            )

    def _root_connected(self, value_id: int, seen: set[int]) -> bool:
        if value_id in seen:
            return True
        seen.add(value_id)
        if self.values[value_id].role != "normal":
            return True
        bucket_ids = tuple(
            bucket_id
            for (candidate_value, _), bucket_id in self._bucket_by_value_type.items()
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
                if all(
                    self._root_connected(self.buckets[child].value_id, seen)
                    for child in candidate.input_bucket_ids
                ):
                    return True
        return False

    def build(self) -> PartitionProblem:
        self._init_buckets()
        for site in self.program.sites:
            self._generate_site(site)
        self._synthesized_reshards()
        self._finish_buckets()
        for value_id in self.program.root_value_ids:
            self.values[value_id] = replace(
                self.values[value_id], is_final_output=True
            )
            if not self._root_connected(value_id, set()):
                raise PartitionProblemError(
                    f"no legal root-connected candidate path for value {value_id}"
                )
        reshards = sum(
            type(candidate.op) is Reshard and candidate.site_id is None
            for candidate in self.candidates.values()
        )
        return PartitionProblem(
            module=self.program.module,
            root=self.program.root,
            topology=self.topology,
            extent=self.extent,
            facts=self.facts,
            types=tuple(self.types),
            values=MappingProxyType(dict(self.values)),
            buckets=MappingProxyType(dict(self.buckets)),
            candidates=MappingProxyType(dict(self.candidates)),
            authored_candidates=MappingProxyType(dict(self.authored_candidates)),
            dependencies=tuple(self.dependencies),
            requirements=tuple(self.requirements),
            root_value_ids=self.program.root_value_ids,
            regions=MappingProxyType(dict(self.program.regions)),
            candidate_enclosing_regions=MappingProxyType(
                dict(self.candidate_enclosing_regions)
            ),
            value_availability_regions=MappingProxyType(
                dict(self.program.value_availability_regions)
            ),
            site_order=self.program.site_order,
            function_instances=self.program.function_instances,
            diagnostics=(
                f"ops={len(self.program.sites)}",
                f"candidates={len(self.candidates)}",
                f"buckets={len(self.buckets)}",
                f"reshards={reshards}",
            ),
        )


def build_partition_problem(
    program: PartitionProgram, facts: PartitionFacts, topology: Topology
) -> PartitionProblem:
    """Close the problem from an immutable program and already-projected facts."""
    if facts.topology != topology.name:
        raise PartitionProblemError(
            f"partition facts describe {facts.topology!r}, not topology "
            f"{topology.name!r}"
        )
    if facts.parallel_units < 1:
        raise PartitionProblemError(
            "partition facts require at least one parallel unit"
        )
    if facts.memory_bandwidth_bytes_per_second < 1:
        raise PartitionProblemError(
            "partition facts require a positive memory bandwidth"
        )
    if facts.memory_capacity_bytes < 1:
        raise PartitionProblemError(
            "partition facts require a positive memory capacity"
        )
    extent = static_dim_value(topology.size)
    if extent is None or extent < 1:
        raise PartitionProblemError(
            f"topology {topology.name!r} requires a static positive extent, got "
            f"{topology.size!r}"
        )
    if extent > facts.parallel_units:
        raise PartitionProblemError(
            f"topology {topology.name!r} extent {extent} exceeds the "
            f"{facts.parallel_units} parallel units {facts.spec.device_id} states"
        )
    return _Closer(program, facts, topology, extent).build()


__all__ = [
    "BucketRequirement",
    "CandidateBucket",
    "CandidateDependency",
    "OpCandidate",
    "PartitionProblem",
    "PartitionProblemError",
    "build_partition_problem",
]
