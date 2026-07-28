"""Target-independent extraction of one immutable partition program view.

This walks the authored HIR once and records what is there: every tensor value a
scheduled operation produces or consumes, every operation site, every grid region
and what it carries, and every authored placement constraint. Nothing here
enumerates a choice or reads a machine. The extents observed on the way are kept
raw, because which of them are usable divisors depends on a topology this stage
has not been told about.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Mapping

from tilefoundry.ir.constraints import (
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraintMetadata,
    constraint_metadata,
)
from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var, diagnostic_location
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import Mesh, ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind

from .facts import PartitionFactsQuery

ValueRole = Literal["normal", "carry", "yield", "result"]


class PartitionProgramError(ValueError):
    """The authored program cannot be extracted into a partition program."""


def expr_location(expr: Expr) -> str:
    """The clearest location this expression can name itself by."""
    return diagnostic_location(expr) or getattr(expr, "name", None) or "<unknown>"


def ceil_div(a: int, b: int) -> int:
    """Integer division that rounds away from zero."""
    return (a + b - 1) // b


def tensor_leaves(
    type: Type, path: tuple[int, ...] = ()
) -> tuple[tuple[tuple[int, ...], TensorType], ...]:
    """Every tensor leaf of a possibly nested type, with its index path."""
    if isinstance(type, TensorType):
        return ((path, type),)
    if isinstance(type, TupleType):
        return tuple(
            leaf
            for index, field in enumerate(type.fields)
            for leaf in tensor_leaves(field, path + (index,))
        )
    return ()


@dataclass(frozen=True)
class ValueInfo:
    """One tensor value the partition decides a placement for."""

    source: Expr
    leaf_path: tuple[int, ...]
    is_const: bool
    source_bucket_ids: tuple[int, ...] = ()
    producer_site_id: int | None = None
    function_path: tuple[int, ...] = ()
    is_final_output: bool = False
    role: ValueRole = "normal"


@dataclass(frozen=True)
class OperationSite:
    """One authored call, and the value IDs on each side of it."""

    site_id: int
    call: Call
    function_path: tuple[int, ...]
    input_value_ids: tuple[tuple[int, ...], ...]
    output_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class RegionInfo:
    """One grid region, its trip count, and the sites inside it."""

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
    """One value carried around a region, at each of its four positions."""

    init_value_id: int
    carried_value_id: int
    yield_value_id: int
    result_value_id: int


@dataclass(frozen=True)
class PartitionProgram:
    """What the authored program states, before any choice is enumerated."""

    module: Module
    root: Function
    values: Mapping[int, ValueInfo]
    value_base_types: Mapping[int, TensorType]
    sites: tuple[OperationSite, ...]
    site_order: tuple[int, ...]
    site_enclosing_regions: Mapping[int, int | None]
    value_availability_regions: Mapping[int, int | None]
    regions: Mapping[int, RegionInfo]
    requirement_annotations: tuple[
        tuple[tuple[int, ...], Expr, ScheduleConstraintMetadata], ...
    ]
    function_instances: tuple[tuple[tuple[int, ...], Function], ...]
    root_value_ids: tuple[int, ...]
    observed_extents: tuple[int, ...]
    required_meshes: tuple[Mesh, ...]

    def facts_query(self, topology: str) -> PartitionFactsQuery:
        """Return the explicit query used to project target facts once."""
        return PartitionFactsQuery(topology=topology)


class _Extractor:
    def __init__(self, module: Module, root: Function) -> None:
        self.module = module
        self.root = root
        self.values: dict[int, ValueInfo] = {}
        self.value_base_types: dict[int, TensorType] = {}
        self.regions: dict[int, RegionInfo] = {}
        self.value_availability_regions: dict[int, int | None] = {}
        self.site_enclosing_regions: dict[int, int | None] = {}
        self.sites: list[OperationSite] = []
        self.site_order: list[int] = []
        self.function_instances: list[tuple[tuple[int, ...], Function]] = []
        self.requirement_annotations: list[
            tuple[tuple[int, ...], Expr, ScheduleConstraintMetadata]
        ] = []
        self._expr_values: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
        self._param_values: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
        self._active_functions: set[int] = set()
        self._active_regions: list[int] = []
        self._next_value = 0
        self._next_site = 0
        self._next_region = 0
        self._next_use = 0
        self.required_meshes: list[Mesh] = []
        self.observed_extents = self._observed_extents()

    def _observed_extents(self) -> tuple[int, ...]:
        """Every static extent the program mentions a division by.

        Split axes, mesh shapes, and authored placement constraints all name a
        number of parallel positions. They are collected raw: how many of them a
        candidate may actually use is a property of the topology, which this
        stage does not see.
        """
        extents: set[int] = set()
        seen_functions: set[int] = set()

        def record(dim: object) -> None:
            if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                extents.add(dim)

        def visit_type(type: Type) -> None:
            for _, tensor in tensor_leaves(type):
                if isinstance(tensor.layout, ShardLayout):
                    for attr in tensor.layout.attrs:
                        if isinstance(attr, Split) and attr.axis < len(
                            tensor.layout.layout.shape
                        ):
                            record(tensor.layout.layout.shape[attr.axis])
                    for dim in tensor.layout.mesh.layout.shape:
                        record(dim)

        def visit_expr(expr: Expr | None) -> None:
            if expr is None:
                return
            visit_type(expr.type)
            metadata = constraint_metadata(expr)
            if metadata is not None:
                for constraint in metadata.constraints:
                    if isinstance(constraint, LayoutConstraint):
                        for _, attr in constraint.bindings:
                            if isinstance(attr, Split) and attr.axis < len(
                                constraint.layout.shape
                            ):
                                record(constraint.layout.shape[attr.axis])
                    elif isinstance(constraint, MeshConstraint) and constraint.mesh is not None:
                        self.required_meshes.append(constraint.mesh)
                        for dim in constraint.mesh.layout.shape:
                            record(dim)
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
        return tuple(sorted(extents))

    def _new_value(
        self,
        source: Expr,
        type: TensorType,
        leaf_path: tuple[int, ...],
        function_path: tuple[int, ...],
        *,
        is_const: bool = False,
        producer_site_id: int | None = None,
        role: ValueRole = "normal",
    ) -> int:
        if type.storage is not StorageKind.GMEM:
            loc = diagnostic_location(source) or expr_location(self.root)
            raise PartitionProgramError(
                f"tensor storage {type.storage!r} at {loc} is unsupported; "
                "partitioned tensor values must reside in GMEM"
            )
        value_id = self._next_value
        self._next_value += 1
        self.values[value_id] = ValueInfo(
            source=source,
            leaf_path=leaf_path,
            is_const=is_const,
            producer_site_id=producer_site_id,
            function_path=function_path,
            role=role,
        )
        self.value_availability_regions[value_id] = (
            self._active_regions[-1] if self._active_regions else None
        )
        self.value_base_types[value_id] = type
        return value_id

    def _source_value(self, param: Var, function_path: tuple[int, ...]) -> tuple[int, ...]:
        refs = tuple(
            self._new_value(param, type, path, function_path, is_const=param.is_const)
            for path, type in tensor_leaves(param.type)
        )
        self._param_values[(function_path, id(param))] = refs
        self._record_requirement(refs, param)
        return refs

    def _record_requirement(self, refs: tuple[int, ...], source: Expr) -> None:
        metadata = constraint_metadata(source)
        if metadata is not None:
            for ref in refs:
                self.requirement_annotations.append(((ref,), source, metadata))

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
            helper_env = dict(zip((id(param) for param in target.params), arg_refs))
            self._process_function(target, call_path, helper_env, parent_call=expr)
            refs = self._process_expr(target.body, target, call_path, helper_env)
            self._expr_values[key] = refs
            self._record_requirement(refs, expr)
            return refs
        if isinstance(target, (PrimFunction, Launch, SymbolRef)):
            raise PartitionProgramError(
                f"kernel boundary {type(target).__name__} at "
                f"{expr_location(expr)} is not a partitioned operation"
            )
        if isinstance(target, TupleGetItem):
            source_refs = arg_refs[0] if arg_refs else ()
            fields = tensor_leaves(expr.args[0].type)
            index = target.index
            if index < 0 or index >= len(fields):
                raise PartitionProgramError(
                    f"TupleGetItem index {index} is out of range at "
                    f"{expr_location(expr)}"
                )
            start = sum(
                len(tensor_leaves(field))
                for field in expr.args[0].type.fields[:index]  # type: ignore[union-attr]
            )
            refs = source_refs[
                start : start
                + len(tensor_leaves(expr.args[0].type.fields[index]))  # type: ignore[union-attr]
            ]
            self._expr_values[key] = refs
            self._record_requirement(refs, expr)
            return refs
        site_id = self._next_site
        self._next_site += 1
        output_refs = tuple(
            self._new_value(expr, type, path, function_path, producer_site_id=site_id)
            for path, type in tensor_leaves(expr.type)
        )
        self.sites.append(
            OperationSite(site_id, expr, function_path, arg_refs, output_refs)
        )
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
        context = (
            f"GridRegion at {diagnostic_location(region) or expr_location(function)}"
        )
        if start is None or stop is None or step is None:
            raise PartitionProgramError(
                f"{context} requires static start, stop, and step"
            )
        if start < 0 or step <= 0:
            raise PartitionProgramError(
                f"{context} has invalid start/step ({start}, {step})"
            )
        trip_count = ceil_div(stop - start, step) if stop > start else 0
        if trip_count <= 0:
            raise PartitionProgramError(f"{context} has non-positive trip count")
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
        self._active_regions.append(region_id)
        try:
            carried_ref_groups = tuple(
                tuple(
                    self._new_value(
                        phi, type, path, function_path + (region_id,), role="carry"
                    )
                    for path, type in tensor_leaves(phi.type)
                )
                for phi in region.carried_args
            )
            phi_env = dict(env)
            for phi, refs in zip(region.carried_args, carried_ref_groups):
                phi_env[id(phi)] = refs
            body_refs = self._process_expr(
                region.body, function, function_path + (region_id,), phi_env
            )
            yield_ref_groups = tuple(
                self._process_expr(
                    value, function, function_path + (region_id,), phi_env
                )
                for value in region.yield_values
            )
            yield_refs = tuple(ref for refs in yield_ref_groups for ref in refs)
        finally:
            self._active_regions.pop()
        parent_region_id = info.parent_region_id
        for ref in yield_refs:
            if self.values[ref].role == "normal":
                self.values[ref] = replace(self.values[ref], role="yield")
        if region.carried_args:
            result_refs = tuple(
                self._new_value(
                    self.values[yield_ref].source,
                    self.value_base_types[yield_ref],
                    self.values[yield_ref].leaf_path,
                    function_path,
                    role="result",
                )
                for yield_ref in yield_refs
            )
        else:
            result_refs = body_refs
        for ref in result_refs:
            self.value_availability_regions[ref] = parent_region_id
        carry_infos = tuple(
            RegionCarryInfo(
                init_value_id=init_ref,
                carried_value_id=carried_ref,
                yield_value_id=yield_ref,
                result_value_id=result_ref,
            )
            for init_ref, carried_ref, yield_ref, result_ref in zip(
                init_refs,
                (ref for refs in carried_ref_groups for ref in refs),
                yield_refs,
                result_refs,
            )
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
            raise PartitionProgramError(
                f"helper function {function.name!r} has no body at "
                f"{expr_location(parent_call or function)}"
            )
        if id(function) in self._active_functions:
            raise PartitionProgramError(
                f"recursive helper call to {function.name!r} at "
                f"{expr_location(parent_call or function)}"
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

    def build(self) -> PartitionProgram:
        self._process_function(self.root, (), {}, parent_call=None)
        root_refs = self._expr_values.get(((), id(self.root.body)), ())
        if not root_refs:
            raise PartitionProgramError("root function has no tensor result value")
        return PartitionProgram(
            module=self.module,
            root=self.root,
            values=MappingProxyType(dict(self.values)),
            value_base_types=MappingProxyType(dict(self.value_base_types)),
            sites=tuple(self.sites),
            site_order=tuple(self.site_order),
            site_enclosing_regions=MappingProxyType(dict(self.site_enclosing_regions)),
            value_availability_regions=MappingProxyType(
                dict(self.value_availability_regions)
            ),
            regions=MappingProxyType(dict(self.regions)),
            requirement_annotations=tuple(self.requirement_annotations),
            function_instances=tuple(self.function_instances),
            root_value_ids=tuple(root_refs),
            observed_extents=self.observed_extents,
            required_meshes=tuple(self.required_meshes),
        )


def build_partition_program(module: Module, function: Function) -> PartitionProgram:
    """Extract one deterministic view of what the authored program states."""
    if not isinstance(function, Function):
        raise TypeError(
            f"partition program: root must be a HIR Function, got "
            f"{type(function).__name__}"
        )
    if not module.owns(function, derived=True):
        raise PartitionProgramError(
            f"{function.name!r} is not a function of module {module.name!r}"
        )
    return _Extractor(module, function).build()


__all__ = [
    "OperationSite",
    "PartitionProgram",
    "PartitionProgramError",
    "RegionCarryInfo",
    "RegionInfo",
    "ValueInfo",
    "build_partition_program",
    "ceil_div",
    "expr_location",
    "tensor_leaves",
]
