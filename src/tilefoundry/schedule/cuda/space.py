from __future__ import annotations

from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.layout import ComposedLayout

from ..graph import ScheduleGraph
from ..registry import ScheduleContext
from ..space import (
    EdgeKind,
    EdgeOption,
    NodeOption,
    PhysicalRepresentation,
    PlacementOption,
    Resource,
    ScheduleSpace,
)


def _static_cta_size(context: ScheduleContext) -> int:
    if context.level != "cta":
        raise ValueError(f"CUDA CTA backend does not support level {context.level!r}")
    layout = context.mesh.layout
    if isinstance(layout, ComposedLayout):
        raise ValueError("CUDA CTA MVP requires an unsliced parent mesh")
    if len(layout.shape) != 1:
        raise ValueError("CUDA CTA MVP supports only one-dimensional meshes")
    size = layout.shape[0]
    if type(size) is not int or size <= 1:
        raise ValueError("CUDA CTA MVP requires a static parent CTA extent greater than one")
    if context.mesh.topology.name != "cta":
        raise ValueError("CUDA CTA MVP requires a cta topology")
    return size


def _value_bytes(value) -> int:
    ty = getattr(value, "type", None)
    if not isinstance(ty, TensorType):
        return 4
    elements = 1
    for dim in ty.shape:
        if type(dim) is not int:
            return 4
        elements *= dim
    itemsize = {
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "i32": 4,
        "i64": 8,
        "bool": 1,
    }.get(ty.dtype.value, 4)
    return max(1, elements * itemsize)


def _work(node_name: str, value) -> tuple[int, int]:
    payload = _value_bytes(value)
    if "combine" in node_name:
        return payload * 2, payload * 2
    if "route" in node_name or "shared" in node_name or "routed" in node_name:
        return payload * 2, payload * 4
    return payload, payload


def build_cuda_space(graph: ScheduleGraph, context: ScheduleContext) -> ScheduleSpace:
    size = _static_cta_size(context)
    half = size // 2
    full = PlacementOption(
        id=0,
        parent_mesh=context.mesh,
        axis_starts=(0,),
        axis_extents=(size,),
    )
    left = PlacementOption(
        id=1,
        parent_mesh=context.mesh,
        axis_starts=(0,),
        axis_extents=(half,),
    )
    right = PlacementOption(
        id=2,
        parent_mesh=context.mesh,
        axis_starts=(half,),
        axis_extents=(size - half,),
    )

    constrained_storage = {
        graph.value(constraint.target).producer: constraint.storage
        for constraint in graph.constraints
    }
    representations = (
        PhysicalRepresentation(0, StorageKind.GMEM, "parent:gmem"),
        PhysicalRepresentation(1, StorageKind.RMEM, "parent:rmem"),
        PhysicalRepresentation(2, StorageKind.GMEM, "left:gmem"),
        PhysicalRepresentation(3, StorageKind.GMEM, "right:gmem"),
    )
    rep_by_key = {(rep.storage, rep.layout_key): rep.id for rep in representations}

    node_options: list[NodeOption] = []
    options_by_node: dict[int, list[NodeOption]] = {}
    next_option_id = 0
    for node in graph.nodes:
        name = node.callee.name.lower()
        input_reps = tuple(
            1 if "routed" in name else 0 for _ in node.inputs
        )
        output_storage = constrained_storage.get(node.id, StorageKind.GMEM)
        output_rep = rep_by_key[(output_storage, "parent:gmem" if output_storage is StorageKind.GMEM else "parent:rmem")]
        value = graph.value(node.outputs[0]).ir_value
        work_bytes, flops = _work(name, value)

        candidates: list[tuple[str, PlacementOption, int]]
        if "routed" in name:
            lane_rep = output_rep if output_storage is not StorageKind.GMEM else 2
            candidates = [("left", left, lane_rep), ("full", full, output_rep)]
        elif "shared" in name:
            lane_rep = output_rep if output_storage is not StorageKind.GMEM else 3
            candidates = [("right", right, lane_rep), ("full", full, output_rep)]
        else:
            candidates = [("full", full, output_rep)]

        options_for_node: list[NodeOption] = []
        for placement_name, placement, selected_output_rep in candidates:
            option = NodeOption(
                id=next_option_id,
                node=node.id,
                implementation_key=f"{name}:cta:{placement_name}",
                placements=(placement,),
                input_representations=input_reps,
                output_representations=(selected_output_rep,),
                lowering_key=f"{name}:ordinary_call",
                work_bytes=work_bytes,
                flops=flops,
            )
            next_option_id += 1
            options_for_node.append(option)
            node_options.append(option)
        options_by_node[node.id] = options_for_node

    edge_options: list[EdgeOption] = []
    next_edge_id = 0
    for use in graph.uses:
        consumer = graph.node(use.consumer)
        destination_rep = options_by_node[consumer.id][0].input_representations[use.operand_index]
        producer = graph.producer_of(use.value)
        if producer is None:
            source_reps = (0,)
        else:
            source_reps = tuple(
                dict.fromkeys(
                    rep
                    for option in options_by_node[producer.id]
                    for rep in option.output_representations
                )
            )
        payload_bytes = _value_bytes(graph.value(use.value).ir_value)
        for source_rep in source_reps:
            same_placement = (
                source_rep == destination_rep
                and producer is not None
                and "route" not in producer.callee.name.lower()
            )
            if source_rep == destination_rep:
                edge_options.append(
                    EdgeOption(
                        id=next_edge_id,
                        use=use.id,
                        kind=EdgeKind.DIRECT,
                        source_representation=source_rep,
                        destination_representation=destination_rep,
                        same_placement_required=same_placement,
                        payload_bytes=payload_bytes,
                    )
                )
                next_edge_id += 1
                if producer is not None:
                    edge_options.append(
                        EdgeOption(
                            id=next_edge_id,
                            use=use.id,
                            kind=EdgeKind.RESHARD,
                            source_representation=source_rep,
                            destination_representation=destination_rep,
                            same_placement_required=False,
                            payload_bytes=payload_bytes,
                        )
                    )
                    next_edge_id += 1
            else:
                edge_options.append(
                    EdgeOption(
                        id=next_edge_id,
                        use=use.id,
                        kind=EdgeKind.RESHARD,
                        source_representation=source_rep,
                        destination_representation=destination_rep,
                        same_placement_required=False,
                        payload_bytes=payload_bytes,
                    )
                )
                next_edge_id += 1

    return ScheduleSpace(
        node_options=tuple(node_options),
        edge_options=tuple(edge_options),
        resources=(Resource(id=0, name="cta", capacity=size),),
        representations=representations,
    )


__all__ = ["build_cuda_space"]
