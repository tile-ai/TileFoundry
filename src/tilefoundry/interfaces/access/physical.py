"""Carry logical tensor coordinates onto byte addresses in affine layouts."""

import isl

from tilefoundry.ir import dim, expr, types

from .relation import Relation, affine_of, parameter_names


def byte_relation(held: types.TensorType) -> Relation:
    """Map coordinates to every byte touched, retaining holes and layout offsets."""
    if not expr.is_literal_shape(held.shape) or held.dtype.bit_width % 8:
        raise ValueError("byte accesses require static shapes and byte-addressable elements")
    layout, offset, values = held.layout, affine_of(0), {}
    while isinstance(layout, (types.ShardLayout, types.ComposedLayout)):
        if isinstance(layout, types.ShardLayout):
            layout = layout.layout
        else:
            if layout.inner is not None:
                raise ValueError("byte accesses require an affine layout without an inner transform")
            offset = offset.add(affine_of(layout.offset, values=values))
            layout = layout.outer
    if not isinstance(layout, types.Layout):
        raise ValueError("byte accesses require an explicit affine tensor layout")
    flat = types.flatten(layout)
    if not all(expr.static_dim_value(value) is not None for value in (*flat.shape, *flat.strides)):
        raise ValueError("byte accesses require static layout extents and strides")
    positions = dim.layout_to_isl_map(held.shape, layout, {}, divided=lambda _: ())
    positions = positions.intersect_domain(dim.shape_to_isl_set(held.shape, {}))
    coords = ", ".join(f"p{i}" for i in range(len(flat.shape)))
    address = " + ".join(("0", *(f"{stride} * p{i}" for i, stride in enumerate(flat.strides))))
    element = isl.pw_aff(f"{{ [{coords}] -> [({address})] }}").add(offset.insert_dims(isl.dim_type.IN, 0, len(flat.shape)))
    width = held.dtype.bit_width // 8
    byte_map = isl.map(f"{{ [e] -> [b] : {width} * e <= b < {width} * e + {width} }}")
    relation = positions.apply_range(isl.map.from_pw_aff(element)).apply_range(byte_map)
    return Relation(relation, {name: values[name] for name in parameter_names(relation)})


__all__ = ["byte_relation"]
