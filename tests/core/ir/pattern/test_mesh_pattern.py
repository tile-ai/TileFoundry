from dataclasses import dataclass

import pytest

from tests.core.fixtures.meshes import CT, CTA
from tilefoundry.ir import expr, pattern, types


@dataclass(frozen=True)
class _BareLayout(pattern.Predicate):
    def holds(self, subject, bindings):
        return isinstance(subject, types.Layout)


def test_mesh_pattern_matches_the_levels_it_names():
    with pytest.raises(ValueError, match="per-mode predicates"):
        pattern.Mesh(
            ("thread",),
            pattern.ComposedLayout(
                outer=pattern.Layout(
                    ((128,),),
                    ((1,),),
                    predicates=(pattern.Forward(),),
                )
            ),
        )

    warpgroup = pattern.Mesh(
        ("thread",),
        pattern.ComposedLayout(
            offset=pattern.Wildcard("p0"),
            outer=pattern.Layout(
                ((128,),),
                ((1,),),
                predicates=(pattern.Forward(per_mode=True), pattern.Injective(per_mode=True)),
            ),
            predicates=(pattern.Wildcard("p0") % 128 == 0,),
        ),
    )
    assert warpgroup.match(CT[1:3, 128:256]).captures["p0"] == 128
    assert warpgroup.match(CT[1:3, 64:192]) is None
    both = pattern.Mesh(
        ("cta", "thread"),
        pattern.ComposedLayout(
            offset=pattern.Wildcard(),
            outer=pattern.Layout(
                ((2,), (128,)),
                ((1,), (1,)),
                predicates=(pattern.Forward(per_mode=True), pattern.Injective(per_mode=True)),
            ),
        ),
    )
    assert both.match(CT[1:3, 128:256]) is not None
    assert both.match(CT[0:1, 128:256]) is None
    assert warpgroup.match(CTA) is None


def test_layout_predicates_read_through_composition():
    declared = pattern.Layout(
        predicates=(
            pattern.WholeVectors(
                pattern.Wildcard("width"),
                "dtype0",
                (4, 8, 16),
            ),
        )
    )
    subject = types.ComposedLayout(None, 0, types.Layout(((128, 4),), ((4, 1),)))

    assert declared.match(subject, {"dtype0": expr.DType.f32}).captures["width"] == 16


def test_composed_layout_predicates_read_the_received_subject():
    declared = pattern.ComposedLayout(
        inner=None,
        offset=0,
        outer=pattern.Layout(),
        predicates=(_BareLayout(),),
    )
    bare = types.Layout((4,), (1,))

    assert declared.match(bare) is not None
    assert declared.match(types.ComposedLayout(None, 0, bare)) is None
