from dataclasses import dataclass

import pytest

from tests.fixtures.meshes import CT, CTA
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    Predicate,
    WildcardPattern,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import ComposedLayout, DType, Layout


@dataclass(frozen=True)
class _BareLayout(Predicate):
    def holds(self, subject, bindings):
        return isinstance(subject, Layout)


def test_mesh_pattern_matches_the_levels_it_names():
    with pytest.raises(ValueError, match="per-mode predicates"):
        MeshPattern(
            ("thread",),
            ComposedLayoutPattern(
                outer=LayoutPattern(
                    ((128,),),
                    ((1,),),
                    predicates=(P.Forward(),),
                )
            ),
        )

    warpgroup = MeshPattern(
        ("thread",),
        ComposedLayoutPattern(
            offset=WildcardPattern("p0"),
            outer=LayoutPattern(
                ((128,),),
                ((1,),),
                predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
            ),
            predicates=(WildcardPattern("p0") % 128 == 0,),
        ),
    )
    assert warpgroup.match(CT[1:3, 128:256]).captures["p0"] == 128
    assert warpgroup.match(CT[1:3, 64:192]) is None
    both = MeshPattern(
        ("cta", "thread"),
        ComposedLayoutPattern(
            offset=WildcardPattern(),
            outer=LayoutPattern(
                ((2,), (128,)),
                ((1,), (1,)),
                predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
            ),
        ),
    )
    assert both.match(CT[1:3, 128:256]) is not None
    assert both.match(CT[0:1, 128:256]) is None
    assert warpgroup.match(CTA) is None


def test_layout_predicates_read_through_composition():
    pattern = LayoutPattern(
        predicates=(
            P.WholeVectors(
                WildcardPattern("width"),
                "dtype0",
                (4, 8, 16),
            ),
        )
    )
    subject = ComposedLayout(None, 0, Layout(((128, 4),), ((4, 1),)))

    assert pattern.match(subject, {"dtype0": DType.f32}).captures["width"] == 16


def test_composed_layout_predicates_read_the_received_subject():
    pattern = ComposedLayoutPattern(
        inner=None,
        offset=0,
        outer=LayoutPattern(),
        predicates=(_BareLayout(),),
    )
    bare = Layout((4,), (1,))

    assert pattern.match(bare) is not None
    assert pattern.match(ComposedLayout(None, 0, bare)) is None
