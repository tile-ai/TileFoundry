import pytest

from tests.fixtures.meshes import CT, CTA
from tilefoundry.ir.pattern import (
    CapturePattern,
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    MultipleOfPattern,
    OneOfPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import ComposedLayout, DType, Layout


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
            offset=CapturePattern("p0", MultipleOfPattern(128)),
            outer=LayoutPattern(
                ((128,),),
                ((1,),),
                predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
            ),
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
                CapturePattern("width", OneOfPattern((4, 8, 16))),
                "dtype0",
                (4, 8, 16),
            ),
        )
    )
    subject = ComposedLayout(None, 0, Layout(((128, 4),), ((4, 1),)))

    assert pattern.match(subject, {"dtype0": DType.f32}).captures["width"] == 16
