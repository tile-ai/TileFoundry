from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, Topology

CTA = Mesh((Topology("cta", 4),), Layout((4,), (1,)))
THR = Mesh((Topology("thread", 384),), Layout((384,), (1,)))
CT = Mesh(
    (Topology("cta", 4), Topology("thread", 384)),
    Layout(((4,), (384,)), ((1,), (1,))),
)
RUN = ComposedLayout(None, 128, Layout(((4,), (128,)), ((1,), (1,))))
