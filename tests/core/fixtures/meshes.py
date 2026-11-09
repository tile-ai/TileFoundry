from tilefoundry.ir import types

CTA = types.Mesh((types.Topology("cta", 4),), types.Layout((4,), (1,)))
THR = types.Mesh((types.Topology("thread", 384),), types.Layout((384,), (1,)))
CT = types.Mesh(
    (types.Topology("cta", 4), types.Topology("thread", 384)),
    types.Layout(((4,), (384,)), ((1,), (1,))),
)
RUN = types.ComposedLayout(None, 128, types.Layout(((4,), (128,)), ((1,), (1,))))
