# TileFoundry in two steps

The work is two steps, and the second one is a loop.

**Step one — describe the model.** Write the published model as authored HIR: one
`Module` per boundary somebody will implement, its kernels as `@func`s, its
weights as `ConstTensor` parameters. What you are building is a *reference* — the
thing every later answer is measured against. It is finished when it agrees with
the published implementation on real weights.

**Step two — make it fast.** Write a runtime twin of one Module, compare it to the
reference the first step produced, and keep going while the numbers disagree. Then
take the next Module. Then fuse neighbours and repeat.

## Why the source is the reference

TileFoundry is **source to source**. The Module you author is source code; the fast
implementation is also source code; both are readable, and each is checkable
against the other. Nothing is hidden behind an opaque graph, so the two can be put
side by side and asked whether they compute the same thing.

That is what makes this way of working available at all: pin one point, optimise it
against a fixed reference, and spread outwards. Without a reference that is itself
readable source, "is it still correct?" has no cheap answer, and the loop stops
being a loop.

## Where the other answers are

This tutorial teaches the **workflow** — what to do, in what order, at what
granularity. It deliberately does not restate reference material that already has a
home:

- `tilefoundry spec <topic>` — the normative contracts: the IR, the parser, the
  runtime, the target. When a page here says "the contract is X", the spec is where
  X is stated.
- `tilefoundry check --help` — the comparison predicates and their bounds, with the
  arithmetic for choosing a tolerance.
- `tilefoundry models` — the models already described, and their authored source to
  copy from.
