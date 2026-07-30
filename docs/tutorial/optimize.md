# Step two: make it fast

This is the page about *how to work*, not about which flags exist. The order and
the granularity are the whole content; every command mentioned has its own
`--help`.

## The constraint that sets the granularity

`@runtime_module` checks, at decoration time, that the twin's function names and
child names are exactly the authored Module's. Not a subset — equal.

So **there is no half a twin.** You cannot implement one kernel of a Module and
leave the rest for later, which means the unit you extend by is a Module, never a
function. Pick your Module boundaries in step one with that in mind: a boundary is
a thing somebody will have to implement all of, at once.

## The loop

```python
def optimize(module):
    # Leaves first: a parent's numbers are only meaningful once the children it
    # calls are trusted, and a leaf is the smallest thing with a reference.
    for leaf in sorted(leaves_of(module), key=how_easily_judged):

        # Inner loop: write the twin, compare, fix what disagrees. The reference
        # is the authored leaf run through the evaluator, so the comparison is
        # available from the first line of the twin onwards.
        while True:
            report = check(twin=your_twin_of(leaf), reference=leaf, inputs=...)
            if report.passed:
                break
            fix(report)                       # the report names the output and the predicate

        # A dimension nobody bound was pinned to run at all. That pin is a
        # decision, and it is reported precisely so it can become a real one.
        if report.unresolved_dimvars:
            specialize(leaf, over=report.unresolved_dimvars)
            # And then check that dispatch lands where you think: run several
            # extents across the boundary and read back which variant each chose.
            check(twin=..., reference=..., dims={name: [several, extents]})

    # Assemble upwards. Every child is trusted before its parent is measured, so a
    # parent that disagrees has exactly one new suspect: the parent.
    for parent in bottom_up(module):
        check(twin=your_twin_of(parent), reference=parent, inputs=...)

    # Fusion is the same loop at a coarser boundary. Rewrite the source so two
    # neighbours become one Module -- your copy, since the authored one is the
    # reference and must not move -- and optimise that.
    coarse = your_copy_of(module.source, fusing=adjacent_boundaries)
    optimize(coarse)
```

When to stop is yours to judge. Nothing in here decides that the numbers are good
enough; it only makes sure that whatever you decide, you decided it against a
reference and not against the last thing you happened to measure.

## Where your copy comes from

`your_copy_of` is a command:

```sh
tilefoundry models qwen3_5_35b_a3b --source > mine.py
```

Then edit `mine.py` — merge two of a Module's functions into one, move a boundary,
whatever the fusion is — and point the tools at it:

```sh
tilefoundry check mine.py:MyFused.fused --inputs random --out output \
    --fn allclose --atol 1e-6 --rtol 1e-6
```

A target is `file:Class[.child...][.function]`, and that file is any file you can
read. Nothing in the command surface knows or cares that the shipped models exist,
so a fused copy is not a special case: it is the ordinary case pointed at your file.

You do not have to protect the original. What `--source` prints lives in the
installation, and an installation is not where you edit — so the reference stays the
reference because of where it sits, not because anything stops you. Keeping it
intact is the point: the moment you fuse, that copy is the only thing that still
says what the answer was supposed to be.

## Siblings are independent; a parent is not

There is no dependency edge between siblings. Two leaves under one parent neither
call nor constrain each other, so they can be worked on — and checked — in parallel,
by different people or in any order you like.

The parent is the opposite: it depends on **every** child. Its numbers only mean
something once all of them are trusted, which is why the loop finishes the whole
row of leaves before it measures the thing above them.

## Why in that order

- **Leaves before parents**, and *easiest to judge* before hardest, because a
  failing comparison should have one suspect. If you implement a parent first, a
  disagreement could be the parent or any child under it.
- **Compare before you specialize.** A run that was pinned to one extent is a real
  run and may already be right; specializing first commits you to variants you have
  no evidence you need.
- **Verify dispatch after you specialize**, at several extents. Variants are chosen
  by size, and a boundary that picks the wrong implementation still produces a
  plausible number.
- **Fuse last, and by rewriting source.** Fusion changes the boundaries, so it
  invalidates the reference you were using; the way through is a new copy whose
  reference is the old one, which is the recursion above.
