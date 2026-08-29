# TileFoundry in two steps

**Step one — describe the model.** Write the published model as authored HIR: one
`Module` per boundary somebody will implement, its kernels as `@func`s, its weights
as `ConstTensor` parameters. That is the *reference*. It is finished when it agrees
with the published implementation on real weights.

**Step two — make it fast.** The authored HIR stays the reference. Write a runtime
twin beside it and `check` the two against each other. `analyze` reports what the
program costs; you decide what to do with that, including changing the authored
HIR.

TileFoundry is **source to source**. The reference is source code, the fast
implementation is source code, and either can be pointed at any command.

## Where the other answers are

- `tilefoundry spec <topic>` — the normative contracts: the IR, the parser, the
  runtime, the target.
- `tilefoundry check --help` — the comparison predicates and their bounds, with the
  arithmetic for choosing a tolerance.
- `tilefoundry analyze --help` — flops, traffic, roofline bounds, a predicted time.
- `tilefoundry models` — the models already described, and their authored source to
  copy from.
- `tilefoundry tutorial orchestrator causal_lm` — the shipped autoregressive decode
  loop.
