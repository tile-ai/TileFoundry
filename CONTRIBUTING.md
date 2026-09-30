# Contributing

CI enforces one thing: the shape of a pull request title
(`.github/workflows/pr-title.yml`). Everything else on this page — the words you
pick, how long a title runs, what a body or an issue says — is what a reviewer
expects to find, not a gate. For the environment, the test commands and the code
principles, read [docs/develop.md](docs/develop.md).

## Titles

One format for commits, issues and pull requests:

```text
<type>(<scope>): <short imperative>
```

- **type** — one lowercase word for the kind of change: what a reader would call
  this work if they saw only the diff. Pick the word that fits; there is no fixed
  list.
- **scope** — the area the change lands in, as this repository names its own
  areas. Name several, comma separated, when a change genuinely spans them, and
  leave the parentheses off when no one area owns it. Append `!` after the
  parentheses for a breaking change.
- **short imperative** — what the change does to the codebase, in the imperative
  and lowercase, stating the outcome rather than the activity. Short enough to
  read at a glance.

```text
fix(analysis): count replicated movement once per mesh position
```

CI checks that shape on the pull request title, and nothing else — not the words
you chose, not the length, and not the commit titles inside your branch.

## Branches

`<type>/<topic>`: the same type word as the title, then a few dashed words naming
what the branch is about — `fix/analysis-replicated-movement`.

## Pull requests

The body is [.github/pull_request_template.md](.github/pull_request_template.md),
which GitHub prefills and which says what belongs in each section.

Push to your fork and open the pull request against `tile-ai/TileFoundry:main`.

```bash
git push -u origin <branch>
gh pr create --repo tile-ai/TileFoundry --base main \
  --head <fork>:<branch> --title "<title>" --body-file <body-file>
gh pr checks --watch --interval 300
```

Do not merge your own pull request. Keep working the CI and the review, and
resolve each comment you have addressed. When `main` moves, rebase, re-verify,
and `--force-with-lease`.

## Issues

[.github/ISSUE_TEMPLATE/report.md](.github/ISSUE_TEMPLATE/report.md), which
GitHub prefills, says what an issue must carry.

## Commits

`git commit` runs pre-commit; fix what it reports and stage again. Use your
global Git identity and add no extra sign-off or co-author trailer.
