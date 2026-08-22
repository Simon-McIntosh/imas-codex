# Minimum safety-factor locus definition release candidate

Recorded 2026-08-22 against `imas-standard-names` commit
`1f3d7224549253d492bcd56f21f22628954c735f` and the live codex graph.

## Outcome

The `minimum_safety_factor` locus now says explicitly that safety-factor
**magnitude** selects the flux surface. This preserves the distinction between
the locus and the signed safety-factor value evaluated there.

In
`/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/vocabularies/locus_registry.yml`,
the entry begins at line 655 and its definition at line 658 changed exactly as
follows:

- Before: `The flux-surface location at which the selected safety-factor measure attains its minimum.`
- After: `The flux-surface location at which the safety-factor magnitude attains its minimum.`

The change is commit `1f3d7224549253d492bcd56f21f22628954c735f`.
Release candidate `v0.8.0rc67` points to that commit and was pushed explicitly
to `origin` only. A remote-tag census found the tag on `origin` and no such tag
on `upstream`.

## Codex pins

Both declarations in `pyproject.toml` now pin the same commit:

- line 18, the build-system requirement;
- line 184, the development/catalog-preview requirement.

Both changed from
`6dd6eae9585f4244fe1ae164604d1de278eb82d0` to
`1f3d7224549253d492bcd56f21f22628954c735f`.

`uv.lock` was outside this node's exclusive write fence and was not changed.
After integration, the coordinator must regenerate that derived lock before a
normal codex sync; the validation below used the shared codex environment with
the released ISN checkout first on `PYTHONPATH`.

## Installed-package check

After `uv sync --reinstall-package imas-standard-names` in the canonical ISN
checkout, the installed public package reported:

```text
version=0.8.0rc67
definition=The flux-surface location at which the safety-factor magnitude attains its minimum.
```

The value was read only through the public `get_grammar_context()` API at
`grammar.vocabularies.locus_registry.minimum_safety_factor.definition`.

## Graph snapshot state

Yes: the live graph grammar snapshot now disagrees with the installed package.
A read-only query returned one active `ISNGrammarVersion`, `0.8.0rc66`, while
the installed package is `0.8.0rc67`. The pin bump alone did not synchronize
the graph and this node did not mutate it.

The documented resynchronization command is:

```text
uv run imas-codex sn run --flush
```

`sn run` performs the idempotent grammar synchronization at startup when the
installed version differs from the graph's active version. Run it only after
the new pin and derived lock are integrated and the root environment has been
synchronized.

## Verification

- ISN full suite: `1,995 passed, 30 skipped, 83 xfailed, 0 failed, 0 errors`
  in 255.38 seconds.
- Codex `tests/standard_names`: `6,571 passed, 8 skipped, 286 deselected,
  0 failed, 0 errors` in 205.23 seconds.
- ISN grammar code generation: in sync after regeneration; no generated file
  changed for this definition-only edit.
- ISN private-module audit: `0` imports of an underscore-prefixed
  `imas_standard_names` private module path across all codex Python files.
- Fork CI for the ISN commit: Python 3.12, Python 3.13, lint, formatting,
  grammar-codegen drift, and documentation checks all passed before the release
  tag was created.

Logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091825363367-isnqdef/logs/isn-pytest.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091825363367-isnqdef/logs/codex-standard-names-pytest.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091825363367-isnqdef/logs/isn-sync.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091825363367-isnqdef/logs/isn-context.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091825363367-isnqdef/logs/graph-active-grammar.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091825363367-isnqdef/logs/private-isn-imports.log`
