# Measure the controls metadata feed that MINT reads

## Goal

Find out whether the controls metadata feed the MINT suite is configured to
read gives real descriptions and units for ITER plant variables, and how much
of the archive catalogue it covers.

## Why

The archive describes almost every variable with the same two words, so there
is nothing to search by meaning. The survey of the installed MINT suite found
that one of its configured data sources names a controls metadata server, read
with a single GET of one route that returns, per variable, a name, a
description, a unit and a type. If those descriptions are real, they are the
fastest route to finding signals by meaning, ahead of any language-model
enrichment. The survey read code only and did not contact the feed.

## What exists

- The survey's fragment,
  `docs/evidence/fragments/codac-uda-signals/survey-plant-signal-meaning-sources.html`,
  names the functions and configuration keys. Its private notes,
  `~/.cache/imas-codex-codac-uda/meaning-sources/survey-private-notes.json`,
  hold the configured source entries.
- The private archive catalogue,
  `~/.cache/imas-codex-codac-uda/full-census/catalogue.sqlite`, holds every
  archive variable name. Open it read-only.

## What to do

1. From the installed MINT configuration, read the address under the controls
   metadata key and the route the client uses. Before any request, confirm the
   host is not the operator-reserved archive host named in the project's
   private configuration; if it is, or if you cannot tell, stop and report.
2. Make the same request the installed client makes: one GET of that one
   route, with the client's own timeout. No other route, no other method, no
   retry loop beyond two attempts a minute apart, no credentials sent unless
   the installed client sends them, and then only as it does.
3. Store the response privately under
   `~/.cache/imas-codex-codac-uda/meaning-sources/`, with its size and digest.
4. Measure, offline, from the stored response:
   - the number of entries, and how many carry a non-empty description, unit
     and type;
   - the number of distinct descriptions and the share taken by the ten most
     common, so a boilerplate description shows up as what it is;
   - description length as a distribution;
   - how many entry names match an archive catalogue name exactly, and after
     the normalisations the two naming forms plausibly differ by, each counted
     separately and named;
   - coverage by plant system (the first field of the name), as a table of
     counts, and for the magnet cold test facility's variables specifically;
   - for thirty entries chosen by a fixed hash, a hand reading of whether the
     description says what the variable is, recorded privately with a count of
     informative, generic and wrong.
5. State whether the feed is refreshed and versioned, if the response says.

## Rules

- Read only: GET, once, of the one route. Nothing is written to the graph and
  no product source changes.
- No real host, variable name or description goes into a tracked file, test,
  report or commit message. The fragment gives counts, shapes and invented
  examples; real values stay in the private notes.
- An empty or failed response is reported as that, with the error, never as
  zero coverage.
- The catalogue comparison reads more than a million names: run it as a job on
  a compute partition, `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch
  --partition=rigel`, with its log on GPFS. The single GET may run where you
  are.

## Done when

The evidence fragment gives every count above, a plain statement of whether
the feed's descriptions are good enough to search by, and what share of the
archive catalogue and of the cold test facility's variables they would cover.
One figure shows coverage by plant system, saved through
`imas_codex.plotting`'s save function, rendered and looked at.

Negative control: a response with no entries, and a response that is not the
expected shape, are each reported as a failure of the measurement with the
cause, shown by a test of the offline measurement code on invented inputs.

## Scope

Write paths: `scripts/` (one measurement script), `tests/discovery/` (its
test), the evidence fragment
`docs/evidence/fragments/codac-uda-signals/measure-controls-metadata-feed.html`
and the figure directory
`docs/figures/codac-uda-signals/measure-controls-metadata-feed`.

Commit locally with a conventional subject and a body. Do not push or merge.
