# Redraw the coil-current figure with similar categories grouped

## Goal

Redraw the figure from the end-to-end trial so that pulse categories of the
same kind sit together, and produce a version that carries no private
identifier, ready for the lead to decide whether it is committed.

## Why

The trial drew one panel per pulse category in the order the search returned
them. The lead asked for similar categories to be grouped together, and to
see the figure in the repository. The repository is public, so the committed
version must carry no real category, pulse, variable or sensor name.

## Inputs

- The trial's private directory,
  `~/.cache/imas-codex-codac-uda/example-request/resumed/complete/`, holds the
  loaded arrays (`samples.sqlite` and the array exports), `plan.json`, the
  selection file binding aliases to identities, the caption and the figure.
  Read from it; issue no archive read.
- The document assistant's findings on the test kinds are in
  `~/.cache/imas-codex-codac-uda/pilot-documents/assistant-findings-2026-11-17.md`.

## What to do

1. Decide the grouping from the real category names, privately: categories
   whose names show the same kind of test belong together. Record the mapping
   from category to group, and the reason for each, in a private file. Where a
   name does not show the kind, keep that category as its own group and say so.
2. Give each group a plain label that says what kind it is, in words a reader
   of the page understands, without quoting any real name. If the kind is
   inferred from a name and not confirmed by a document, the caption says so.
3. Redraw: one panel per group, time from each pulse's start, current in kA,
   one colour per measurement with the two DC-link candidates distinguishable
   where they differ and drawn as one where they do not, pulses within a group
   distinguishable. Groups with data come first; groups whose windows returned
   no samples are stated in the caption with their counts, not drawn as empty
   panels. Mark timed-out gaps as gaps.
4. Follow the plot style rules in the repository's instructions: no gridlines,
   despined axes, one figure-level legend, labels that name and a caption that
   explains, SI prefixes sized to the numbers, the slide canvas width. Save
   through `imas_codex.plotting`'s save function so the text checks run.
   Render the SVG to PNG and look at it; record the rendered check.
5. Write two files with the same drawing: the private one, in the trial's
   private directory, may keep the measurement tags in its legend; the
   publishable one, at
   `docs/figures/codac-uda-signals/example-request/coil-current-profiles.svg`,
   must carry no real category, pulse, variable, sensor or host name anywhere,
   including inside the SVG's text, metadata and identifiers. Search the
   publishable SVG's bytes for every private identifier in the selection file
   and show none is present.
6. Write the caption for the publishable figure in a short evidence fragment.

## Rules

- No archive read, no graph write, no product source change.
- Real names stay private. The publishable figure and its fragment carry
  aliases and plain labels only.
- Run the plotting as a job on a compute partition if it takes more than a
  minute: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`.

## Done when

The publishable SVG exists at the path above and passes the save function's
checks; the evidence fragment gives its caption, the number of groups, the
number of pulses per group with and without data, and the result of the
identifier search (zero hits, with the number of identifiers searched for);
the private figure and the grouping file exist and their paths are in the
manifest.

Negative control: the identifier search reports a hit when run against the
private figure, which keeps the measurement tags.

## Scope

Write paths: `docs/figures/codac-uda-signals/example-request` and
`docs/evidence/fragments/codac-uda-signals/regroup-the-coil-current-figure.html`.

Commit locally with a conventional subject and a body. Do not push or merge.
