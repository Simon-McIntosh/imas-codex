# Brief: blind sheets covering every published name not yet read by two readers

Two independent blind readers are the dependable way to find a name whose definition says more, or less, than the name. 741 published names have been read that way. You prepare the material for the rest. You label nothing.

## Inputs
- Published names: the catalog checkout at `/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names/*.yml` (read-only; one file per physics domain). Use each entry's name, unit, one-line description and full documentation as published.
- Names already read, to exclude: every `name` in `docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json` (381 rows) and every item of `docs/evidence/fragments/sn-name-definition-fresh-labels/sheet-part-{1,2,3}.json` (360 items).
- The draw script and reader briefs to adapt, not to rewrite: `docs/evidence/fragments/sn-name-definition-fresh-labels/sheet-draw.py` and `reader-a-part-1.md` there. Copy the `classes` block of those sheets unchanged.

## Deliver, under `docs/evidence/fragments/sn-name-definition-catalog-reading/`
1. `sheet-part-<nn>.json`, two-digit `nn` from 01: every eligible name exactly once, in parts of at most 110 names, each part mixing domains (shuffle with a recorded seed, do not sort by domain). Same shape as the earlier sheets (`classes`, `items` with `id`, name, unit, description, documentation). Item ids unique across all parts.
2. `sheet-draw.json`: the seed, the catalog revision (`git -C <catalog> rev-parse HEAD`), the eligible count, the count excluded as already read, the part count and the names per part.
3. For every part, two reader briefs `reader-a-part-<nn>.md` and `reader-b-part-<nn>.md`, adapted from the earlier reader brief: each names its own sheet part and its own output file `labels-<a|b>-part-<nn>.json`, keeps the species ruling, the output format and the done-when, and forbids opening `docs/evidence/fragments/sn-definitions-by-construction/`, `docs/evidence/fragments/sn-name-definition-second-reading/`, `docs/evidence/fragments/sn-name-definition-fresh-labels/`, every other file in this directory except its own sheet part, `docs/plans/`, `~/.cache/`, any other worktree, and any tool that returns review results for standard names. A reader brief does not say there is another reader's file to find.
4. The draw script `sheet-draw.py` in the same directory, with a `--check` mode, so the draw can be repeated from the seed.

## Done when
The committed check (its output in your manifest and in a log beside the manifest ending `EXIT=<n>`) prints: the eligible count, as many items as eligible names, as many distinct names, 0 names shared with the 741 already read, no part over 110 names, class keys equal to the earlier sheets', and two reader briefs per part each naming a distinct output file. The check also refuses a copy of the sheets in which one already-read name is substituted, and says so.

## Standing rules
Light work: no graph, no model calls, no job submission, no test run. Export `UV_NO_SYNC=1`; plain `python3` suffices. Commit locally with a conventional subject and a body, staging explicit paths only; no version, plan or ticket labels in file names or comments. Manifest status is `complete`, `blocked` or `failed`, written before any wait.
