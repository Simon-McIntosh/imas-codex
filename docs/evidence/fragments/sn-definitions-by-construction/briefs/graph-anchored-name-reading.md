# Brief: anchor the name-against-definition reading to the graph

The lead ruled on 2026-11-13: the graph is the single source of truth, no name is published yet (only release candidates exist), and no catalog YAML file is an input. Two blind readers have read 1,975 names, but from the files of an earlier export; a third reading is wanted for the names they could not settle. You re-anchor all of it to the live graph and prepare the third reading. You label nothing and write nothing to the graph.

## What exists
- `docs/evidence/fragments/sn-name-definition-catalog-reading/two-reader-agreement.json`: one row per name read, with both readers' classes and quoted evidence, a verdict (`mismatched`, `agreeing`, `disputed`) and whether the chat seat flagged it. Its names and the text the readers saw came from an export of 2026-11-11, not from the graph as it is now.
- The sheets the readers saw (name, unit, description, documentation per item): `docs/evidence/fragments/sn-name-definition-second-reading/`, `docs/evidence/fragments/sn-name-definition-fresh-labels/sheet-part-*.json`, `docs/evidence/fragments/sn-name-definition-catalog-reading/sheet-part-*.json`.
- The graph property `catalog_commit_sha` marks an export of July and is not a candidate marker. Do not select by it.

## Deliver, under `docs/evidence/fragments/sn-name-definition-graph-reading/`
1. `candidate-set.json`: the names the next export would carry, read from the live graph by the export's own selection (find the selection the release export uses and call it, or run its dry run; do not write a second rule). Record the rule's location, the count, and the graph read time.
2. `reconciliation.json`: every candidate name with one status:
   - `label-carried`: read by both readers, and the graph's name, unit, description and documentation equal what the readers saw (compare after whitespace normalisation only); carry both classes and the verdict;
   - `text-changed`: read, but the graph text differs; record which field differs;
   - `never-read`: a candidate no reader saw;
   and, apart from the candidates, every name the readers saw that is no longer a candidate, with what the graph says of it (absent, superseded onto which name, stage).
   Totals by status, and the verdict counts among `label-carried`.
3. Third-reading sheets `sheet-part-<nn>.json` (at most 110 names each, shuffled with a recorded seed, same shape and `classes` block as the earlier sheets, text taken from the graph): every candidate that is `text-changed` or `never-read`, every `label-carried` name whose verdict is `disputed`, and every `label-carried` name both readers passed that the seat flagged. One reader brief per part, `reader-c-part-<nn>.md`, adapted from `docs/evidence/fragments/sn-name-definition-catalog-reading/reader-a-part-01.md`, naming its own sheet and output `labels-c-part-<nn>.json`, forbidding every earlier label and sheet directory, and not saying other readings exist. Names that are `text-changed` or `never-read` need two readings: give those parts a second brief `reader-d-part-<nn>.md` as well, and put them in parts of their own so a part needs either one reader or two.
4. The script that produced all of it, with a `--check` mode, and a log of that check ending `EXIT=<n>` beside your manifest.

## Done when
The check prints: the candidate count; every candidate in exactly one status; the three status totals summing to the candidate count; the sheets holding exactly the names item 3 lists, each once; no sheet over 110 names; one or two reader briefs per part as item 3 states. The check refuses a reconciliation in which one `text-changed` name is relabelled `label-carried`, and says so.

## Working conditions
- Read the live graph read-only, as one `all_debug` job if the read takes more than a few seconds: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:30:00 --output=<a GPFS log under ~/.cache/imas-codex-graph-reading/>`. A bare `srun --partition` hangs where you start. Bound every query; prove a property exists before believing a zero.
- No catalog YAML file is read. No model calls. No graph write.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in file names or comments.
