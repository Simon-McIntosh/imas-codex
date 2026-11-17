# Coordinator instructions for nodes of the archive plan

The plan section is the authority for what is delivered. These instructions
add the reuse map and the operating rules every node of this plan works under.

## Reuse map

- The archive is read only through the project's existing reader and its module
  environment under `imas_codex/remote/` (`module load
  m-uda-client/7.4-iimkl-2025b`, in one persistent shell). Extend it; do not
  open a second client path or call the archive client directly from new code.
- The catalogue scan and its private store
  (`imas_codex/remote/archive_catalogue.py`, `catalogue_scan.py`) show how a
  resumable metadata listing is run and recorded.
- Variable names are parsed by `imas_codex/discovery/signals/variable_names.py`
  and grouped by `archive_grouping.py`; plant system, function, component
  category and field come from there.
- Signal search and the server's tools live under `imas_codex/llm/` and
  `imas_codex/tools/`; extend an existing tool where it fits and count any tool
  you add.
- Graph writes follow the claim and deadlock rules in
  `imas_codex/discovery/AGENTS.md` and the schema rules in
  `imas_codex/graph/AGENTS.md`. After a schema edit, rebuild the generated
  models in your own worktree and never stage a generated file.
- The installed plotting suite's data-access layer lists pulses with six
  columns (identity, start, end, duration, status, description) and lists
  pulse categories; the survey fragment
  `docs/evidence/fragments/codac-uda-signals/survey-plant-signal-meaning-sources.html`
  records the calls. The pulse description is worth keeping.
- The controls metadata feed's stored response and its measurement are
  described in
  `docs/evidence/fragments/codac-uda-signals/measure-controls-metadata-feed.html`;
  the response itself is private.

## Operating rules

- Archive hosts and cohort settings are in private files under
  `~/.cache/imas-codex-codac-uda/` (`catalogue-private-config.yaml`). The
  archive entry marked with the operator role and excluded is never contacted.
- No real host, category, pulse, variable name, description or title goes into
  a tracked file, test, report or commit message. Tests use invented names;
  real values live in private receipts under `~/.cache/imas-codex-codac-uda/`.
- Listing variables, pulses and categories is metadata. Issue a data read only
  where the section asks for one, and then over a bounded window.
- Anything that lists more than a few hundred items, writes to the graph in
  bulk, or embeds runs as a job on a compute partition: `env -u SLURM_JOB_ID
  -u SLURM_JOBID sbatch --partition=rigel`, with its log on GPFS. Nothing heavy
  runs on the fleet node.
- Every node and link created for ITER reaches the ITER facility node.
- A missing property returns zero rows without an error: assert a positive
  control before reporting any count of zero.
- Write the manifest with what you have before any long wait; give `commits`
  as a JSON list of full shas. Commit locally with a conventional subject and
  a body. Do not push or merge.
