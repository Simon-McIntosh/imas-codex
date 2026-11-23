# Coordinator instructions for the ontology migration

Plan section 3 of `iter-plant-signal-ontology` is the authority for what is
delivered. The design is the approved research record
`docs/research/iter-plant-signal-ontology-review.html`; read it in full first.
These instructions add the decisions, the reuse map and the rules.

## Decisions the lead made (2026-11-23)

- The record's proposal is approved as the design: keep both role axes with the
  source-assigned one authoritative and a combined filter; retire
  `CANDIDATE_PLANT_COMPONENT` and `IN_ARCHIVE_CONTROL_NODE` with their scalar
  mirrors; add the ITER-only `ArchiveFieldCode` class and retire the private
  `field_code_meanings` entries; add the first-level control-node binding to the
  plant breakdown; merge `PlantContextDocument` into the shared `Document` with
  optional cataloguing slots and a version-state vocabulary; add the ITER-only
  `PlantConnection` class with its medium vocabulary.
- The plant-breakdown class is renamed `PlantBreakdownNode` (not
  `PlantBreakdownStructure`), with the relationship and slot renames the record
  lists.

## What this section populates, and what it does not

- Migrate every existing ITER node and relationship the decisions touch.
- `ArchiveFieldCode`: populate from the existing field-code table (built from
  graph sources by `iter_codex/discovery/signals/field_codes.py` and
  `field_codes_graph.py`) and the 69 private meanings, then retire the private
  entries' readers. Keep a private copy of the retired entries.
- First-level binding to the plant breakdown: add the slot and populate it only
  where a source already in the graph states the binding (the pilot signal-list
  rows). The other roots are bound when section 4 catalogues their documents;
  say in the evidence which roots are bound and from which source.
- `PlantConnection`: schema only, with no nodes; the import belongs to the
  pipeline plan.
- `Document` merge: migrate the seven plant-context documents; no new documents.

## Folded in from the section 2 review

- The class census missed `DataSource` (2 ITER nodes, one hop from
  `ArchiveEndpoint` through `HOSTED_BY_ARCHIVE`); include it in the before and
  after counts.
- The first-level count must be read live from the graph, not from a fixed list.
- The record's check receipt predated its head; this section's checks are run at
  the head they gate, and their logs are named in the manifest.

## Reuse map

- Schema: `iter_codex/schemas/facility.yaml` (and `common.yaml` for shared
  enums); rebuild models with `build-models --force` in your worktree, and say in
  the manifest that the main checkout needs the same.
- Readers and writers per changed class are listed in the record's section
  "Readers and writers of every changed class"; use it as the starting list and
  extend it with a search of the code and tests.
- Migrations in this repository: follow an existing migration's pattern in
  `iter_codex/graph/` (find one; do not invent a new mechanism).
- The plant search tools (`iter_codex/tools/plant_search.py`) and the shared
  query filters (`iter_codex/graph/domain_queries.py`) are read paths whose
  answers must not change for the fixed twenty signals.

## Rules

- Only ITER-facility nodes and relationships of the classes the record lists
  change. Take a private before-image of everything you change under
  `~/.cache/imas-codex-codac-uda/ontology-migration/`, with the restore command.
- Count every other facility's nodes and relationships, the Data Dictionary's
  and the standard names' before and after; they must be identical.
- Heavy work, the migration itself and graph-marked tests, runs as its own job:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs on GPFS,
  each pytest log ending with pytest's summary and one EXIT line.
- No real signal name, document identity or private text in a tracked file.
- Record suite arms as single-line JSON objects with whole test ids; the
  `tests` field is one literal command that parses.
- Run `ruff check --fix` and `ruff format` on exactly the files you change.
- Commit locally with a conventional subject and a body, and give `commits` as
  a JSON list of full shas. Do not push or merge.
