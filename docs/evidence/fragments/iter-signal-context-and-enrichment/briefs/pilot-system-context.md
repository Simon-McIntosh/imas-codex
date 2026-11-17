# Coordinator instructions for importing the pilot system's context

The plan section is the authority for what is delivered. These instructions add
the reuse map, the private inputs and the operating rules.

## Reuse map

- Plant structure and component records come through the read-only client in
  `imas_codex/discovery/icp/`. Use its record reads; open no second session.
- The seven pilot documents are already on disk. Read them from the private
  store; do not fetch them again. If one must be re-read from the host, the
  client's document read is the only route, and the request budget is five.
- Spreadsheet rows are read by the project's existing spreadsheet extraction;
  PDF text by the project's existing document extraction.
- Signals are the catalogued archive signals already in the graph, with their
  parsed name parts. Link to them; do not re-import them.
- Graph writes follow the schema: add node and relationship types to the
  LinkML schema and rebuild the models, and scope every node to the ITER
  facility.
- The facility-scoped archive is taken with the project's existing graph
  archive command.

## Private inputs

- Documents, extraction and receipts:
  `~/.cache/imas-codex-codac-uda/pilot-documents/guarded-download/`.
- Document identities, versions and what each holds:
  `~/.cache/imas-codex-codac-uda/pilot-documents/assistant-findings-2026-11-17.md`.
- Plant-structure comparison receipt:
  `~/.cache/imas-codex-codac-uda/icp/pbs-comparison-receipt.json`.
- Hosts and identities are in the gitignored private facility configuration.

## Rules

- ICP and the document host are read-only: GET and HEAD only, enforced in
  code. Never print or store credentials, cookies or tokens.
- No real host, document identity, title, row content, variable name or
  component number goes into a tracked file, test, report or commit message.
  Tests use invented rows and names. Live evidence stays under
  `~/.cache/imas-codex-codac-uda/`.
- The imported content is private. It lives in the graph under the ITER
  facility and in the private archive, never in the repository.
- The operator-reserved archive host is never contacted. This node issues no
  archive data read.
- Heavy work runs as a job of its own:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, with logs on
  GPFS. Graph-marked tests run there too.
- The archive step stops the graph. Before it, confirm no other graph writer
  is running and say so in the manifest; if one is, write the manifest with
  the import complete and the archive outstanding rather than waiting.
- Commit locally with a conventional subject and a body, and give `commits` in
  the manifest as a JSON list of full shas. Do not push or merge.
