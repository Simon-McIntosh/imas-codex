# Brief: the equivalent-electron renames (sn-definitions-by-construction §6)

The first rewrite listed 56 names as waiting on the grammar word `equivalent_electron`. The refine pass found the word is already a registered qualifier in the grammar the pipeline imports (`Qualifier.EQUIVALENT_ELECTRON`, tagged `v0.12.0rc1`), and renamed three of them through the signed path (`accumulated_ammonia_count`, `accumulated_argon_count`, `accumulated_neon_count`). This node renames the rest. It is resumable from a progress file.

## State at restart (2026-11-17)
An earlier run of this brief stopped at its probe: eight names are renamed and live, four of them reviewed and not accepted, forty-eight untouched, USD 0.54 spent. The cause was the name review judging the qualifier against the source leaf only; that is repaired on main (the name review reads the source's enclosing structures), and all eight renamed names now answer both source-support checks at 0.0.
- Start from the live graph, not from the earlier list: re-read every one of the fifty-six.
- The earlier run's progress file is `~/.cache/imas-codex-live-rewrite/equivalent-electron-progress.jsonl`; keep appending to it. Its report builder, check script, data file and page are kept at `~/.cache/imas-codex-live-rewrite/equivalent-electron-first-run/` (`equivalent-electron-renames.py`, `check_equivalent_electron_renames.py`, `.json`, `.html`); copy them into your tree and build on them rather than writing new ones.
- First have the four renamed-but-not-accepted names reviewed again; if any is still refused for source support of `equivalent_electron`, stop and report at once: that is a failure of the review repair, not a name to list.
- The section's measure now requires every renamed name, the eight and the forty-eight, to be accepted on name or listed with the check that holds it. Spend counts from the start of the earlier run: USD 8 in total.
- Run scratch and progress files under `~/.cache/imas-codex-live-rewrite/` are yours to write; your declared write paths bound only what you commit. Manifest lists are comma-separated.

## Precondition (check first; stop and report if it fails)
You are the only graph writer: no `sn run`, `sn edit` or signed apply of this account in `squeue` or the process table. Say what you checked. The grammar the pipeline imports parses a probe name carrying `equivalent_electron`; show the parse.

## Population
`docs/evidence/fragments/sn-definitions-by-construction/live-rewrite-bulk.json`, the `waiting_equivalent_electron` list (56 names with the counting basis each states). Re-read each from the live graph first: a name already renamed (see `held-names-refine-pass.json`, `renames` and `grammar_route`) or no longer live is recorded as settled with what settled it.

## Ruling (recorded on the plan; do not re-decide)
A count or rate stated in equivalent electrons is named with the word `equivalent_electron` next to the base, after `prefill`: `accumulated_argon_equivalent_electron_count`, `neon_prefill_equivalent_electron_count`, `lithium_equivalent_electron_source_rate_due_to_gas_injection`. Keep any qualifier the definition and the Data Dictionary source support (for example `due_to_gas_injection`); add none they do not. Where the source is not in equivalent electrons after all, decline the rename with the evidence and leave the name.

## Write path
Renames go through the signed rename path, one at a time, so sources, reviews and paid history are kept; read `.claude/skills/sn-edit` first and dry-run once. Never write name, description or documentation through Cypher. A rename that collides with a live identity is not forced: list it with the colliding name for a fold decision.

## How to run it
- Probe five names first; stop and report if they average over USD 0.20 a name or fewer than three land.
- Then serial batches as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`. A bare `srun --partition` hangs where you start.
- Progress file `~/.cache/imas-codex-live-rewrite/equivalent-electron-progress.jsonl`: one line per name when submitted and one when it settles, with old and new identity, description, unit, source bindings and review scores before and after, the outcome and the spend from the pipeline's own cost record (`LLMCost` rows joined to each edit's run id, each row counted once).
- Commit your evidence before you write `status: complete` in the manifest: the supervisor stops a run the moment its manifest reads complete.

## Limits
- Spend ceiling USD 8.
- Stop and report at once if a signed apply is refused for drift, if a documentation review fails with an error rather than a verdict, or if live names without a source (`imas-codex sn sourceless`) rise above 10.
- Never kill by pattern; list pids and signal those.

## Deliver, under `docs/evidence/fragments/sn-definitions-by-construction/`
`equivalent-electron-renames.json` and `equivalent-electron-renames.html`: every one of the 56 with old identity, new identity or the reason it was left, scores before and after, source bindings before and after (they must be carried over), spend; a read-only check script that re-reads the live identities and exits non-zero on a mismatch, with one red run against an altered count, both logs ending `EXIT=<n>`.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Status is `complete`, `blocked` or `failed`, nothing else; one field per line; commits as bare hashes. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
