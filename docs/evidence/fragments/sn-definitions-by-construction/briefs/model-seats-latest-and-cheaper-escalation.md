# Brief: every model seat on its provider's latest version, and a cheaper escalation seat

Lead instruction, verbatim: "cheaper escalation - ensure that all of our models are set to the providers latest version" and "this is managed in pyproject". A live rewrite of several hundred names is stopped until this lands.

## Why the escalation seat changes
The documentation refine chain runs its last rotation on `[tool.imas-codex.sn-escalation]`. That seat is `openrouter/anthropic/claude-fable-5`, cataloged at USD 10 / 50 per million tokens, five times the first-draft seat. In the live rewrite 29 of 78 family parents cost over USD 0.30 each and three cost USD 1.26, almost all of it this one call (`docs/evidence/fragments/sn-definitions-by-construction/live-rewrite-cost-diagnosis.json`).

## Deliver
1. **Escalation seat.** `[tool.imas-codex.sn-escalation]` moves to the latest Anthropic Sonnet-line model on OpenRouter (today's catalog holds `openrouter/anthropic/claude-sonnet-5` at USD 2 / 10). It stays a different vendor from the refine seat and the compose seat, which is the reason the seat exists; keep that sentence of the comment true.
2. **Every other seat on its provider's latest version.** Cover every model identifier `pyproject.toml` names under `[tool.imas-codex.*]`: single seats, review panels and profiles, benchmark lists, fan-out, and the pricing catalog. For each, the target is the newest generally available model of the same provider and the same line (Sonnet stays Sonnet, Haiku stays Haiku, a flash model stays a flash model); a preview is kept only where its line has no generally available successor. Never move a seat to a different tier or vendor except the escalation seat above. Establish "latest" from records, not memory:
   - OpenRouter seats: the public model list, `https://openrouter.ai/api/v1/models` (no key, send nothing about this project). Record the list's fetch time.
   - Local seats (`local/...`, route `ambix-local`): the router's own `/v1/models`. Keep the seat on what the router serves.
   - Jev seats (`typesafe/...`): the provider's model list through the key the project already uses. Keep the seat if no newer version is listed.
3. **Prices.** A seat without a cataloged price refuses to run. Every identifier a seat names after your change has an entry under `[tool.imas-codex.llm.openrouter-pricing]` with prompt, completion and cache prices, the input limit, tier overrides where the provider has them, its `source` and today's `verified-at`, all read from the provider's record. Entries for models no seat or list names any longer are removed.
4. **Identifiers written in source.** Model identifiers also sit in Python defaults and fallbacks under `imas_codex/` (for example `claude-sonnet-4.6`, `claude-opus-4.6`, `gemini-2.0-flash-001`, `local/deepseek-v4-flash`). List each with its file and what reads it. Where it is a default that duplicates a pyproject seat, make the code read the seat; where it is a genuine fallback, bring it to the same latest version. Comments that name a switch-back model are brought up to date or removed.
5. **A before-and-after table** in `docs/evidence/fragments/sn-definitions-by-construction/model-seats-latest.json` and at the top of your manifest: seat, model before, model after, price before and after, and the record that shows the after is the latest. Seats that did not move are listed with the record showing they are current.

## Done when
- `uv run python -c` over the settings accessors resolves every seat to its new identifier with a non-null price, shown in the gate log.
- One smallest-possible completion per changed OpenRouter identifier returns a completion and a recorded cost (not just a 200), in the gate log. Total spend under USD 1; report it.
- Tests that read seats or prices pass: `tests/config`, `tests/llm`, and the files under `tests/standard_names` that import `settings`, the budget module or the refine workers (find them by grep, name them in the manifest). A test that pins an old identifier is updated to the seat it tests, not deleted.
- The escalation path is shown choosing the new seat: an existing or new test in which the last refine rotation resolves the escalation model from the seat, failing when the seat is unset.

## Working conditions
- Write paths: `pyproject.toml`, `uv.lock` only if a dependency truly changes (none is expected), the Python files that hold identifiers, their tests, and the evidence file above. If a file you must edit is outside your write paths, edit it and record it in `changed_paths` with a `scope_note`.
- No live graph write. No pipeline run.
- Tests run as a job, not on the node you start on: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:50:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`. A bare `srun --partition` hangs where you start.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Never print or copy the contents of `.env`. Never kill by pattern.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
