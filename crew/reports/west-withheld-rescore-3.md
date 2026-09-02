NEEDS-HELP: the exact batch review route processed zero of the ten successfully staged rescores, so none received a fresh quorum draw.

tried: Read the live plan at version 39, verified the identity-by-name fix is present at `3d28224e`, ran exactly one `sn rescore <name> --stage-only` for each of the ten names, then ran exactly one `sn run --batch west_production_dd_paths --only review_name --cost-limit 10`. All eleven mutations exited 0. The batch run finalized as `no_eligible_work`, with 0 reviews, 0 LLM calls, and $0.000000 spend.

options: (1) run one exact-name review drain over the ten already-drafted identities using repeated `--name`; (2) change batch scoping so the WEST manifest can bind the ten names through their current source provenance, then repeat the governed stage-and-drain route; (3) amend the recovery contract to use inline `sn rescore` so each rescore's own run id scopes its fresh quorum draw.

leaning: Option 1 is the smallest governed recovery. The ten identities are already `drafted / pending / valid`; an exact-name scope would select the intended names without changing identity or bypassing review. The present batch manifest has zero source overlap with all ten current name bindings, so repeating the same batch command would again be empty.

cost-if-wrong: An incorrect exact-name scope can review unintended drafted names or miss members; it must atomically preflight all ten. A batch-scoping code change requires focused tests and another ten-name stage-and-review cycle. Inline rescore would spend up to its per-name limits and no longer provide the single shared $10 ceiling requested here.

# WEST withheld rescore rerun

## Outcome

The exact operational sequence was executed, but the node is **blocked** because the named batch route did not apply a fresh quorum draw.

- Rescore staging: **10/10** commands succeeded, each changing `reviewed -> drafted` and clearing the prior score.
- Batch review: **0/10** names reviewed and **0/10** name-accepted. The only authorized drain exited 0 with `stop_reason=no_eligible_work`.
- Spend: **$0.000000 / $10.000000**. The receipt has 0 LLM calls; both `SNRun.cost_spent` and the `LLMCost.llm_cost` sum are exactly zero.
- Final target state: all ten are `drafted / pending / valid`; no fresh quorum score exists.
- Final graph census: **4,691** `StandardName` candidates, **4,691** with `id`, **4,691** with `name_stage`, **2,335** accepted, **0 approved**, and **0 contested**.
- No direct acceptance, direct graph-text mutation, catalog mutation, plan/index write, or source-code edit was performed.

## Command ledger

Every project Python or CLI command used the shared environment with `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv`, `PYTHONPATH=$PWD`, and `uv run --no-sync`.

| Command | Exit | SNRun or audit id | Exact spend | Result |
|---|---:|---|---:|---|
| Read the complete live plan HTML | 0 | none | $0.000000 | Version 39 records the identity-by-name acceptance fix and authorizes this rerun. |
| `pwd`; `git rev-parse HEAD`; `git status --short` | 0 | none | $0.000000 | Correct detached worktree at base `0e103f86476fdd74ae2be521d94dc38d08e0fbfc`; clean at orientation. |
| Read `reckon-ship` and scoped Standard Names instructions; inspect the prior recovery report and fix history | 0 | none | $0.000000 | Target list and route confirmed; `3d28224e` is present. |
| `sn rescore --help`; `sn run --help` | 0, 0 | none | $0.000000 | Confirmed `--stage-only`, `--batch`, `--only review_name`, and `--cost-limit`. |
| Initial GraphClient lifecycle/census read | 1 | none | $0.000000 | Read query's census string lost its status literals to shell quoting; no write occurred and no evidence was emitted. |
| Corrected initial GraphClient lifecycle/census read | 0 | none | $0.000000 | All 10 targets found at `reviewed / pending / valid`; key-coverage control 4,691/4,691; approved 0; contested 0. |
| `sn rescore beryllium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164538Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore carbon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164554Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore deuterium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164609Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore helium_4_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164624Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore hydrogen_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164635Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore lithium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164650Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore neon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164703Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore poloidal_magnetic_flux --stage-only` | 0 | `sn-rescore-20260902T164719Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore tungsten_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164733Z` | $0.000000 | `reviewed -> drafted`. |
| `sn rescore xenon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T164747Z` | $0.000000 | `reviewed -> drafted`. |
| `sn run --batch west_production_dd_paths --only review_name --cost-limit 10` | 0 | `26479cb7-feb9-4a35-89e5-4a99f0988848` | **$0.000000 exact** | `no_eligible_work`; 0 names reviewed; 0 LLM calls; elapsed 352.945649 s. |
| Final GraphClient lifecycle, receipt, cost, and census reads | 0 | none | $0.000000 | Verified the ten drafted states, exact zero-spend receipt, accepted 2,335, approved 0, contested 0. |
| Read-only manifest/source overlap diagnostic | 0 | none | $0.000000 | WEST manifest has 355 paths; each of the 10 targets has 0 currently bound sources in that set. |
| Auto-log `stat` | 0 | none | $0.000000 | `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`, 1,992,390 bytes at 2026-09-02 18:55:03 +0200. |

The initial GraphClient read failure was corrected once. It was a read-only quoting error, not a graph or pipeline failure. No command was retried after two distinct failed fixes.

## Per-name lifecycle and score evidence

Triplets are `name_stage / docs_stage / validation_status`. “Prior score” is the stored score immediately before staging. “Fresh score” is the post-run quorum result; all are null because the one authorized review run made zero calls.

| Standard name | Rescore run id | Before | Prior score | After | Fresh quorum score |
|---|---|---|---:|---|---|
| `beryllium_density_at_plasma_boundary` | `sn-rescore-20260902T164538Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `carbon_density_at_plasma_boundary` | `sn-rescore-20260902T164554Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `deuterium_density_at_plasma_boundary` | `sn-rescore-20260902T164609Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `helium_4_density_at_plasma_boundary` | `sn-rescore-20260902T164624Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `hydrogen_density_at_plasma_boundary` | `sn-rescore-20260902T164635Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `lithium_density_at_plasma_boundary` | `sn-rescore-20260902T164650Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `neon_density_at_plasma_boundary` | `sn-rescore-20260902T164703Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `poloidal_magnetic_flux` | `sn-rescore-20260902T164719Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `tungsten_density_at_plasma_boundary` | `sn-rescore-20260902T164733Z` | reviewed / pending / valid | 1.00000 | drafted / pending / valid | not produced (null) |
| `xenon_density_at_plasma_boundary` | `sn-rescore-20260902T164747Z` | reviewed / pending / valid | 0.96875 | drafted / pending / valid | not produced (null) |

**Name-accepted after the rerun: 0/10.**

## Run receipt

The complete persisted receipt is:

```text
id=26479cb7-feb9-4a35-89e5-4a99f0988848
started_at=2026-09-02T16:48:16.667710Z
stopped_at=2026-09-02T16:54:09.613359Z
ended_at=2026-09-02T16:55:03.547000Z
stop_reason=no_eligible_work
cost_limit=10.000000
cost_spent=0.000000
cost_total=0.000000
compose_cost=0.000000
review_cost=0.000000
names_reviewed=0
llm_calls=0
sum(LLMCost.llm_cost)=0.000000
```

The command printed a focus scope id `d71954c3-feca-4889-9b0d-ca1b83111bc7`; that ephemeral scope is distinct from the durable `SNRun` receipt above.

## Route failure evidence

The batch command reported `scoped run (d71954c3-feca-4889-9b0d-ca1b83111bc7): 5 names`, but the review pool immediately observed `review_name: 0`, processed 0, and finalized `no_eligible_work`. A post-run positive-control diagnostic loaded the exact committed manifest (355 unique DD source ids), traversed the authored `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)` direction, and found **zero manifest-overlapping live source bindings for every target name**. Representative bindings illustrate the mismatch:

- `beryllium_density_at_plasma_boundary` is bound to `summary/local/separatrix/n_i/beryllium/value`, which is not in the WEST manifest.
- `xenon_density_at_plasma_boundary` is bound to `summary/local/separatrix/n_i/xenon/value`, which is not in the WEST manifest.
- `poloidal_magnetic_flux` has 20 live bindings, but none is the manifest's `core_profiles/profiles_1d/grid/psi`; its current bindings include `core_sources/source/profiles_1d/grid/psi`, `equilibrium/time_slice/profiles_2d/psi`, and other non-manifest paths.

Thus the exact batch selector was not aimed at the ten names. Its reported five-name scope came from other names attached to the manifest, while the ten staged rescores remained outside it. Repeating the same command would not generate fresh quorum evidence.

## Final GraphClient census

The final census used schema-owned `StandardName.id` and `StandardName.name_stage` properties and included key-coverage controls before trusting either zero:

```text
candidates=4691
with_id=4691
with_name_stage=4691
accepted=2335
approved=0
contested=0
```

The instrument was aimed at the `StandardName.name_stage` lifecycle property required by the release contract. Therefore the graph proves **approved 0** and **contested 0**.

## Evidence locations

- Report: `crew/reports/west-withheld-rescore-3.md`
- Auto-rotating CLI log: `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`
- Durable graph receipt: `26479cb7-feb9-4a35-89e5-4a99f0988848`

The only tracked write is this report.
