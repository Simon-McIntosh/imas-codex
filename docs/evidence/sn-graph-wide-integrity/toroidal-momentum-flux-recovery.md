# Toroidal momentum flux interrupted-run recovery

## Outcome

**PASS.** The interrupted `toroidal_momentum_flux` lifecycle was completed
through the ordinary name-review and refinement pools. The identity moved from
`drafted` to `reviewed` on a fresh quorum score of **0.750**, then to
`exhausted` after the three configured refinement attempts each failed closed
at the source-migration compare-and-set. No successor was persisted.

Because both integrity ratchets define a live target as a name whose
`name_stage` is neither `superseded` nor `exhausted`, the eight historical
`PRODUCED_NAME` relationships to `toroidal_momentum_flux` no longer appear in
either live-count surface. The relationships remain as provenance on the
exhausted identity; they were not deleted or rewritten.

| Ratchet query | Before this recovery | After this recovery | Frozen ceiling | Verdict |
|---|---:|---:|---:|---|
| `_MULTIPLE_LIVE_TARGETS_QUERY` | 27 | **23** | 23 | **PASS — equal to ceiling** |
| `_STALE_LIVE_BINDINGS_QUERY` | 7 | **3** | 3 | **PASS — equal to ceiling** |

Both ratchets therefore reach their frozen ceilings exactly. No ceiling
constant was changed.

## Route selection

The ordinary-quorum route was chosen from measured graph state rather than
restoring lifecycle fields manually:

- The pre-run snapshot recorded `toroidal_momentum_flux` as `superseded`.
- The interrupted scoped run had left it `drafted`, valid, unreviewed, with no
  refinement history and no active claim in scope
  `25c083b5-d385-4a0e-bfd0-b0da1e1d1e67`.
- Its earlier deterministic DD-resolution blocker had been repaired before this
  node began.
- A production-quiet check found no active Standard Name claims and no `SNRun`
  whose `stop_reason` was null.
- The exact-name dry run admitted exactly one existing identity and performed
  no graph writes.

Those facts made an ordinary, one-name review/refinement drain the narrowest
sanctioned recovery. It could earn a decision without reseeding sources,
resetting lifecycle state, running global maintenance, or reconstructing
pre-run state by direct graph mutation.

## Before state

Immediately before the bounded run:

- `toroidal_momentum_flux.name_stage = 'drafted'`
- `validation_status = 'valid'`
- `reviewer_score_name = null`
- `refine_attempts = null`, `chain_length = 0`
- no incoming review record was present and no successor existed
- original focus scope:
  `25c083b5-d385-4a0e-bfd0-b0da1e1d1e67`
- global counters: 27,591 `LLMCost` nodes, USD 1,365.422231 total recorded
  cost, and 7,706 `StandardNameChange` nodes

The name had eight incoming source relationships. Four were stale sources whose
scalar still selected `toroidal_momentum_flux`:

1. `dd:core_transport/model/profiles_1d/momentum_tor/flux`
2. `dd:edge_transport/model/ggd/ion/momentum/flux/toroidal`
3. `dd:plasma_transport/model/ggd/momentum/flux/toroidal`
4. `dd:plasma_transport/model/profiles_1d/momentum_tor/flux`

The other four were live composed sources whose scalar selected a different,
accepted target, making the relationship to `toroidal_momentum_flux` the extra
live target counted by the dual-binding ratchet:

1. `dd:edge_sources/source/ggd/neutral/momentum/phi` — scalar
   `toroidal_neutral_momentum_source`
2. `dd:plasma_sources/source/ggd/ion/momentum/phi` — scalar
   `toroidal_ion_torque_density`
3. `dd:plasma_sources/source/ggd/momentum/phi` — scalar
   `toroidal_torque_density`
4. `dd:plasma_sources/source/ggd/neutral/momentum/phi` — scalar
   `toroidal_neutral_torque_density`

The measured partitions were therefore:

- multiple-live targets: **23 standing refusals + 4 interrupted-run rows =
  27**;
- stale-live bindings: **3 standing refusals + 4 interrupted-run rows = 7**.

## Bounded execution

Dry-run command:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH="$PWD" \
uv run --env-file /home/ITER/mcintos/Code/imas-codex/.env --no-sync \
  imas-codex sn run \
  --name toroidal_momentum_flux \
  --names-only \
  --skip-global-maintenance \
  --cost-limit 20 \
  --dry-run
```

The dry run reported exactly one eligible existing name and no graph write.

Live command:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH="$PWD" \
uv run --env-file /home/ITER/mcintos/Code/imas-codex/.env --no-sync \
  imas-codex sn run \
  --name toroidal_momentum_flux \
  --names-only \
  --skip-global-maintenance \
  --cost-limit 20 \
  --time 12
```

- `SNRun`: `11906954-9251-4173-868b-d3687b883175`
- exact-name scope: `d0336150-65b3-4214-9a72-5b2be4221402`
- stop reason: `no_eligible_work`
- elapsed time: 227.569290 s
- names reviewed: 1
- names regenerated: 0
- provider calls: 5
- actual spend: **USD 0.660631 / USD 20.00 authorized**
- unused authorization: **USD 19.339369**

Spend by pool and model:

| Pool | Model | Calls | USD |
|---|---|---:|---:|
| `review_name` | `openrouter/x-ai/grok-4.5` | 1 | 0.084164 |
| `review_name` | `openrouter/openai/gpt-5.6-luna` | 1 | 0.009805 |
| `refine_name` | `openrouter/openai/gpt-5.5` | 2 | 0.153270 |
| `refine_name` | `openrouter/anthropic/claude-fable-5` | 1 | 0.413392 |
| **Total** |  | **5** | **0.660631** |

The quorum's source-fidelity finding was decisive: the spelling is valid for
the transport-flux paths, but the same source closure also includes momentum
source-term paths whose accepted scalar identities distinguish source from
flux. The two reviewer calls resolved by `quorum_consensus` at **0.750**, below
the configured acceptance threshold of 0.85. The name was not accepted.

Each refinement attempt proposed a correction but persistence refused before a
successor could be created. The compare-and-set re-read all eight sources and
found exactly the interrupted closure: four stale sources and four sources
already scalar-bound to a different accepted target. The refusal was repeated
at attempts 1, 2, and 3; the configured attempt budget then moved the existing
identity to:

- `name_stage = 'exhausted'`
- `reviewer_score_name = 0.750`
- `review_resolution_method = 'quorum_consensus'`
- `refine_attempts = 3`
- `refine_stop_reason = 'attempts_exhausted'`
- no successor
- no active claim

This is a decided ordinary-pipeline lifecycle, not a hand acceptance or a
manual promotion. The three compare-and-set refusals are the safety mechanism
working: none of the eight historical bindings was migrated to an unreviewed
successor.

## After state and verification

The same read-only queries used before execution measured:

- `_MULTIPLE_LIVE_TARGETS_QUERY`: **23** rows;
- `_STALE_LIVE_BINDINGS_QUERY`: **3** rows;
- transient `toroidal_momentum_flux` rows in either result: **0**;
- active claims on the name or original focus scope: **0**;
- `LLMCost`: 27,591 to 27,596, exactly **+5**;
- recorded total cost: USD 1,365.422231 to USD 1,366.082862, exactly
  **+USD 0.660631**;
- `StandardNameChange`: 7,706 to 7,706, exactly **0**, consistent with review
  plus refused refinement persistence rather than a source rewrite.

Credentialed graph verification:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH="$PWD" \
uv run --env-file /home/ITER/mcintos/Code/imas-codex/.env --no-sync \
  pytest -m graph tests/graph/test_sn_integrity_ratchets.py -q \
  --timeout=240 -p no:cacheprovider
```

Result: **4 passed, 0 failed, 0 skipped**. In particular, both named ratchet
tests passed against production with their constants unchanged.

## Safety and durable receipts

- No raw Cypher mutation was used. Bespoke graph queries were read-only.
- No `sn edit`, hand acceptance, direct lifecycle promotion, reseed, reset, or
  source detach was used.
- Global startup, background, and post-drain maintenance writes were bypassed
  by the exact scope contract.
- No ceiling constant or test source was edited.
- Exact-name dry-run log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T111439503568-torflux/exact-name-dry-run.log`
  (SHA-256
  `1429c8ab0925af2a93910ab4f18f9c708bf461d2b1f7eb778e654ac9751d7bb2`).
- Pipeline log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T111439503568-torflux/exact-name-run.log`
  (SHA-256
  `6c36db3736a1658d2f0f5fe275b4b7de1059cc689682d22bf019706068e10842`).
- Focused graph-test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T111439503568-torflux/focused-ratchets-after.log`
  (SHA-256
  `f8ab291e7bf020e894a94cdf851ad8b77ce3c6c4e48726b22494dd4510d7d976`).

The requested recovery is complete: the four interrupted stale bindings and
the four interrupted dual-binding rows are absent from the live integrity
surfaces, and both frozen ratchets are exactly at their ceilings.
