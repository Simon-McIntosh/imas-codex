# Exhausted orphan supersession

## Result

The exact-manifest operator superseded all 296 fresh non-derived exhausted
Standard Names that had no live `PRODUCED_NAME` source. Every name received its
own `StandardNameChange` receipt. No accepted, reviewed, pending, drafted, or
derived-origin name was admitted.

The fresh cohort was larger than the approximate 192-row expectation. The
signed census resolved the difference as 96 `catalog_edit`, 96 `pipeline`, and
104 null-origin legacy rows. All 296 satisfied the locked lifecycle and source
predicate; there were zero refusals. Representative retired identities include
`absorbed_plasma_heating_power`, `atomic_mass_of_pellet`,
`brightness_of_soft_xray_detector`, and `ion_state_charge_number`. Each had no
live source-path binding at preview time.

## Signed execution

- Manifest SHA-256:
  `a9743fc54b84da87acfc9ea35592412cf131a800937cb3c4519038d65015452f`
- Preview: 296 requested, 296 admitted, 0 refused.
- Apply: 296 names superseded and 296 per-name ledger rows created.
- Ledger verification: 296 participants each had exactly one matching receipt;
  all 296 ended at `name_stage=superseded`, `status=superseded`, with zero live
  producing sources.
- `StandardNameChange` count: 7,155 before and 7,451 after, an exact delta of
  296. `LLMCost` remained 27,467 and `SNRun` remained 489. Provider calls: 0.
- Replay: `already_applied`, `changed=0`, `persistent_writes=0`.
- Replay participant snapshot SHA-256, before and after:
  `05d690fd8395bb28b4620b07bff3618b000650cb892cce1e5e0f556ef484ab9c`.

The post-apply census found zero non-derived exhausted orphans. Other source-less
lifecycle populations were unchanged: 69 accepted, 4 reviewed, 8 pending, and
4 drafted. The accepted total includes 36 derived parents, while the pending
total includes 7 derived parents; none was mutated.

## Transaction and regression evidence

The disposable-Neo4j suite contains five cases: one successful exhausted-orphan
transition, three refusal shapes (live source, accepted lifecycle, and derived
origin), and an exact participant-snapshot replay check. The initial red run
failed during collection because the public operator was absent. After the
implementation, all five cases passed.

Full logs:

- `red-proof.log`: expected pre-implementation collection failure.
- `pytest-green.log`: five disposable-Neo4j cases passed.
- `live-preview.log`: fresh signed cohort and complete name-id list.
- `live-apply.log`: apply receipts, ledgers, replay counters, and before/after
  censuses.

The logs are stored in
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T005245363468-orphan-exhausted-supersede/`.
