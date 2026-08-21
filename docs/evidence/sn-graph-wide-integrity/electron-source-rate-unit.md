# Electron source-rate unit review

NEEDS-HELP: Ordinary review did not settle the identity because the DD-grounded compose attempt exhausted both source claims before producing a canonical, persistable successor.

tried: Reset both pre-service failed sources through `sn retry --failed`, previewed and attempted a DD-grounded rename, then ran a scoped `sn edit --hint` generate-to-review cycle with `--cost-limit 15`; every mutation gate remained fail-closed and the two credentialed graph tests were measured before and after.

options: (1) repair candidate canonicalization so the composer converts the rejected `volumetric_electron_source_rate` ordering to the grammar-reported `electron_volumetric_source_rate`, then repeat the ordinary scoped review; (2) add a sanctioned rename path that derives a successor's unit from the complete DD source cohort instead of inheriting the predecessor's unit; or (3) independently adjudicate and use the attachment operator if the two DD paths should be detached from the volume-integrated structural parent rather than renamed with it.

leaning: Option 1, because it preserves DD-authoritative unit injection and the required ordinary quorum path while addressing the directly observed persistence failure rather than weakening attachment or unit guards.

cost-if-wrong: The canonicalization repair and its tests would have to be reverted or superseded, both failed sources reset again, and the exact scoped compose/review repeated; no accepted identity or unit would need to be undone because this run accepted nothing.

## Outcome

Status: **blocked, fail-closed**. The required pass condition was not met.

The authoritative DD evidence resolves the physical contradiction but the ordinary pipeline did not persist a reviewable replacement:

- `core_sources/source/profiles_1d/electrons/particles_decomposed/explicit_part` is a `FLT_1D` explicit source term added directly to the electron-density transport equation. Its scalar property and `HAS_UNIT` edge both declare `m^-3.s^-1`.
- `edge_sources/source/ggd/electrons/particles/values` is a `FLT_1D` source term for the electron-density equation on grid subsets. Its scalar property and `HAS_UNIT` edge both declare `m^-3.s^-1`.
- `electron_source_rate` instead declares and links `s^-1`, and its description and documentation explicitly define a volume integral. It therefore does not identify either attached DD quantity.
- Before intervention the name was `reviewed`, valid, scored `0.68125`, and both attached sources were `failed` at `attempt_count=5` from the unavailable pre-service compose attempt.

This is a volumetric-rate identity problem, not a DD unit defect: the two independent DD declarations, their unit edges, their transport-equation descriptions, and their array-on-grid representations agree. No unit property or unit edge was edited.

## Sanctioned execution record

1. The credentialed baseline selection ran both required tests and exited **1**, not 5: **2 failed / 0 passed** in 12.81 s. Both failures named exactly the two `electron_source_rate (s^-1)` to DD `m^-3.s^-1` attachments.
2. `sn retry --failed --dry-run` selected **2 of 2** sources. The applied retry wrote the normal retry ledger and reset **2 of 2** sources.
3. `sn edit electron_source_rate --rename volumetric_electron_source_rate ... --dry-run` produced a valid one-name preview. The applied rename exited **2** and rolled back because rename inherits the predecessor's `s^-1` unit; the attachment guard rejected both proposed pairings against `m^-3.s^-1`.
4. A name-axis `sn edit --hint` was then previewed and run with exact scope `sn-edit-20260821T115815Z` and `--cost-limit 15`. It reset both producing sources to `extracted` and entered the ordinary generate-to-review pipeline.
5. The composer repeatedly emitted `volumetric_electron_source_rate`; the strict ISN parser refused it and reported the canonical flat order as `electron_volumetric_source_rate`. A transient `net_electron_source_rate` proposal also failed the source-migration compare-and-set while the source closure contained both the old and proposed bindings. Those transient writes did not survive postflight.
6. The run exited **3** with `electron_source_rate` still below threshold at its prior score `0.68`; it created **0 fresh review rows**, accepted **0 names**, and reported **$0.0000 actual spend / $15.00 authorized cap**. Both sources returned to `failed`, `attempt_count=5`, with `last_error="compose claim-attempt cap reached"`.
7. The credentialed after selection again exited **1**, not 5: **2 failed / 0 passed** in 20.15 s, with the same two unit disagreements. Current postflight found no surviving `net_electron_source_rate`, `electron_volumetric_source_rate`, or `volumetric_electron_source_rate` node.

No name was hand-accepted or hand-promoted. No raw Cypher mutation was used. No name unit, DD unit, or attachment was hand-edited. Acceptance remains gated on a fresh quorum score.

## Logs and read-only evidence

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/before-tests.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/graph-before.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/identity-evidence.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/retry-dry-run.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/retry-apply.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/edit-dry-run.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/edit-review.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/hint-dry-run.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/hint-review.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/graph-after-attempt.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T115003549608-esrate/after-tests.log`
