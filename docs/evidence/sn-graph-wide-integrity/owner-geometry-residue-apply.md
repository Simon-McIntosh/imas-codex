# Owner-geometry ordinary-source migration evidence

## Outcome

**COMPLETE.** The ordinary-source migration program applied all three exact
owner-geometry source dispositions in one production transaction. The signed
cohort was unchanged semantically from the adjudication: the same three source
ids and the same three authoritative target identities were used. The current
authority builder emitted the canonical 1–3 mutation sequence and explicit
admitted-row receipt rule required by the landed program.

The preview disposition was **3 admitted + 0 refused = 3 signed rows**. Because
there were no refusals, the requirement that every refusal carry its verbatim
reason is satisfied with an empty refusal list. The apply changed **3 rows**
and wrote **3 `StandardNameChange` receipts**. The program read the live
`StandardNameChange` baseline inside the applying transaction and permits
commit only when its delta equals the receipt cardinality; the observed live
counter was **7,780 → 7,783, delta 3**, exactly equal to `receipt_rows=3`.

Replay returned `already_applied` with **changed=0** and
**persistent_writes=0**. The driver then raised a false-negative assertion
after that return because the evidence-only check used `value or -1`: Python
therefore converted each correct numeric zero to `-1`. In accordance with the
two-failure stop rule, no apply was run again. A separate read-only recovery
audit proved the committed transaction, the replay receipt closure, and zero
additional graph writes.

| Required measure | Observed result |
|---|---:|
| Signed authority rows | **3** |
| Preview admitted | **3** |
| Preview refused | **0** |
| Admitted + refused | **3** |
| Apply outcome | **applied** |
| Mutated rows (`changed`) | **3** |
| Receipt rows | **3** |
| `StandardNameChange` baseline | **7,780** |
| `StandardNameChange` after apply | **7,783** |
| `StandardNameChange` delta | **3** |
| Delta equals receipt rows | **yes: 3 = 3** |
| Replay outcome | **already_applied** |
| Replay changed | **0** |
| Replay persistent writes | **0** |
| 49 signed owner rows found after apply | **49** |
| 49 signed owner rows still selecting the old target | **2** |
| All live producers still selecting the old target | **25** |
| Recovery `StandardNameChange` delta | **0** |
| Recovery `PRODUCED_NAME` delta | **0** |
| Recovery `LLMCost` delta | **0** |

## Exact migrated identities

| DD source | Before | Authoritative identity after apply | Target review state | Postcondition |
|---|---|---|---|---|
| `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi` | `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_active_spatial_resolution_zone` | accepted, valid, name score **0.99375** | edge, scalar, and backing projection all select the authoritative identity |
| `dd:spectrometer_visible/channel/detector/centre/phi` | `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_detector` | accepted, valid, name score **1.00000** | edge, scalar, and backing projection all select the authoritative identity |
| `dd:spectrometer_visible/channel/polarizer/centre/phi` | `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_polarizer` | accepted, valid, name score **1.00000** | edge, scalar, and backing projection all select the authoritative identity |

All three sources remain `status='attached'` with both claim fields absent.
The two remaining members of the signed 49-row owner cohort that select
`toroidal_angle_of_measurement_position` are therefore the already-adjudicated
field-map-grid and reflector-center refusals, not incomplete members of this
three-row migration.

## Signed authority and exact receipt proof

The authority was emitted by `build_repair_authority` and executed by
`apply_signed_manifest` through the registered ordinary-source migration
program.

- Authority file SHA-256:
  `fc3bfcdbec42f30a639350ab519fe6e02eb8044bb066c6e604dc717a9736af32`
- Canonical signed-payload SHA-256:
  `065a0b52a89434fe7d61cd4d056df036ef3f62e11fd09db28bbe7e3585566d27`
- Applying run id: `owner-geometry-residue-apply`
- Authorized manifest SHA-256:
  `0e61392401e47df29c74f3eb99c23ad0cbc05008c94e6060fe6c256c8eff012e`

The receipt audit did not infer success from an operation spelling or a bare
global counter. It matched `StandardNameChange` nodes on the exact pair
`run_id='owner-geometry-residue-apply'` and
`manifest_sha256='0e61392401e47df29c74f3eb99c23ad0cbc05008c94e6060fe6c256c8eff012e'`.
That query returned exactly these three row ids:

1. `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi`
2. `dd:spectrometer_visible/channel/detector/centre/phi`
3. `dd:spectrometer_visible/channel/polarizer/centre/phi`

Every receipt independently carries the authority file digest, signed-payload
digest, exact manifest digest, and the complete three-id admitted cohort. A
run-id-only audit also returned exactly these same three receipts and one
manifest digest, ruling out a hidden receipt under another manifest.

## Apply and replay chronology

The first driver launch stopped before authority construction or mutation when
a read-only preflight query referenced an incorrect local alias. The corrected
launch then:

1. read the live three-source closure and the live counter baseline;
2. rebuilt the canonical signed authority from the unchanged three semantic
   rows;
3. previewed all three rows as admitted;
4. applied all three migrations and verified receipt cardinality and the live
   counter delta;
5. queried the receipts by exact run id and manifest digest;
6. called replay, which returned the already-applied, zero-write receipt; and
7. raised only in the evidence harness's subsequent zero-value assertion.

The false-negative assertion is outside the production migration program and
ran after replay returned. The read-only recovery audit subsequently observed
identical before/after counters:

```text
StandardNameChange  7,783 -> 7,783
PRODUCED_NAME        5,779 -> 5,779
LLMCost             27,631 -> 27,631
exact run+manifest receipts  3 -> 3
```

No third apply or replay was attempted.

## Verification

The focused ordinary-source loader contract completed with **1 passed,
5 graph cases deselected, 0 failed**. The graph cases require an explicitly
configured disposable Neo4j endpoint and were not pointed at production. The
production transaction supplied the live closure, receipt, counter, and replay
evidence above.

## Durable artifacts

- Builder-emitted signed authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T002535030026-ownerapply2/owner-geometry-source-migration-authority.json`
  (SHA-256 `fc3bfcdbec42f30a639350ab519fe6e02eb8044bb066c6e604dc717a9736af32`).
- Read-only exact receipt and postcondition audit:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T002535030026-ownerapply2/owner-geometry-source-migration-recovery.json`
  (SHA-256 `c84bbb1d4d7551854667cfbb59388e91a47973248d45a6bb3a7c82d00733cc35`).
- Applying launch and replay false-negative log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T002535030026-ownerapply2/owner-geometry-source-migration-retry.log`
  (SHA-256 `1e024e6d68eed7396aba7afcab589590471196fae2ccc5d0e1271e5da806bd93`).
- Initial preflight-only alias failure log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T002535030026-ownerapply2/owner-geometry-source-migration.log`.
- Focused contract-test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T002535030026-ownerapply2/ordinary-source-migration-tests.log`
  (SHA-256 `efd71c2866cfe62e298df02242a2fc49eb7c5e7f10975ba19417648f4d915785`).
