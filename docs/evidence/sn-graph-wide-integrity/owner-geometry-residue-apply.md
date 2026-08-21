NEEDS-HELP: the three authoritative rows are old-target-only, but the signed envelope has no closed program for an ordinary source migration.

# Owner-geometry residue apply blocker

## Outcome

**BLOCKED before mutation.** A fresh production-graph invocation derived the
three exact adjudicated rows, built a canonical signed repair authority through
`build_repair_authority`, and stopped when the signed-manifest loader rejected
the first row verbatim:

> `repair row 'dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi' is not a closed source-target reconciliation program`

This is a program-registry gap, not a lifecycle or semantic-authority refusal.
All three intended targets exist, are `accepted` and `valid`, and retain the
review scores recorded by the adjudication. The three sources are not
dual-bound: each has exactly one live `PRODUCED_NAME` edge, one scalar mirror,
and one backing projection, all selecting
`toroidal_angle_of_measurement_position`. The registered reconciliation
program can delete losing relationships only when the signed survivor is
already a live relationship. The registered relationship-add program is
restricted to stale `derived:<target>` structural sources. Neither program can
express these three ordinary DD-source migrations.

| Required measure | Observed result |
|---|---:|
| Exact adjudicated source rows | **3** |
| Rows emitted by `build_repair_authority` | **3** |
| Builder file SHA-256 | `9e2e9606b2e72317033b28026877e52b4af224c36c7c68bd016f244894aa039f` |
| Builder canonical payload SHA-256 | `af0ce5683bdc01a2503c0f0969f7a3ed53a01cf2f2e9935811935b3cfb57c0e8` |
| Signed-loader classification | **Unavailable: closed-program validation refused before preview** |
| Apply / replay | **Not run** |
| `StandardNameChange` | **7,778 → 7,778 (delta 0)** |
| `PRODUCED_NAME` relationships | **5,768 → 5,768 (delta 0)** |
| `LLMCost` rows | **27,631 → 27,631 (delta 0)** |
| Exact run-id + digest receipts | **0** |
| All receipts with the exact run id | **0** |
| Production graph mutations | **0** |
| 49-row authority still selecting the old target | **5** |
| All live producers of the old target | **28** |

The done-when is therefore not met: there is no honest `admitted + refused =
3` apply result, no receipt cardinality, no apply delta, and no replay receipt.
Inventing those fields from a direct call to the older retarget helper would
misrepresent an unsigned operation as the required signed-envelope apply.

## Exact current rows

| DD source | Live binding before/after | Authoritative target | Target lifecycle and score | Disposition |
|---|---|---|---|---|
| `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi` | `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_active_spatial_resolution_zone` | accepted, valid, **0.99375** | blocked before mutation |
| `dd:spectrometer_visible/channel/detector/centre/phi` | `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_detector` | accepted, valid, **1.00000** | blocked before mutation |
| `dd:spectrometer_visible/channel/polarizer/centre/phi` | `toroidal_angle_of_measurement_position` | `toroidal_coordinate_of_polarizer` | accepted, valid, **1.00000** | blocked before mutation |

For every row, `source.status='attached'`, both claim fields are null, the
complete live target set is the singleton old target, and the complete backing
projection set is the same singleton. This rules out drift, an active writer,
or target lifecycle as the cause of the refusal.

The other two residual owner/geometry rows remain the already-adjudicated named
refusals. The post-diagnostic 49-row authority count of **5** old-target scalar
selections is exactly the three blocked migrations plus those two refusals.

## Why the registered programs cannot execute this authority

The repair builder correctly owns the signed schema, selection literal, row
projection, receipt count rule, signature, and both digests. It does not invent
an execution program. The loader separately validates every mutation sequence
against the closed registry.

The source-target reconciliation validator requires:

- at least two signed `StandardName` participants;
- one live `PRODUCED_NAME` relationship participant for every signed target;
- exactly one retained relationship to the signed survivor;
- deletion of all other relationship participants; and
- a scalar update to that already-bound survivor.

The live three-row cohort instead has one old relationship per source and no
relationship to the authoritative target. A faithful migration row therefore
needs to delete the old relationship, add the authoritative relationship, and
update the scalar. The loader refuses that mixed program because the survivor
is not already bound.

The only registered `add_relationship` program is structural-source revival.
It requires the source identity to be exactly `derived:<target>`, requires a
supporting `HAS_PARENT` relationship, and fixes a derived-source lifecycle
payload. These three `dd:` sources deliberately fail that shape. Generalizing
that structural exception ad hoc would erase the authority boundary between
ordinary DD attachment migration and derived-parent recovery.

The existing `retarget_standard_name_sources` helper does implement the needed
atomic graph mechanics: exclusive source-edge retarget, scalar mirror,
`HAS_STANDARD_NAME` backing projection, and both names' `source_paths` mirrors.
It is nevertheless not an adequate substitute for this node's contract. Its
receipt is the legacy `source_migration_manifest` event keyed by its own
internal per-target payload; it does not execute the emitted repair authority
and does not persist the signed envelope's `manifest_sha256`. Calling it would
make the demanded run-id-and-manifest receipt proof impossible without an
unauthorized raw receipt rewrite.

## Zero-mutation proof

The diagnostic read the live baseline immediately before authority
construction and read the same counters after the loader refusal. All three
counters were identical:

```text
StandardNameChange  7,778 -> 7,778
PRODUCED_NAME        5,768 -> 5,768
LLMCost             27,631 -> 27,631
```

The receipt proof did not filter or infer by an operation label. It queried the
exact intended run id, `owner-geometry-residue-apply`, together with the
builder's signed digest in the receipt `manifest_sha256` field, and returned
`receipt_rows=0`, `receipt_ids=[]`. A second query over the exact run id without
any operation predicate also returned zero rows. Thus no receipt is hidden
under a different operation spelling. The loader rejection occurs before an
applying transaction or mutation dispatch.

## Tried

1. Re-read the live plan and the landed three-row semantic adjudication.
2. Re-read all three production source closures and all three target lifecycle
   records.
3. Constructed the only faithful typed mutation sequence for each row through
   `build_repair_authority`: delete incumbent binding, add authoritative
   binding, update scalar, with last-producer and collateral guards.
4. Passed the emitted bytes and both recomputed digests to the signed loader.
   It refused the ordinary migration shape before graph execution.
5. Re-read counters, exact run/digest receipts, the full 49-row authority
   residue, and the all-producer old-target count.

The focused builder and source-target contract suite is green at **15 passed,
3 graph cases deselected, 0 failed**. The deselections reflect the repository's
default exclusion of graph-marked disposable-Neo4j cases; the closed-registry
unit contract and builder cases executed.

## Options

1. **Add one closed ordinary-source migration program to the signed envelope.**
   Its validator should require one exact old binding and accepted-valid new
   target, and its executor should carry the existing retarget helper's edge,
   scalar, backing projection, `source_paths`, claim, last-producer, closure,
   receipt, and replay semantics. Cover old-only apply, attachment refusal,
   drift refusal, last-producer refusal, exact receipt attribution, and
   write-free replay on disposable Neo4j. Then redispatch this three-row apply.
2. Authorize the existing exact retarget helper as the execution vehicle and
   explicitly amend the evidence contract to accept its three internal
   per-target migration hashes instead of signed-envelope `manifest_sha256`
   receipts. This is smaller code-wise but weakens the newly locked generic
   authority boundary and contradicts the stated reuse decision.
3. Add the three target relationships in a preliminary signed operation and
   run reconciliation second. This requires two applies, temporarily creates
   dual authority, and violates the one-invocation/one-apply measure.

## Leaning

Choose option 1. It preserves the signed-envelope direction, gives the
three ordinary DD migrations a real closed program instead of stretching the
derived-source exception, and makes the required exact run-id plus manifest
receipt and write-free replay natively provable.

## Cost if the wrong path is chosen

No production repair must be undone now: this node made **zero mutations**. If
option 2 or 3 is chosen and later rejected, the authority driver, receipt
interpretation, and replay evidence must be redone; option 3 may additionally
require a governed cleanup of temporarily dual-bound sources. Choosing option
1 requires an out-of-scope production-and-test change before rerunning this
unchanged three-row semantic authority.

## Durable evidence

- Machine result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T222629644810-ownerapply/owner-geometry-apply-blocker.json`
  (SHA-256 `8c2842bb4069a09794da9ca921ab6bd27f8ceca186d71d782449cf4f154c137f`).
- Builder-emitted authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T222629644810-ownerapply/owner-geometry-source-migration-authority.json`
  (SHA-256 `9e2e9606b2e72317033b28026877e52b4af224c36c7c68bd016f244894aa039f`).
- Complete diagnostic log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T222629644810-ownerapply/owner-geometry-apply-diagnostic.log`
  (SHA-256 `8427d6975d33d5627ad4be69ec389c0452a6e1b5147206ff9bcfc624f50ed2ab`).
- Focused contract-test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T222629644810-ownerapply/focused-contract-tests.log`
  (SHA-256 `d1747b0d6fe831fe2a46d2e9f02139765e6cb4b1a6b40dabee12db670c4285af`).
- Source checkout before this evidence commit:
  `a53104b904fed553a10ecf4967de10c3bb6d93d3`.
